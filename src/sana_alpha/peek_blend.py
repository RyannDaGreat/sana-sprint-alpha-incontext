"""
The learned blend, sneak peek highway v2 (manifest §18, m00901): instead of the fitted table
x̂0_show = w·h + (1 − w)·p, each block's preview is learned from the head's own Tweedie estimate h, the previous step's
x̂0 p and noise prediction e, the noisy latent x_t and the timestep conditioning c:

    x̂0_show = p + g_k(c) ⊙ (h − p) + R_k([h; p; e; x_t]),   g_k(c) = sigmoid(a_k + G_k·c),   e = (x_t − (1 − σ)·p)/σ

Step 1 has no previous step: p = 0 (the x_t-only Tweedie estimate at σ = 1). R_k's last layer starts at zero and g is
fitted to the shipped table first, so the untrained blend is the table blend. Trained offline on sampler trajectories
(collect -> train -> evaluate): the trajectories give exactly the inputs the app has.
Usage: CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src uv run --with lpips --with pytorch-msssim python -m sana_alpha.peek_blend run \
           models/fibo2/fibo2_split/step_004000 agents/.scratchpad/peekcalib/turbo4_step004000.json OUT_DIR
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

BLEND_FILE = "sneak_peek_blend.safetensors"  # in a step folder, next to sneak_peek.HIGHWAY_FILE
HIDDEN = 128  # R_k's hidden width: the correction is a small per-token MLP, cheap next to the head (3584 -> 128)
GATE_CLIP = 1e-3  # the table's w = 1 at step 1 has no finite logit: fit g to w clipped to [GATE_CLIP, 1 − GATE_CLIP]


class LearnedBlend(torch.nn.Module):
    """All blocks' blend parameters stacked (K, ...), so one graph serves every block (chosen by an index input)."""

    def __init__(self, n_blocks, c_dim=256, ch=128, hidden=HIDDEN):
        """Command (allocates parameters). A fresh blend has g = 1/2 and R = 0; fit_gate sets g to a table."""
        super().__init__()
        k, z = n_blocks, torch.zeros
        self.a, self.G = torch.nn.Parameter(z(k, ch)), torch.nn.Parameter(z(k, ch, c_dim))
        self.w1 = torch.nn.Parameter(torch.randn(k, 4 * ch, hidden) / (4 * ch) ** 0.5)
        self.b1, self.w2, self.b2 = torch.nn.Parameter(z(k, hidden)), torch.nn.Parameter(z(k, hidden, ch)), torch.nn.Parameter(z(k, ch))

    def forward(self, block, h, p, xt, sigma, c):
        """
        Pure function (given the parameters). One preview per batch row.

        Args:
            block (torch.Tensor): (B,) int64 block indices
            h, p, xt (torch.Tensor): (B, N, 128) tokens: the head's x̂0, the previous step's x̂0 (0 at step 1), x_t
            sigma (torch.Tensor): (B,) the step's noise level
            c (torch.Tensor): (B, D) the timestep conditioning the final layer reads

        Returns:
            torch.Tensor: (B, N, 128) the shown x̂0

        Examples:
            >>> m = LearnedBlend(2, c_dim=3, ch=2, hidden=4)
            >>> h, p, xt = torch.ones(1, 5, 2), torch.zeros(1, 5, 2), torch.randn(1, 5, 2)
            >>> m(torch.tensor([1]), h, p, xt, torch.tensor([0.5]), torch.randn(1, 3))[0, 0].tolist()  # g = 1/2, R = 0
            [0.5, 0.5]
        """
        s = sigma.view(-1, 1, 1)
        e = (xt - (1 - s) * p) / s  # the previous step's ε̂: the Euler step keeps x_t = (1 − σ)·x̂0 + σ·ε̂
        # batched matmuls, not einx: einx traces N as a constant into the ONNX graph (the app's N varies with the size)
        g = torch.sigmoid(self.a[block] + (self.G[block] @ c[..., None])[..., 0])
        z = torch.cat([h, p, e, xt], -1)
        r = torch.nn.functional.gelu(z @ self.w1[block] + self.b1[block][:, None])
        r = r @ self.w2[block] + self.b2[block][:, None]
        return p + g[:, None] * (h - p) + r


@torch.no_grad()
def fit_gate(blend, w, c):
    """
    Command (sets blend.a, blend.G; R untouched). Least squares in logit space so that g_k(c_j) ≈ w_{j,k} for every
    row j; with no more rows than c's dimension + 1 (e.g. 4 steps) the min-norm solution is exact.

    Args:
        w (torch.Tensor): (J, K) table weights, one row per conditioning row (e.g. per step)
        c (torch.Tensor): (J, D) the conditioning of each row

    Examples:
        >>> m = LearnedBlend(2, c_dim=3, ch=2, hidden=4)
        >>> c = torch.eye(3)[:2]
        >>> fit_gate(m, torch.tensor([[0.2, 0.9], [0.6, 0.5]]), c)
        >>> [round(v, 3) for v in torch.sigmoid(m.a[1] + m.G[1] @ c[0]).tolist()]
        [0.9, 0.9]
    """
    logit = torch.logit(w.float().clamp(GATE_CLIP, 1 - GATE_CLIP))
    a1 = torch.cat([torch.ones(len(c), 1, device=c.device), c.float()], 1)  # (J, D + 1)
    x = torch.linalg.pinv(a1) @ logit  # (D + 1, K)
    blend.a.copy_(x[0][:, None].expand_as(blend.a))
    blend.G.copy_(x[1:].T[:, None, :].expand_as(blend.G))


def save_blend(blend, folder):
    """Command. Writes the blend to folder/BLEND_FILE."""
    save_file({k: v.detach().float().cpu().contiguous() for k, v in blend.state_dict().items()}, Path(folder) / BLEND_FILE)


def load_blend(folder):
    """Query. The LearnedBlend saved in folder/BLEND_FILE, or None when there is none (a v1 highway)."""
    path = Path(folder) / BLEND_FILE
    if not path.exists():
        return None
    sd = load_file(path)
    k, ch, d = sd["G"].shape
    blend = LearnedBlend(k, d, ch, sd["w1"].shape[2])
    blend.load_state_dict(sd)
    return blend


@torch.no_grad()
def collect(model, highway, tap, items, seed0, steps, lams, adapter):
    """
    Query (runs the GPU; merges the λ adapter into the model, leaving it at the last λ). Sampler trajectories of the
    shipped schedule (no CFG, as the turbo app runs): per (item, λ), every step's head estimates h for every block, x_t,
    σ, c and the full model's x̂0, all as (N, 128) tokens (h in float16, on the CPU).

    Returns:
        list[dict]: h (S, K, N, 128) f16, xt (S, N, 128), x0 (S, N, 128) the step-end x̂0s, sigma (S,), c (S, D), lam
    """
    from sana_alpha import fibo2
    t = model["transformer"]
    passes, sig = [], [0.0]
    orig = t.forward

    def fwd(*a, **k):
        sig[0] = float(k["timestep"][0])
        return orig(*a, **k)

    def keep(module, args, output):
        s, xt = sig[0], tap.xt_tokens.float()
        h = torch.stack([xt - s * highway.predict(b, tap.image_tokens(b), tap.c, xt_tokens=xt) for b in range(len(highway.heads))])
        passes.append(dict(h=h[:, 0].half().cpu(), xt=xt[0].cpu(), x0=(xt - s * output[..., :128].float())[0].cpu(), sigma=s, c=tap.c[0].float().cpu()))

    t.forward = fwd
    hook = t.final_layer.register_forward_hook(keep)
    tap.capturing = True
    out, merged = [], 0.0
    try:
        for lam in lams:
            if lam != merged:  # λ = 1: the adapter merged at 1 (m00834), λ in increasing order
                fibo2.merge_lora(t, load_file(adapter), scale=lam - merged)
                merged = lam
            for i, (caption, trimask) in enumerate(items):
                passes.clear()
                fibo2.generate_images(model, [caption], trimask[None], [seed0 + i], steps, 1.0)
                tap.clear()
                assert len(passes) == steps, (len(passes), steps)
                out.append(dict(h=torch.stack([p["h"] for p in passes]), xt=torch.stack([p["xt"] for p in passes]),
                                x0=torch.stack([p["x0"] for p in passes]), sigma=torch.tensor([p["sigma"] for p in passes]),
                                c=torch.stack([p["c"] for p in passes]), lam=lam))
                print(f"collect λ={lam} item {i + 1}/{len(items)}", flush=True)
    finally:
        hook.remove()
        t.forward = orig
        tap.capturing = False
    return out


def previous_x0(traj):
    """Pure function. (S, N, 128) p per step: 0 at step 1, else the previous step's full-model x̂0."""
    return torch.cat([torch.zeros_like(traj["x0"][:1]), traj["x0"][:-1]])


def table_blend(traj, weights):
    """
    Pure function. The shipped v1 preview for every (step, block): w·h + (1 − w)·p (w = 1 at step 1).

    Returns:
        torch.Tensor: (S, K, N, 128)
    """
    w = torch.tensor(weights, dtype=torch.float32).view(*traj["h"].shape[:2], 1, 1)
    p = previous_x0(traj)[:, None]
    return w * traj["h"].float() + (1 - w) * p


def train(blend, trajs, steps_n=4000, batch=8, lr=1e-3, device="cuda"):
    """
    Command (trains blend in place). Adam on the latent MSE of x̂0_show against each generation's final latent, on
    random (trajectory, step, block) samples. Prints the loss every 250 iterations.
    """
    blend.to(device).train()
    opt = torch.optim.Adam(blend.parameters(), lr=lr)
    n_s, n_k = trajs[0]["h"].shape[:2]
    gen = torch.Generator().manual_seed(0)
    run = 0.0
    for it in range(steps_n):
        ti, si, ki = (torch.randint(n, (batch,), generator=gen) for n in (len(trajs), n_s, n_k))
        rows = [trajs[i] for i in ti.tolist()]
        h = torch.stack([r["h"][s, k] for r, s, k in zip(rows, si.tolist(), ki.tolist())]).to(device).float()
        p = torch.stack([previous_x0(r)[s] for r, s in zip(rows, si.tolist())]).to(device)
        xt = torch.stack([r["xt"][s] for r, s in zip(rows, si.tolist())]).to(device)
        sig = torch.stack([r["sigma"][s] for r, s in zip(rows, si.tolist())]).to(device).float()
        c = torch.stack([r["c"][s] for r, s in zip(rows, si.tolist())]).to(device)
        target = torch.stack([r["x0"][-1] for r in rows]).to(device)
        loss = ((blend(ki.to(device), h, p, xt, sig, c) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run += loss.item()
        if (it + 1) % 250 == 0:
            print(f"train it {it + 1}/{steps_n} loss {run / 250:.5f}", flush=True)
            run = 0.0
    blend.eval()


@torch.no_grad()
def apply_blend(blend, traj, device="cuda"):
    """Query (GPU). (S, K, N, 128) the learned preview of every (step, block) of one trajectory."""
    n_s, n_k = traj["h"].shape[:2]
    p, out = previous_x0(traj), []
    for s in range(n_s):
        ks = torch.arange(n_k, device=device)
        rep = lambda x: x.to(device).float()[None].expand(n_k, *x.shape)
        out.append(blend(ks, traj["h"][s].to(device).float(), rep(p[s]), rep(traj["xt"][s]),
                         traj["sigma"][s].to(device).float().expand(n_k), rep(traj["c"][s])).cpu())
    return torch.stack(out)


def latent_mse(previews, traj):
    """Pure function. (S, K) mean squared error of each preview's tokens against the final latent."""
    return ((previews.float() - traj["x0"][-1].float()) ** 2).mean((-1, -2))


def run(checkpoint, calib_json, out_dir, base="turbo", n_train=48, n_eval=36, train_iters=4000,
        fast="agents/.scratchpad/fastdec_ship/r5a3_long16.onnx",
        adapter="models/fibo2/fibo2_adapter/step_001000/lora.safetensors"):
    """
    Command (GPU; writes out_dir/{BLEND_FILE, results.json} and prints). The pilot of §18's learned blend against the
    shipped table: collect train and held-out trajectories at λ = 0 and λ = 1, fit g to the table, train, then score every
    (step, block) preview of the held-out items by LPIPS (alex) and SSIM after the shipped fast decoder, against the
    full decoder's final image: raw head (no blend), table blend (shipped v1), learned blend at init and trained (v2).
    """
    import lpips
    import onnxruntime as ort
    from pytorch_msssim import ssim
    from sana_alpha import fibo2, sneak_peek as sp
    from sana_alpha.eval_set import load_eval_set

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = json.loads(Path(calib_json).read_text())
    weights = calib["decoders"]["fast_decoder"]["weights"]
    steps = calib["steps"]
    model = fibo2.load_eval_model(checkpoint, "cuda", fibo2.TURBO_DIR if base == "turbo" else fibo2.FIBO2_DIR)
    t = model["transformer"]
    tap, highway = sp.HighwayTap(t), sp.SneakPeekHighway(t)
    assert sp.load_highway(highway, Path(checkpoint)), f"{checkpoint}: no trained highway"
    highway.eval()
    split = {"train": [], "held-out": []}
    for m in ("morph_band", "partial_view"):  # calib.py's held-out items, in its order
        d = load_eval_set(Path("resources/showcase_trimask") / m)
        for k, it in enumerate(d["items"]):
            split[it["split"]].append((it["caption"], d["mattes"][k]))
    held = split["held-out"][:n_eval]
    tr = split["train"][: n_train // 2] + split["train"][len(split["train"]) // 2:][: n_train - n_train // 2]  # both mask types
    lams = [0.0, 1.0]
    # held-out first, λ = 0 then 1 (calib.py's seeds); then train items (λ is already 1: merge back to 0 first)
    trajs_eval = collect(model, highway, tap, held, 3000, steps, lams, adapter)
    fibo2.merge_lora(t, load_file(adapter), scale=-1.0)
    trajs_train = collect(model, highway, tap, tr, 5000, steps, lams, adapter)
    vae, bn = model["vae"], model["bn"]
    del t, tap, highway
    model = {"vae": vae, "bn": bn}
    torch.cuda.empty_cache()

    n_k = trajs_eval[0]["h"].shape[1]
    blend = LearnedBlend(n_k, trajs_eval[0]["c"].shape[1])
    all_c = torch.cat([tj["c"] for tj in trajs_train])  # (J·S, D)
    fit_gate(blend, torch.tensor(weights, dtype=torch.float32).repeat(len(trajs_train), 1), all_c)
    init = LearnedBlend(n_k, all_c.shape[1])
    init.load_state_dict(blend.state_dict())
    init.cuda().eval()
    train(blend, trajs_train, steps_n=train_iters)
    save_blend(blend, out_dir)

    sess = ort.InferenceSession(fast, providers=["CUDAExecutionProvider"])
    name = sess.get_inputs()[0].name
    LP = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    side = int(round(trajs_eval[0]["h"].shape[2] ** 0.5))
    lat = lambda tok: fibo2.unpatchify(tok.transpose(-1, -2).reshape(-1, 128, side, side))

    @torch.no_grad()
    def score(tokens, ref):
        """Query (GPU). (M, N, 128) tokens -> (M,) LPIPS and (M,) SSIM vs ref (1, 3, H, W) in [0, 1]."""
        lp_, ss_ = [], []
        for i in range(0, len(tokens), 8):
            x = lat(tokens[i:i + 8].float())
            img = torch.cat([torch.from_numpy(sess.run(None, {name: x[j:j + 1].numpy()})[0]) for j in range(len(x))]).cuda().clamp(0, 1)
            r = ref.expand(len(img), -1, -1, -1)
            lp_.append(LP(img * 2 - 1, r * 2 - 1).flatten().cpu())
            ss_.append(ssim(img, r, data_range=1.0, size_average=False).cpu())
        return torch.cat(lp_), torch.cat(ss_)

    variants = ("raw", "table", "init", "learned")
    res = {v: {f"lam{int(l)}": dict(lpips=[], ssim=[], mse=[]) for l in lams} for v in variants + ("step_end",)}
    for j, tj in enumerate(trajs_eval):
        with torch.no_grad():
            ref = ((vae.decode(fibo2.denormalize_latents(lat(tj["x0"][-1:]).cuda(), *bn).to(vae.dtype)).sample.float().clamp(-1, 1) + 1) / 2)
        n_s = tj["h"].shape[0]
        prev = dict(raw=tj["h"].float(), table=table_blend(tj, weights), init=apply_blend(init, tj), learned=apply_blend(blend, tj))
        key = f"lam{int(tj['lam'])}"
        for v, pv in prev.items():
            lpv, ssv = score(pv.flatten(0, 1), ref)
            res[v][key]["lpips"].append(lpv.view(n_s, n_k).tolist())
            res[v][key]["ssim"].append(ssv.view(n_s, n_k).tolist())
            res[v][key]["mse"].append(latent_mse(pv, tj).tolist())
        lpv, ssv = score(tj["x0"], ref)
        res["step_end"][key]["lpips"].append(lpv.tolist())
        res["step_end"][key]["ssim"].append(ssv.tolist())
        print(f"eval {j + 1}/{len(trajs_eval)}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(dict(checkpoint=checkpoint, steps=steps, n_train=len(tr), n_eval=len(held),
                                                          train_iters=train_iters, results=res)))
    print(summary(out_dir / "results.json"))


def worse_count(lp_sk, end_s):
    """
    Pure function. Shown in order, every block preview then the step end (no gate, 'Every layer'): how many previews are
    worse (higher LPIPS) than the one before.

    Examples:
        >>> worse_count([[0.5, 0.4], [0.35, 0.2]], [0.3, 0.1])
        1
    """
    seq = [x for row, e in zip(lp_sk, end_s) for x in list(row) + [e]]
    return sum(1 for a, b in zip(seq, seq[1:]) if b > a + 1e-4)


def summary(results_json):
    """Query (reads results.json). A text table: per λ and step, the mean LPIPS and SSIM over items and blocks."""
    r = json.loads(Path(results_json).read_text())["results"]
    lines = []
    for lam in r["raw"]:
        lines.append(f"λ = {lam[3:]}: mean over held-out items and blocks; LPIPS ↓ / SSIM ↑; worse = previews worse than the one before, per generation")
        for v in ("raw", "table", "init", "learned"):
            lp, ss = np.array(r[v][lam]["lpips"]), np.array(r[v][lam]["ssim"])  # (items, S, K)
            end = np.array(r["step_end"][lam]["lpips"])
            worse = np.mean([worse_count(a, e) for a, e in zip(lp, end)])
            per_step = " ".join(f"s{s + 1} {lp[:, s].mean():.3f}/{ss[:, s].mean():.3f}" for s in range(lp.shape[1]))
            lines.append(f"  {v:8s} all {lp.mean():.4f}/{ss.mean():.4f}  {per_step}  worse {worse:.1f}")
        end, ends = np.array(r["step_end"][lam]["lpips"]), np.array(r["step_end"][lam]["ssim"])
        lines.append("  step end " + " ".join(f"s{s + 1} {end[:, s].mean():.3f}/{ends[:, s].mean():.3f}" for s in range(end.shape[1])))
    return "\n".join(lines)


if __name__ == "__main__":
    import fire
    fire.Fire(dict(run=run, summary=summary))
