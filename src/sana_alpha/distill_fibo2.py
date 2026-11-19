"""
Few-step FIBO2 by distribution matching distillation (manifest §14 "Few-step FIBO2", route B; DMD2, Yin et al. 2024,
arXiv 2405.14867): a 4-step student of FIBO2 dev, text to image, as a LoRA, so the trimask control LoRA can then be
added on top of it exactly as §13 moved SANA's control onto the Sprint student.

One transformer holds all three models: dev's frozen weights (= the teacher, adapters off) plus two LoRAs, "student"
and "fake" (the fake score, initialised to the teacher since a LoRA starts at zero).
    student   x0 = x_s − s·v_student(x_s, s) at the 4 sampler levels s_i of fibo2.sample (steps = 4, the resolution shift);
              training picks a level i and reaches x_{s_i} by running the student from pure noise (backward simulation,
              no grad), deterministic Euler as at inference: x_next = x + (s_next − s)·v.
    DMD loss  x_t = (1 − t)·x0 + t·ε, t from FIBO2's shifted logit-normal; x0_real = x_t − t·v_teacher_cfg (CFG 5 vs the null
              caption), x0_fake = x_t − t·v_fake; g = (x0_fake − x0_real) / mean|x0 − x0_real| (per sample);
              loss = ½·mean((x0 − sg(x0 − g))²), whose gradient w.r.t. x0 is g (DMD's distribution-matching gradient).
    fake      flow matching on the student's (detached) samples: mean((v_fake(x_t', t') − (ε' − x0))²), FAKE_UPDATES per
              student update (DMD2's two time-scale rule, which replaces DMD's regression loss).
No GAN term and no regression pairs (DMD2 shows TTUR alone is stable; a GAN on MAGICK images would pull the student to
MAGICK's look, which the adapter trick exists to avoid). Data: MAGICK train captions only (no images), as JSON prompts.

Outputs in models/fibo2/<run>/: step_XXXXXX/{lora.safetensors (student), fake.safetensors, optimizer.pt, meta.json} every
--save_every steps, latest.txt, previews/step_XXXXXX.png (full resolution, lossless): [student 4 steps | dev 30 steps CFG 5] rows.
Launch (one GPU): CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src uv run python -m sana_alpha.distill_fibo2 train fibo2_dmd
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from sana_alpha import fibo2, models
from sana_alpha.train_control import LANCE, step_resolution

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/fibo2"
STUDENT_STEPS = 4  # the target: turbo's step count
TEACHER_CFG = 5.0  # FIBO2 dev's default guidance; the student learns the guided distribution and runs without CFG
FAKE_UPDATES = 5  # fake-score updates per student update (DMD2's TTUR)
PREVIEW_COUNT = 4
PREVIEW_CAPTIONS_SEED = 1234  # fixed preview prompts across runs


def student_levels(image_tokens, steps=STUDENT_STEPS):
    """
    Pure function. The student's noise levels: fibo2.sample's sigmas for `steps` steps, the last being 0.

    Examples:
        >>> [round(s, 3) for s in student_levels(4096)]
        [1.0, 0.905, 0.76, 0.513, 0.0]
        >>> [round(s, 3) for s in student_levels(1024)]
        [1.0, 0.849, 0.652, 0.385, 0.0]
    """
    mu = fibo2.shift_mu(image_tokens)
    return [float(fibo2.shift_sigmas(torch.tensor(s), mu)) for s in np.linspace(1.0, 1 / steps, steps)] + [0.0]


def dmd_gradient(x0, x0_real, x0_fake):
    """
    Pure function. DMD's distribution-matching gradient, normalised per sample by the mean |x0 − x0_real|.

    Args:
        x0, x0_real, x0_fake (torch.Tensor): (B, C, H, W) the student's sample and both scores' x0 estimates

    Returns:
        torch.Tensor: (B, C, H, W) g, with g = 0 where the fake score equals the real one

    Examples:
        >>> x0, real = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
        >>> dmd_gradient(x0, real, real).abs().sum().item(), dmd_gradient(x0, real, 3 * real)[0, 0, 0, 0].item()
        (0.0, 2.0)
    """
    weight = (x0 - x0_real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x0_fake - x0_real) / weight


def dmd_loss(x0, g):
    """
    Pure function. ½·mean((x0 − sg(x0 − g))²): a loss whose gradient with respect to x0 is g / x0.numel().

    Examples:
        >>> x0 = torch.zeros(2, requires_grad=True)
        >>> dmd_loss(x0, torch.tensor([2.0, -4.0])).backward(); x0.grad.tolist()
        [1.0, -2.0]
    """
    return 0.5 * ((x0.float() - (x0 - g).detach().float()) ** 2).mean()


def velocity(transformer, adapter, x, t, embeds, mask):
    """
    Query (runs the GPU). FIBO2's velocity v(x, t) with one LoRA active ("student" or "fake") or none (None = the teacher).

    Args:
        x (torch.Tensor): (B, 32, LH, LW) normalized latents; t (torch.Tensor): (B,) noise levels
        embeds, mask: (B, 6, L, 7680), (B, L) from fibo2.encode_prompts

    Returns:
        torch.Tensor: (B, 32, LH, LW) float32
    """
    if adapter is None:
        transformer.disable_adapters()
    else:
        transformer.enable_adapters()
        transformer.set_adapter(adapter)
    with torch.autocast("cuda", torch.bfloat16):
        out = transformer(hidden_states=x.to(torch.bfloat16), timestep=t, encoder_hidden_states=embeds,
                          encoder_attention_mask=mask, return_dict=False)[0]
    return out.float()


def teacher_velocity(transformer, x, t, embeds, mask, null, null_mask, guidance=TEACHER_CFG):
    """Query (runs the GPU). The teacher's classifier-free-guided velocity against the null caption. (B, 32, LH, LW)."""
    b = len(x)
    cond = velocity(transformer, None, x, t, embeds, mask)
    uncond = velocity(transformer, None, x, t, null.expand(b, -1, -1, -1), None if null_mask is None else null_mask.expand(b, -1))
    return uncond + guidance * (cond - uncond)


@torch.no_grad()
def student_sample(transformer, noise, embeds, mask, levels, until=None):
    """
    Query (runs the GPU). The student's deterministic Euler sampler from pure noise over `levels` (student_levels),
    stopping at level index `until` (None = the end: the final sample).

    Returns:
        torch.Tensor: (B, 32, LH, LW) x at levels[until], or the sample
    """
    x = noise
    stop = len(levels) - 1 if until is None else until
    for s, s_next in zip(levels[:stop], levels[1:stop + 1]):
        v = velocity(transformer, "student", x, torch.full((len(x),), s, device=x.device), embeds, mask)
        x = x + (s_next - s) * v
    return x


def load_captions(count, seed):
    """
    Query (reads magick_samples). `count` MAGICK train captions (not nsfw), shuffled with seed.

    Returns:
        list[str]
    """
    import lance

    rows = lance.dataset(str(LANCE)).to_table(columns=["subject"], filter="split = 'train' AND NOT nsfw").to_pandas()
    captions = rows.subject.astype(str).tolist()
    order = np.random.default_rng(seed).permutation(len(captions))[:count]
    return [captions[i] for i in order]


def encode_pool(text, captions, batch=16):
    """
    Query (runs the GPU). Every caption's prompt embeddings, encoded once and kept on the CPU, so the 9 GB text encoder
    can leave the GPU during training.

    Returns:
        list[tuple]: per caption (embeds (1, 6, L, 7680) bf16, mask (1, L) bool), L = its batch's longest prompt
    """
    out = []
    for i in range(0, len(captions), batch):
        embeds, mask = fibo2.encode_prompts(*text, [fibo2.caption_prompt(c) for c in captions[i:i + batch]])
        out += [(embeds[j:j + 1].cpu(), mask[j:j + 1].cpu()) for j in range(len(embeds))]
    return out


def write_preview(transformer, vae, bn, embeds, mask, references, levels, size, path):
    """
    Command. Writes a full-resolution lossless PNG: one row per preview prompt, [student 4 steps | dev 30 steps CFG 5].

    Args:
        references (torch.Tensor): (N, 3, S, S) the dev images in [0, 1], computed once
    """
    shape = (1, 32, size // 8, size // 8)
    rows = []
    for i in range(len(references)):
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(i)).to(vae.device)
        x = student_sample(transformer, noise, embeds[i:i + 1], mask[i:i + 1], levels)
        rows.append(torch.cat([decode(vae, x, bn)[0].cpu(), references[i]], dim=2))
    grid = torch.cat(rows, dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def decode(vae, x, bn):
    """Query (runs the GPU). Normalized latents (B, 32, LH, LW) -> images (B, 3, 8·LH, 8·LW) in [0, 1]."""
    return (vae.decode(fibo2.denormalize_latents(x, *bn).to(vae.dtype)).sample.float().clamp(-1, 1) + 1) / 2


def train(run, rank=64, lr=1e-4, fake_lr=1e-4, batch_size=1, max_steps=3000, save_every=500, preview_every=500,
          log_every=10, seed=0, multires=False, size=512, init_from=None, captions=2000):
    """
    Command. Distills FIBO2 dev into a 4-step student LoRA on one GPU; writes models/fibo2/<run>/.

    Args:
        run (str): output folder name under models/fibo2/
        rank (int): LoRA rank of the student and of the fake score (alpha = rank)
        lr, fake_lr (float): AdamW learning rates (betas 0, 0.999 as DMD2) of the student and of the fake score
        batch_size (int): student samples per update
        max_steps (int): student updates (each with FAKE_UPDATES fake-score updates)
        save_every, preview_every, log_every (int): in student updates
        seed (int): captions, noise, levels
        multires (bool): train_control.step_resolution's mix (50% 1024, the rest 768/512/256), as the control; False = `size`
        size (int): the training resolution without multires (512 keeps the run ≤ 25 GB on a shared GPU; previews stay 1024)
        init_from (str | None): a step folder of this run to resume from (both LoRAs, optimizer.pt, the step count)
        captions (int): size of the caption pool (embeddings precomputed on the CPU, ≈ 4.6 MB each)
    """
    from diffusers import AutoencoderKLFlux2
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = OUT_DIR / run
    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
    for part in ("transformer", "text_encoder", "processor", "vae"):
        models.localize(f"{fibo2.FIBO2_DIR}/{part}")
    bn = fibo2.bn_stats(f"{fibo2.FIBO2_DIR}/vae")
    text = fibo2.load_text_encoder(device)
    null, null_mask = fibo2.encode_prompts(*text, [fibo2.NULL_PROMPT])
    vae = AutoencoderKLFlux2.from_pretrained(models.localize(f"{fibo2.FIBO2_DIR}/vae"), torch_dtype=torch.bfloat16).to(device)
    transformer = fibo2.load_transformer(device)
    for name in ("student", "fake"):
        transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights=True, target_modules=fibo2.LORA_TARGETS), adapter_name=name)
    params = {name: [p for n, p in transformer.named_parameters() if f".{name}." in n and "lora_" in n] for name in ("student", "fake")}
    for group in params.values():
        for p in group:
            p.data = p.data.float()
            p.requires_grad_(True)
    log(f"LoRA params: student {sum(p.numel() for p in params['student']) / 1e6:.1f}M, fake {sum(p.numel() for p in params['fake']) / 1e6:.1f}M")
    optimizers = dict(student=torch.optim.AdamW(params["student"], lr=lr, betas=(0.0, 0.999), weight_decay=0.0),
                      fake=torch.optim.AdamW(params["fake"], lr=fake_lr, betas=(0.0, 0.999), weight_decay=0.0))
    start_step = 0
    if init_from:
        folder = Path(init_from)
        for name, file in (("student", "lora.safetensors"), ("fake", "fake.safetensors")):
            state = {k: v for k, v in load_file(folder / file).items() if not k.endswith(".alpha")}
            result = set_peft_model_state_dict(transformer, state, adapter_name=name)
            assert not result.unexpected_keys, f"{file}: unexpected keys {result.unexpected_keys[:3]}"
        saved = torch.load(folder / "optimizer.pt", map_location=device, weights_only=True)
        for name in optimizers:
            optimizers[name].load_state_dict(saved[name])
        start_step = json.loads((folder / "meta.json").read_text())["step"]
        log(f"resumed from {folder} at step {start_step}")
    transformer.enable_gradient_checkpointing()
    transformer.train()

    pool = encode_pool(text, load_captions(captions, seed))
    preview_captions = load_captions(PREVIEW_COUNT, PREVIEW_CAPTIONS_SEED)
    pembeds, pmask = fibo2.encode_prompts(*text, [fibo2.caption_prompt(c) for c in preview_captions])
    del text
    torch.cuda.empty_cache()
    log(f"{len(pool)} prompts encoded; text encoder freed")
    references_path = out_dir / "previews" / "reference_dev30.png"
    preview_size = 1024
    pshape = (1, 32, preview_size // 8, preview_size // 8)
    if references_path.exists():
        grid = torch.from_numpy(np.asarray(Image.open(references_path)).astype(np.float32) / 255).permute(2, 0, 1)
        references = grid.view(3, PREVIEW_COUNT, preview_size, preview_size).permute(1, 0, 2, 3)
    else:
        transformer.disable_adapters()
        with torch.no_grad():
            refs = [fibo2.sample(transformer, pembeds[i:i + 1], pmask[i:i + 1], null, null_mask, None, 30, TEACHER_CFG, seed=i, shape=pshape)
                    for i in range(PREVIEW_COUNT)]
        references = torch.cat([decode(vae, x, bn) for x in refs]).cpu()
        references_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((torch.cat(list(references), dim=1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(references_path)
        log(f"dev 30-step references -> {references_path}")
    plevels = student_levels(pshape[2] * pshape[3] // 4)
    if not init_from:
        write_preview(transformer, vae, bn, pembeds, pmask, references, plevels, preview_size, out_dir / "previews" / "step_000000.png")

    gen = torch.Generator().manual_seed(seed * 1000 + 7919 * start_step)
    meta = dict(run=run, rank=rank, lr=lr, fake_lr=fake_lr, batch_size=batch_size, multires=multires, size=size, student_steps=STUDENT_STEPS,
                teacher_cfg=TEACHER_CFG, fake_updates=FAKE_UPDATES, base=fibo2.FIBO2_DIR, method="DMD2 (DMD loss + TTUR, no GAN)",
                init_from=str(init_from), start_step=start_step)
    step, t_last = start_step, time.time()
    torch.cuda.reset_peak_memory_stats(device)
    stats = dict(dmd=0.0, fake=0.0, n=0)
    while step < max_steps:
        res = step_resolution(step, seed) if multires else size
        shape = (batch_size, 32, res // 8, res // 8)
        tokens = shape[2] * shape[3] // 4
        levels = student_levels(tokens)
        assert batch_size == 1, "the pool's prompts are padded per encoding batch; batch_size > 1 needs re-padding"
        embeds, mask = (t.to(device) for t in pool[int(torch.randint(len(pool), (1,), generator=gen))])

        # Student update: backward simulation to a random level, one student step with grad, DMD gradient
        level = int(torch.randint(STUDENT_STEPS, (1,), generator=gen))
        noise = torch.randn(shape, generator=gen).to(device)
        xs = student_sample(transformer, noise, embeds, mask, levels, until=level)
        s = torch.full((batch_size,), levels[level], device=device)
        x0 = xs - s.view(-1, 1, 1, 1) * velocity(transformer, "student", xs, s, embeds, mask)
        t = fibo2.sample_sigmas(batch_size, tokens, gen).to(device).clamp(0.02, 0.98)
        eps = torch.randn(shape, generator=gen).to(device)
        xt = (1 - t.view(-1, 1, 1, 1)) * x0.detach() + t.view(-1, 1, 1, 1) * eps
        with torch.no_grad():
            x0_real = xt - t.view(-1, 1, 1, 1) * teacher_velocity(transformer, xt, t, embeds, mask, null, null_mask)
            x0_fake = xt - t.view(-1, 1, 1, 1) * velocity(transformer, "fake", xt, t, embeds, mask)
        loss = dmd_loss(x0, dmd_gradient(x0.detach(), x0_real, x0_fake))
        transformer.enable_adapters()
        transformer.set_adapter("student")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params["student"], 1.0)
        optimizers["student"].step()
        optimizers["student"].zero_grad(set_to_none=True)
        stats["dmd"] += loss.item()

        # Fake-score updates: flow matching on the student's samples (detached)
        for _ in range(FAKE_UPDATES):
            tf = fibo2.sample_sigmas(batch_size, tokens, gen).to(device)
            ef = torch.randn(shape, generator=gen).to(device)
            xf = (1 - tf.view(-1, 1, 1, 1)) * x0.detach() + tf.view(-1, 1, 1, 1) * ef
            fake_loss = ((velocity(transformer, "fake", xf, tf, embeds, mask) - (ef - x0.detach())) ** 2).mean()
            fake_loss.backward()
            torch.nn.utils.clip_grad_norm_(params["fake"], 1.0)
            optimizers["fake"].step()
            optimizers["fake"].zero_grad(set_to_none=True)
            stats["fake"] += fake_loss.item() / FAKE_UPDATES
        step, stats["n"] = step + 1, stats["n"] + 1

        if step % log_every == 0:
            dt = time.time() - t_last
            log(f"step {step} res {res} dmd {stats['dmd'] / stats['n']:.5f} fake {stats['fake'] / stats['n']:.4f} "
                f"{dt / stats['n']:.2f} s/step mem {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GB")
            t_last, stats = time.time(), dict(dmd=0.0, fake=0.0, n=0)
        if step % save_every == 0 or step == max_steps:
            folder = out_dir / f"step_{step:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, file in (("student", "lora.safetensors"), ("fake", "fake.safetensors")):
                state = {k: v.detach().float().cpu().contiguous() for k, v in get_peft_model_state_dict(transformer, adapter_name=name).items()}
                for module in {k.rsplit(".lora_A.weight", 1)[0] for k in state if k.endswith(".lora_A.weight")}:
                    state[f"{module}.alpha"] = torch.tensor(float(rank))
                save_file(state, folder / file)
            torch.save({name: opt.state_dict() for name, opt in optimizers.items()}, folder / "optimizer.pt")
            (folder / "meta.json").write_text(json.dumps(dict(meta, step=step, mode="student")) + "\n")
            (out_dir / "latest.txt").write_text(f"step_{step:06d}\n")
            log(f"saved {folder}")
        if step % preview_every == 0 or step == max_steps:
            write_preview(transformer, vae, bn, pembeds, pmask, references, plevels, preview_size, out_dir / "previews" / f"step_{step:06d}.png")
            log(f"preview {out_dir / 'previews' / f'step_{step:06d}.png'}")


def transfer(student, control):
    """
    Command. Writes models/fibo2/<student run>/<student step>_<control run>_<control step>/: the control LoRA (lora.safetensors,
    meta.json of a control checkpoint) and the student LoRA (student.safetensors), as hard links, so fibo2.load_eval_model
    and generate_eval's "fibo2" pipeline run W_dev + student + control. Prints the folder.

    Args:
        student (str): a distill_fibo2 step folder, e.g. models/fibo2/fibo2_dmd/step_000500
        control (str): a train_fibo2 control step folder, e.g. models/fibo2/fibo2_tri/step_002000
    """
    import os

    student, control = Path(student), Path(control)
    folder = student.parent / f"{student.name}_{control.parent.name}_{control.name}"
    folder.mkdir(exist_ok=True)
    for source, name in ((control / "lora.safetensors", "lora.safetensors"), (student / "lora.safetensors", "student.safetensors")):
        if not (folder / name).exists():
            os.link(source, folder / name)
    meta = json.loads((control / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps(dict(meta, student=str(student), control=str(control))) + "\n")
    print(folder.relative_to(ROOT) if folder.is_absolute() else folder)


if __name__ == "__main__":
    fire.Fire(dict(train=train, transfer=transfer))
