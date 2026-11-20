"""
Splits an exported web graph (transformer, text encoder, VAE encoder, VAE decoder) into K chained graphs between its
repeated units (transformer blocks, text-encoder layers, VAE resnet / attention / sampler blocks), so the browser app
can report progress inside every stage (manifest §12 "Progress"). The parts reuse the unsplit graph's external weight
files (same file names and offsets), so publishing them adds only the small graph files.

Default: one part per unit (m00738: "Each layer and each transformer bit would count as a loading bar part"; m00753:
"the VAE and text encoder even that has sub components ... it could be convolutional blocks or whatever").

Usage: PYTHONPATH=src uv run python -m sana_alpha.web_chunks split models/web/<version> [--roles transformer,vae_decoder] [--k K]
       PYTHONPATH=src uv run python -m sana_alpha.web_chunks verify models/web/<version>
"""
import json
import re
from pathlib import Path

import fire
import numpy as np
import onnx

# The unit of each graph: the module path (the dynamo exporter's 'namespace' metadata) of one repeated block
UNIT_PATTERNS = {
    "transformer": re.compile(r"transformer\.[a-z_]+\.\d+(?=[.:])"),  # SANA transformer_blocks, FIBO2 noise_refiner + layers
    "text_encoder": re.compile(r"model\.layers\.\d+(?=[.:])"),  # Gemma-2 / Qwen3-VL decoder layers
    "vae_encoder": re.compile(r"vae\.encoder\.(?:down_blocks\.\d+(?:\.(?:resnets|downsamplers))?|mid_block\.(?:resnets|attentions))\.\d+(?=[.:])"),
    "vae_decoder": re.compile(r"vae\.decoder\.(?:up_blocks\.\d+(?:\.(?:resnets|upsamplers))?|mid_block\.(?:resnets|attentions))\.\d+(?=[.:])"),
}  # VAE: SANA's DC-AE down/up_blocks.<i>.<j>, FLUX.2's down/up_blocks.<i>.resnets/samplers.<j> and mid_block.<kind>.<j>
PART_NAME = "{}.part{}.onnx"  # role, index
LEGACY_KEYS = ("chunks", "chunk_gpu_outputs", "blocks")  # the transformer-only format before 2026-11-20 (transformer.chunk<i>.onnx)
SEQUENCE_OPS = {"SplitToSequence", "SequenceConstruct", "SequenceInsert", "SequenceErase", "SequenceEmpty"}  # outputs are sequences
TYPE_QUERY_NAME = "{}.typequery.onnx"  # temporary model next to the external data (removed after use)
VERIFY_LATENT_SIDE = {"sana": 8, "fibo2": 16}  # small latents: the check is chunked == unsplit, not image quality
VERIFY_PROMPT_TOKENS = 16  # FIBO2's dynamic prompt length L for the check
VERIFY_TEXT_TOKENS = 12  # text-encoder input length T for the check
VAE_FACTOR = 8 * 4  # SANA's DC-AE is 32x; FIBO2's FLUX.2 is 8x: an image side of 8 latents x 32 fits both
PERTURBATION = 1e-5  # relative input change whose effect on the unsplit graph is verify's yardstick for rounding chaos


def unit_ordinals(namespaces, pattern, has_inputs):
    """
    Pure function. The unit number of each node (units numbered in the order their first computing node appears, i.e.
    data-flow order for an exported module), or None outside every unit. Nodes without inputs (Constants) do not set
    the order: a constant can be emitted before the block that uses it.

    Args:
        namespaces (list[str]): each node's 'namespace' metadata (module path), in topological order
        pattern (re.Pattern): matches one unit's module path (UNIT_PATTERNS)
        has_inputs (list[bool]): whether each node has inputs

    Returns:
        list[int | None]

    Examples:
        >>> p = UNIT_PATTERNS["vae_decoder"]
        >>> unit_ordinals(["M/vae.decoder.conv_in: C", "M/vae.decoder.mid_block.resnets.0.conv1: C", "M/vae.decoder.mid_block.attentions.0.q: L",
        ...                "M/vae.decoder.mid_block.resnets.1.conv1: C", "M/vae.decoder.up_blocks.0.resnets.0.conv1: C"], p, [True] * 5)
        [None, 0, 1, 2, 3]
        >>> unit_ordinals(["M/transformer.layers.1.c: K", "M/transformer.layers.0.ff: L", "M/transformer.layers.1.ff: L"],
        ...               UNIT_PATTERNS["transformer"], [False, True, True])
        [1, 0, 1]
    """
    keys = [m[0] if (m := pattern.search(ns)) else None for ns in namespaces]
    order = {}
    for key, computes in zip(keys, has_inputs):
        if key is not None and computes and key not in order:
            order[key] = len(order)
    for key in keys:  # units with only input-less nodes (none in practice) go last
        if key is not None and key not in order:
            order[key] = len(order)
    return [order[k] if k is not None else None for k in keys]


def assign_chunks(ordinals, inputs, outputs, graph_outputs, k, sequence_nodes=()):
    """
    Pure function. The chunk (0..k−1) of every node: unit b of B goes to chunk ⌊b·k/B⌋; a node outside the blocks goes
    to the chunk of its first consumer (so shape arithmetic on graph inputs is computed where it is used, not passed
    along). Readers of a sequence (the outputs of sequence_nodes) join the sequence's chunk, with any outside-block nodes
    they depend on (e.g. a Constant index first used later): onnxruntime-web passes only tensors between sessions.
    Fails loudly when a block would need a tensor from a later chunk.

    Args:
        ordinals (list[int | None]): unit_ordinals of the nodes, in topological order
        inputs, outputs (list[list[str]]): each node's input / output tensor names
        graph_outputs (set[str]): the graph's output names
        k (int): number of chunks
        sequence_nodes (iterable[int]): nodes whose outputs are sequences (SplitToSequence, SequenceConstruct, ...)

    Returns:
        list[int]

    Examples:
        >>> # shape -> block0 -> add -> block1 -> out, two chunks
        >>> assign_chunks([None, 0, None, 1], [["x"], ["x", "s"], ["a"], ["b", "s"]],
        ...               [["s"], ["a"], ["b"], ["y"]], {"y"}, 2)
        [0, 0, 1, 1]
        >>> # block0 makes a sequence q; its reader (q -> r) moves into chunk 0 so only the tensor r crosses
        >>> assign_chunks([0, None, 1], [["x"], ["q"], ["r"]], [["q"], ["r"], ["y"]], {"y"}, 2, sequence_nodes=[0])
        [0, 0, 1]
        >>> # a constant k first used in chunk 1 also feeds the reader of chunk 0's sequence q: both move to chunk 0
        >>> assign_chunks([0, None, None, 1, 1], [["x"], [], ["q", "k"], ["r"], ["k", "z"]],
        ...               [["q"], ["k"], ["r"], ["z"], ["y"]], {"y"}, 2, sequence_nodes=[0])
        [0, 0, 0, 1, 1]
    """
    blocks = 1 + max(o for o in ordinals if o is not None)
    producer = {t: i for i, outs in enumerate(outputs) for t in outs}
    consumers = {}
    for i, ins in enumerate(inputs):
        for t in ins:
            consumers.setdefault(t, []).append(i)
    lo = []
    for i, ins in enumerate(inputs):
        lo.append(max((lo[producer[t]] for t in ins if t in producer), default=0))
        if ordinals[i] is not None:
            chunk = ordinals[i] * k // blocks
            assert chunk >= lo[i], f"node {i} (block {ordinals[i]}) needs a tensor from chunk {lo[i]} > {chunk}"
            lo[i] = chunk
    chunk = list(lo)
    for i in reversed(range(len(inputs))):
        if ordinals[i] is not None:
            continue
        users = [chunk[j] for t in outputs[i] for j in consumers.get(t, [])]
        hi = min(users) if users else (k - 1 if set(outputs[i]) & graph_outputs else lo[i])
        assert hi >= lo[i], f"node {i}: consumers in chunk {hi} before its inputs' chunk {lo[i]}"
        chunk[i] = hi
    def pull(i, c):  # move a node outside the blocks (and its outside-block producers) back to chunk c
        if chunk[i] <= c:
            return
        assert ordinals[i] is None, f"node {i} (block {ordinals[i]}) is needed in chunk {c} < its chunk {chunk[i]}"
        for t in inputs[i]:
            if t in producer:
                pull(producer[t], c)
        chunk[i] = c
    for s in sequence_nodes:
        for j in [j for t in outputs[s] for j in consumers.get(t, [])]:
            pull(j, chunk[s])
    return chunk


def boundaries(chunk, inputs, outputs, graph_inputs, graph_outputs, k):
    """
    Pure function. Each chunk's inputs (graph inputs and earlier chunks' tensors it reads) and outputs (its tensors that
    later chunks read, and graph outputs), in first-use order.

    Returns:
        tuple[list[list[str]], list[list[str]]]: (chunk_inputs, chunk_outputs)

    Examples:
        >>> boundaries([0, 0, 1, 1], [["x"], ["x", "s"], ["a"], ["b", "s"]], [["s"], ["a"], ["b"], ["y"]], ["x"], {"y"}, 2)
        ([['x'], ['a', 's']], [['a', 's'], ['y']])
    """
    producer = {t: chunk[i] for i, outs in enumerate(outputs) for t in outs}
    ins, outs = [[] for _ in range(k)], [[] for _ in range(k)]
    for i, names in enumerate(inputs):
        c = chunk[i]
        for t in names:
            if (t in graph_inputs or producer.get(t, c) < c) and t not in ins[c]:
                ins[c].append(t)
                if t in producer and t not in outs[producer[t]]:
                    outs[producer[t]].append(t)
    for i, names in enumerate(outputs):
        for t in names:
            if t in graph_outputs and t not in outs[chunk[i]]:
                outs[chunk[i]].append(t)
    return ins, outs


def tensor_types(folder, role, model, names):
    """
    Query (writes and removes a temporary model in folder). ONNX element types of intermediate tensors, read from
    onnxruntime's resolved graph (no inference needed: the session is only created).

    Returns:
        dict[str, int]: name -> onnx.TensorProto element type
    """
    import onnxruntime as ort
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    have = {o.name for o in probe.graph.output}
    probe.graph.output.extend(onnx.ValueInfoProto(name=n) for n in names if n not in have)
    path = folder / TYPE_QUERY_NAME.format(role)
    onnx.save_model(probe, path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    finally:
        path.unlink()
    by_name = {"tensor(float)": onnx.TensorProto.FLOAT, "tensor(float16)": onnx.TensorProto.FLOAT16,
               "tensor(int64)": onnx.TensorProto.INT64, "tensor(int32)": onnx.TensorProto.INT32,
               "tensor(bool)": onnx.TensorProto.BOOL, "tensor(bfloat16)": onnx.TensorProto.BFLOAT16}
    return {o.name: by_name[o.type] for o in session.get_outputs() if o.name in set(names)}


def external_files(initializers):
    """
    Pure function. The external data files a set of initializers reads (their 'location' entries), sorted.

    Args:
        initializers (list[onnx.TensorProto])

    Returns:
        list[str]

    Examples:
        >>> t = onnx.TensorProto(name="w", data_location=onnx.TensorProto.EXTERNAL)
        >>> _ = t.external_data.add(key="location", value="transformer.onnx.p1.data")
        >>> external_files([t, onnx.TensorProto(name="b")])
        ['transformer.onnx.p1.data']
    """
    return sorted({e.value for t in initializers if t.data_location == onnx.TensorProto.EXTERNAL
                   for e in t.external_data if e.key == "location"})


def split_role(folder, manifest, role, k=None):
    """
    Command (writes folder/<role>.part{i}.onnx, edits manifest in place). Splits folder/<role>.onnx into K parts and
    records manifest["chunked"][role] = {"parts": K, "units": B, "gpu_outputs": [...], "external": [...]}: B units
    (UNIT_PATTERNS), unit b in part ⌊b·K/B⌋ (the app labels part c with units ⌈c·B/K⌉+1 … ⌈(c+1)·B/K⌉); gpu_outputs: per
    part the float tensors the app keeps on the GPU for the next part; external: per part the external data files it
    reads (the app hands each session only those). Any earlier split of the role is replaced. A graph without units
    (no match) is left unsplit.

    Args:
        folder (Path): models/web/<version>
        manifest (dict): the folder's manifest.json content
        role (str): a key of UNIT_PATTERNS
        k (int | None): number of parts; default one per unit
    """
    model = onnx.load(folder / f"{role}.onnx", load_external_data=False)
    g = model.graph
    assert not model.functions, "local functions are not handled"
    nodes = list(g.node)
    namespaces = [next((p.value for p in n.metadata_props if p.key == "namespace"), "") for n in nodes]
    ordinals = unit_ordinals(namespaces, UNIT_PATTERNS[role], [len(n.input) > 0 for n in nodes])
    if all(o is None for o in ordinals):
        print(f"{role}: no units, left unsplit", flush=True)
        return
    units = 1 + max(o for o in ordinals if o is not None)
    k = k or units
    old = re.compile(rf"{role}\.(?:part|chunk)\d+\.onnx")
    for name in [n for n in manifest["files"] if old.fullmatch(n)]:
        del manifest["files"][name]
        (folder / name).unlink()
    inputs, outputs = [list(n.input) for n in nodes], [list(n.output) for n in nodes]
    initializers = {t.name: t for t in g.initializer}
    graph_inputs = [i.name for i in g.input]
    graph_outputs = {o.name for o in g.output}
    sequences = [i for i, n in enumerate(nodes) if n.op_type in SEQUENCE_OPS]
    chunk = assign_chunks(ordinals, inputs, outputs, graph_outputs, k, sequences)
    ins, outs = boundaries(chunk, [[t for t in names if t and t not in initializers] for names in inputs], outputs,
                           graph_inputs, graph_outputs, k)
    crossing = sorted({t for o in outs for t in o} - graph_outputs)
    types = tensor_types(folder, role, model, crossing)
    declared = {v.name: v for v in list(g.input) + list(g.output)}
    info = lambda t: declared[t] if t in declared else onnx.helper.make_tensor_value_info(t, types[t], None)
    external = []
    for c in range(k):
        members = [n for n, a in zip(nodes, chunk) if a == c]
        used = {t for n in members for t in n.input}
        weights = [initializers[t] for t in initializers if t in used]
        graph = onnx.helper.make_graph(members, f"{role}_part{c}", [info(t) for t in ins[c]], [info(t) for t in outs[c]], weights)
        part = onnx.helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
        path = folder / PART_NAME.format(role, c)
        path.unlink(missing_ok=True)  # never write through a hard link into another version
        onnx.save_model(part, path)
        manifest["files"][path.name] = path.stat().st_size
        external.append(external_files(weights))
    floats = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}  # the app keeps these on the GPU between parts
    manifest.setdefault("chunked", {})[role] = dict(parts=k, units=units, external=external,
                                                    gpu_outputs=[[t for t in outs[c] if types.get(t) in floats] for c in range(k - 1)] + [[]])
    sizes = [manifest["files"][PART_NAME.format(role, c)] for c in range(k)]
    print(f"{role}: {units} units -> {k} parts, {len(crossing)} tensors cross part boundaries, part graphs "
          f"{sum(sizes) / 1e6:.1f} MB ({min(sizes) / 1e6:.2f}-{max(sizes) / 1e6:.2f} MB each)", flush=True)


def split(folder, roles=tuple(UNIT_PATTERNS), k=None):
    """
    Command. Splits each graph of a version folder (split_role) and writes its manifest.json; drops the transformer-only
    keys of the format before 2026-11-20 (LEGACY_KEYS).

    Args:
        folder (str): models/web/<version>
        roles (tuple[str] | str): graphs to split, default all four
        k (int | None): parts per graph; default one per unit
    """
    folder = Path(folder)
    roles = roles.split(",") if isinstance(roles, str) else roles
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key in LEGACY_KEYS:
        manifest.pop(key, None)
    for role in roles:
        split_role(folder, manifest, role, k)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


class ChainedSession:
    """
    Query object (holds the sessions). A folder's parts of one graph behind the two onnxruntime InferenceSession calls the
    verify tools use (get_inputs, run), so web_export.verify and fibo2_web.verify can test the chain like the unsplit graph.
    """

    def __init__(self, folder, make_session, role="transformer"):
        """
        Args:
            folder (Path): models/web/<version> after split
            make_session (callable): path -> onnxruntime.InferenceSession (the verify tool's own, e.g. CUDA EP)
            role (str): which graph
        """
        manifest = json.loads((folder / "manifest.json").read_text())
        self.sessions = [make_session(folder / PART_NAME.format(role, c)) for c in range(manifest["chunked"][role]["parts"])]
        graph = onnx.load(folder / f"{role}.onnx", load_external_data=False).graph
        declared = {i.name: i for s in self.sessions for i in s.get_inputs()}
        self.inputs = [declared[i.name] for i in graph.input]
        self.outputs = [o.name for o in graph.output]

    def get_inputs(self):
        """Query. The unsplit graph's inputs (name, type), in its order."""
        return self.inputs

    def run(self, output_names, feeds):
        """Command (runs the sessions). Runs the parts in order; returns the unsplit graph's outputs (or output_names)."""
        values = dict(feeds)
        for s in self.sessions:
            names = [o.name for o in s.get_outputs()]
            values.update(zip(names, s.run(names, {i.name: values[i.name] for i in s.get_inputs()})))
        return [values[n] for n in output_names or self.outputs]


def chain_problems(full, chunks):
    """
    Pure function. Why a chain of chunk graphs would not compute the same function as the unsplit graph: every node
    must appear in exactly one chunk, byte-identical, and every chunk input must be a graph input or an earlier chunk's
    output. Returns [] for a correct chain.

    Args:
        full (onnx.GraphProto): the unsplit graph
        chunks (list[onnx.GraphProto]): the chunk graphs, in run order

    Returns:
        list[str]

    Examples:
        >>> h = onnx.helper
        >>> a, b = h.make_node("Relu", ["x"], ["y"]), h.make_node("Neg", ["y"], ["z"])
        >>> full = h.make_graph([a, b], "g", [h.make_tensor_value_info("x", 1, None)], [h.make_tensor_value_info("z", 1, None)])
        >>> part = lambda n, i, o: h.make_graph([n], "c", [h.make_tensor_value_info(i, 1, None)], [h.make_tensor_value_info(o, 1, None)])
        >>> chain_problems(full, [part(a, "x", "y"), part(b, "y", "z")])
        []
        >>> chain_problems(full, [part(b, "y", "z"), part(a, "x", "y")])
        ['chunk 0 reads y, which no earlier chunk makes']
    """
    problems = []
    whole = sorted(n.SerializeToString(deterministic=True) for n in full.node)
    parts = sorted(n.SerializeToString(deterministic=True) for g in chunks for n in g.node)
    if whole != parts:
        problems.append(f"{len(parts)} chunk nodes vs {len(whole)} unsplit nodes, or some differ")
    available = {i.name for i in full.input}
    for c, g in enumerate(chunks):
        problems += [f"chunk {c} reads {i.name}, which no earlier chunk makes" for i in g.input if i.name not in available]
        available |= {o.name for o in g.output}
    return problems


def verify_inputs(spec, family, rng):
    """
    Near-pure function (advances rng). Small random inputs for one graph's check: latents and embeddings N(0, 1),
    images U(−1, 1), token ids below 1000, masks 1, scalars at typical values.

    Args:
        spec (list): manifest["graphs"][role]["inputs"], [[name, shape, dtype], ...]
        family (str): 'sana' or 'fibo2'
        rng (np.random.Generator)

    Returns:
        dict[str, np.ndarray]

    Examples:
        >>> f = verify_inputs([["input_ids", [1, "T"], "int64"], ["attention_mask", [1, "T"], "int64"]], "sana", np.random.default_rng(0))
        >>> f["input_ids"].shape, int(f["attention_mask"].sum())
        ((1, 12), 12)
    """
    side = VERIFY_LATENT_SIDE[family]
    sizes = {"LH": side, "LW": side, "L": VERIFY_PROMPT_TOKENS, "T": VERIFY_TEXT_TOKENS, "H": side * VAE_FACTOR, "W": side * VAE_FACTOR}
    scalar = {"timestep": 1.3, "guidance": 4.5, "sigma": 0.6, "adapter_scale": 0.0, "gamma": 1.0}
    feeds = {}
    for name, shape, dtype in spec:
        shape = [sizes.get(d, d) for d in shape]
        feeds[name] = (np.full(shape, scalar[name]) if name in scalar else np.ones(shape) if "mask" in name
                       else rng.integers(0, 1000, shape) if name == "input_ids" else rng.uniform(-1, 1, shape) if name == "image"
                       else rng.standard_normal(shape)).astype(dtype)
    return feeds


def verify(folder, seed=0):
    """
    Command (CPU). Checks every split graph of a version: (1) structure, the proof: chain_problems must be empty (the
    parts hold exactly the unsplit graph's nodes, wired in order), so the chain computes the same function; (2) a numeric
    smoke run on small random inputs, chained vs unsplit, printed next to the unsplit graph's own change when its first
    float input is scaled by 1 + PERTURBATION (none for the text encoder, whose inputs are token ids). On random inputs these graphs can be chaotic (fp16 activations near
    overflow: SANA int4 transformer, 8² latents, a 1e-5 relative input change moves the output by 0.71 at mean |output|
    0.42), so ORT's different fusions across a part boundary give a difference of that size; only non-finite chained
    outputs where the unsplit output is finite fail (manifest §12 "Progress").

    Args:
        folder (str): models/web/<version> after split
        seed (int): input noise seed
    """
    import onnxruntime as ort
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    cpu = ["CPUExecutionProvider"]
    for role, info in manifest["chunked"].items():
        full_graph = onnx.load(folder / f"{role}.onnx", load_external_data=False).graph
        part_graphs = [onnx.load(folder / PART_NAME.format(role, c), load_external_data=False).graph for c in range(info["parts"])]
        problems = chain_problems(full_graph, part_graphs)
        assert not problems, f"{folder.name} {role}: " + "; ".join(problems[:5])
        feeds = verify_inputs(manifest["graphs"][role]["inputs"], manifest.get("family", "sana"), np.random.default_rng(seed))
        first_float = next((n for n, v in feeds.items() if v.dtype.kind == "f" and v.size > 1), None)  # None: token ids only
        full = ort.InferenceSession(str(folder / f"{role}.onnx"), providers=cpu)
        reference = full.run(None, feeds)[0]
        nudged = first_float and full.run(None, {**feeds, first_float: feeds[first_float] * np.float32(1 + PERTURBATION)})[0]
        yardstick = f"{first_float} × (1 + {PERTURBATION:g}): {np.abs(nudged - reference).max():.3g}" if first_float else "integer inputs: no yardstick"
        del full
        out = ChainedSession(folder, lambda path: ort.InferenceSession(str(path), providers=cpu), role).run(None, feeds)[0]
        print(f"{folder.name} {role}: {info['parts']} parts hold the unsplit graph's {len(full_graph.node)} nodes; random inputs: "
              f"chained vs unsplit max |diff| {np.abs(out - reference).max():.3g} ({yardstick}), output |mean| {np.abs(reference).mean():.3g}, "
              f"nonfinite {int((~np.isfinite(out)).sum())} (unsplit {int((~np.isfinite(reference)).sum())})", flush=True)
        assert (np.isfinite(out) | ~np.isfinite(reference)).all(), f"{folder.name} {role}: non-finite chained output"


if __name__ == "__main__":
    fire.Fire(dict(split=split, verify=verify))
