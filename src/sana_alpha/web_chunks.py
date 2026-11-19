"""
Splits an exported web transformer graph into K chained graphs between transformer blocks (manifest §12 "Progress"),
so the browser app can report progress inside one transformer pass. The chunks reuse the unsplit graph's external
weight files (same file names and offsets), so publishing them adds only the small graph files.

Usage: PYTHONPATH=src uv run python -m sana_alpha.web_chunks split models/web/<version> [--k K]
       PYTHONPATH=src uv run python -m sana_alpha.web_chunks verify models/web/<version>
"""
import json
import re
from pathlib import Path

import fire
import numpy as np
import onnx

BLOCK_PATTERN = re.compile(r"transformer\.([a-z_]+)\.(\d+)[.:]")  # module path in the dynamo exporter's 'namespace' metadata
CHUNK_NAME = "transformer.chunk{}.onnx"
SEQUENCE_OPS = {"SplitToSequence", "SequenceConstruct", "SequenceInsert", "SequenceErase", "SequenceEmpty"}  # outputs are sequences
DEFAULT_K = {"sana": 5, "fibo2": 11}  # SANA 20 blocks -> 4 per chunk; FIBO2 2 refiner + 44 layers -> 4-5 per chunk
TYPE_QUERY_NAME = "transformer.typequery.onnx"  # temporary model next to the external data (removed after use)
VERIFY_LATENT_SIDE = {"sana": 8, "fibo2": 16}  # small latents: the check is chunked == unsplit, not image quality
VERIFY_PROMPT_TOKENS = 16  # FIBO2's dynamic prompt length L for the check


def block_ordinals(namespaces):
    """
    Pure function. The global block number of each node (families numbered in order of first appearance), or None.

    Args:
        namespaces (list[str]): each node's 'namespace' metadata (module path)

    Returns:
        list[int | None]

    Examples:
        >>> block_ordinals(["x: M/transformer.noise_refiner.1.attn: A", "x: M/add: B", "x: M/transformer.layers.0.ff: C",
        ...                 "x: M/transformer.noise_refiner.0.norm: D"])
        [1, None, 2, 0]
    """
    found = [BLOCK_PATTERN.search(ns) for ns in namespaces]
    sizes = {}
    for m in found:
        if m:
            sizes[m[1]] = max(sizes.get(m[1], 0), int(m[2]) + 1)
    offsets, total = {}, 0
    for family, size in sizes.items():  # dict order = first appearance
        offsets[family], total = total, total + size
    return [offsets[m[1]] + int(m[2]) if m else None for m in found]


def assign_chunks(ordinals, inputs, outputs, graph_outputs, k, sequence_nodes=()):
    """
    Pure function. The chunk (0..k−1) of every node: block b of B goes to chunk ⌊b·k/B⌋; a node outside the blocks goes
    to the chunk of its first consumer (so shape arithmetic on graph inputs is computed where it is used, not passed
    along). Readers of a sequence (the outputs of sequence_nodes) join the sequence's chunk: onnxruntime-web passes only
    tensors between sessions. Fails loudly when a block would need a tensor from a later chunk.

    Args:
        ordinals (list[int | None]): block_ordinals of the nodes, in topological order
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
    for s in sequence_nodes:
        for j in [j for t in outputs[s] for j in consumers.get(t, [])]:
            needed = max((chunk[producer[t]] for t in inputs[j] if t in producer), default=0)
            assert needed <= chunk[s], f"sequence reader {j} needs chunk {needed} > its sequence's chunk {chunk[s]}"
            chunk[j] = chunk[s]
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


def tensor_types(folder, model, names):
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
    path = folder / TYPE_QUERY_NAME
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


def split(folder, k=None):
    """
    Command. Writes folder/transformer.chunk{i}.onnx (i < K) for folder/transformer.onnx and adds them to the folder's
    manifest.json "files" (the exporter format), plus "chunks": K. Each chunk references the unsplit graph's external
    data files. K defaults to DEFAULT_K of the manifest family.

    Args:
        folder (str): models/web/<version>
        k (int | None): number of chunks
    """
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    k = k or DEFAULT_K[manifest.get("family", "sana")]
    model = onnx.load(folder / "transformer.onnx", load_external_data=False)
    g = model.graph
    assert not model.functions, "local functions are not handled"
    nodes = list(g.node)
    namespaces = [next((p.value for p in n.metadata_props if p.key == "namespace"), "") for n in nodes]
    ordinals = block_ordinals(namespaces)
    inputs, outputs = [list(n.input) for n in nodes], [list(n.output) for n in nodes]
    initializers = {t.name: t for t in g.initializer}
    graph_inputs = [i.name for i in g.input]
    graph_outputs = {o.name for o in g.output}
    sequences = [i for i, n in enumerate(nodes) if n.op_type in SEQUENCE_OPS]
    chunk = assign_chunks(ordinals, inputs, outputs, graph_outputs, k, sequences)
    ins, outs = boundaries(chunk, [[t for t in names if t and t not in initializers] for names in inputs], outputs,
                           graph_inputs, graph_outputs, k)
    crossing = sorted({t for o in outs for t in o} - graph_outputs)
    types = tensor_types(folder, model, crossing)
    declared = {v.name: v for v in list(g.input) + list(g.output)}
    info = lambda t: declared[t] if t in declared else onnx.helper.make_tensor_value_info(t, types[t], None)
    for c in range(k):
        members = [n for n, a in zip(nodes, chunk) if a == c]
        used = {t for n in members for t in n.input}
        graph = onnx.helper.make_graph(members, f"transformer_chunk{c}", [info(t) for t in ins[c]],
                                       [info(t) for t in outs[c]], [initializers[t] for t in initializers if t in used])
        part = onnx.helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
        onnx.save_model(part, folder / CHUNK_NAME.format(c))
        manifest["files"][CHUNK_NAME.format(c)] = (folder / CHUNK_NAME.format(c)).stat().st_size
        print(f"chunk {c}: {len(members)} nodes, blocks {sorted({o for o, a in zip(ordinals, chunk) if a == c and o is not None})[::3]}..., "
              f"{len(ins[c])} inputs, {len(outs[c])} outputs", flush=True)
    manifest["chunks"] = k
    floats = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}  # the app keeps these on the GPU between chunks
    manifest["chunk_gpu_outputs"] = [[t for t in outs[c] if types.get(t) in floats] for c in range(k - 1)] + [[]]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(crossing)} tensors cross chunk boundaries", flush=True)


class ChainedSession:
    """
    Query object (holds the sessions). A folder's transformer chunks behind the two onnxruntime InferenceSession calls the
    verify tools use (get_inputs, run), so web_export.verify and fibo2_web.verify can test the chain like the unsplit graph.
    """

    def __init__(self, folder, make_session):
        """
        Args:
            folder (Path): models/web/<version> after split
            make_session (callable): path -> onnxruntime.InferenceSession (the verify tool's own, e.g. CUDA EP)
        """
        manifest = json.loads((folder / "manifest.json").read_text())
        self.sessions = [make_session(folder / CHUNK_NAME.format(c)) for c in range(manifest["chunks"])]
        graph = onnx.load(folder / "transformer.onnx", load_external_data=False).graph
        declared = {i.name: i for s in self.sessions for i in s.get_inputs()}
        self.inputs = [declared[i.name] for i in graph.input]
        self.outputs = [o.name for o in graph.output]

    def get_inputs(self):
        """Query. The unsplit graph's inputs (name, type), in its order."""
        return self.inputs

    def run(self, output_names, feeds):
        """Command (runs the sessions). Runs the chunks in order; returns the unsplit graph's outputs (or output_names)."""
        values = dict(feeds)
        for s in self.sessions:
            names = [o.name for o in s.get_outputs()]
            values.update(zip(names, s.run(names, {i.name: values[i.name] for i in s.get_inputs()})))
        return [values[n] for n in output_names or self.outputs]


def verify(folder, seed=0):
    """
    Command (CPU). Runs the unsplit transformer and the chained chunks on the same small random inputs and prints the
    largest absolute difference of the output (expected 0: the chunks are the same nodes).

    Args:
        folder (str): models/web/<version> after split
        seed (int): input noise seed
    """
    import onnxruntime as ort
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    family = manifest.get("family", "sana")
    side = VERIFY_LATENT_SIDE[family]
    sizes = {"LH": side, "LW": side, "L": VERIFY_PROMPT_TOKENS}
    rng = np.random.default_rng(seed)
    scalar = {"timestep": 1.3, "guidance": 4.5, "sigma": 0.6, "adapter_scale": 0.0}
    feeds = {}
    for name, shape, dtype in manifest["graphs"]["transformer"]["inputs"]:
        shape = [sizes.get(d, d) for d in shape]
        feeds[name] = (np.full(shape, scalar[name]) if name in scalar else np.ones(shape) if name == "prompt_mask"
                       else rng.standard_normal(shape)).astype(dtype)
    cpu = ["CPUExecutionProvider"]
    full = ort.InferenceSession(str(folder / "transformer.onnx"), providers=cpu)
    reference = full.run(None, feeds)[0]
    del full
    values = dict(feeds)
    for c in range(manifest["chunks"]):
        session = ort.InferenceSession(str(folder / CHUNK_NAME.format(c)), providers=cpu)
        names = [o.name for o in session.get_outputs()]
        values.update(zip(names, session.run(names, {i.name: values[i.name] for i in session.get_inputs()})))
        del session
    out = values[manifest["graphs"]["transformer"]["outputs"][0][0]]
    print(f"{folder.name}: chunked vs unsplit max |diff| {np.abs(out - reference).max():.3g}, "
          f"output |mean| {np.abs(reference).mean():.3g}, nonfinite {int((~np.isfinite(out)).sum())}", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(split=split, verify=verify))
