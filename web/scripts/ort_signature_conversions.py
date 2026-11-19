"""
Query. Emscripten SIGNATURE_CONVERSIONS for onnxruntime-web's exported C API (onnxruntime/wasm/api.h), for a Memory64
build (build_ort_memory64.sh). In wasm64, pointers and size_t are i64, which JS must pass as BigInt; ORT's JS passes
Numbers, and emscripten converts only the functions listed in SIGNATURE_CONVERSIONS ('p' = pointer-sized, converted
Number <-> BigInt; '_' = unchanged). Prints the comma-separated list. Usage: python ort_signature_conversions.py <api.h>
"""
import re
import sys

POINTER_SIZED = re.compile(r"\*|_handle_t\b|\bsize_t\b|\buintptr_t\b|\bptrdiff_t\b|\bWGPU\w+")
EXTRA = ["wgpuBufferRelease:_p", "wgpuCreateInstance:pp"]  # webgpu.h functions the WebGPU build exports (onnxruntime_webassembly.cmake)
DECLARATION = re.compile(r"([\w:<>* ]+?)\s+EMSCRIPTEN_KEEPALIVE\s+(\w+)\s*\(([^)]*)\)\s*;", re.S)


def sig_char(type_text):
    """
    Pure function. 'p' for a pointer-sized C type (pointer, handle, size_t), '_' for any other (int, bool, float, void).

    Examples:
        >>> [sig_char(t) for t in ("size_t graph_optimization_level", "const char** error_message", "int num_threads", "void", "WGPUDevice")]
        ['p', 'p', '_', '_', 'p']
    """
    return "p" if POINTER_SIZED.search(type_text) else "_"


def conversions(header_text):
    """
    Pure function. name:signature (return first, then each argument) for every EMSCRIPTEN_KEEPALIVE declaration with a
    pointer-sized value (emscripten rejects, and needs no wrapper for, a signature without one).

    Examples:
        >>> conversions("int EMSCRIPTEN_KEEPALIVE OrtInit(int num_threads, int logging_level);\\n"
        ...             "ort_session_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSessionOptions(size_t level,\\n bool arena);")
        ['OrtCreateSessionOptions:pp_']
    """
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", header_text, flags=re.S)
    out = []
    for ret, name, args in DECLARATION.findall(text):
        params = [a for a in (p.strip() for p in args.split(",")) if a and a != "void"]
        sig = sig_char(ret) + "".join(sig_char(p) for p in params)
        if "p" in sig:
            out.append(f"{name}:{sig}")
    return out


if __name__ == "__main__":
    print(",".join(conversions(open(sys.argv[1]).read()) + EXTRA))
