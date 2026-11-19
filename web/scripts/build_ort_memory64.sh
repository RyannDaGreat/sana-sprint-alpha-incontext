#!/usr/bin/env bash
# Builds onnxruntime-web with WebAssembly Memory64 (manifest §16): the WebGPU EP build that npm ships as
# ort-wasm-simd-threaded.asyncify.{mjs,wasm}, but with 64-bit wasm pointers, so a session can hold more than 4 GB.
# ORT removed its Memory64 build option; the source still has the pieces (ORT_WASM64 in wasm/api.cc,
# wasm/js_post_js_64.js with PTR_SIZE = 8, PTR_SIZE-aware JS), so this script turns them on:
#   -sMEMORY64 for every compile and link (CFLAGS / CXXFLAGS / LDFLAGS), -DORT_WASM64, js_post_js_64.js,
#   and MAXIMUM_MEMORY 16 GB (Chrome's wasm64 cap).
# Same version as web/package.json's onnxruntime-web. github.com/microsoft/onnxruntime answers 404 from this node,
# so the source comes from a mirror (MIRROR). Output: web/public/ort-memory64/ (git-ignored;
# run it before publish_pages --build_web; sizes in manifest §16).
# Usage: web/scripts/build_ort_memory64.sh            BUILD_DIR=<fast local disk> JOBS=24 to override
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ORT_VERSION=$(jq -r '.version' web/node_modules/onnxruntime-web/package.json)
MIRROR=${MIRROR:-https://gitcode.com/gh_mirrors/on/onnxruntime.git}
BUILD_DIR=${BUILD_DIR:-agents/.scratchpad/ort_build}
JOBS=${JOBS:-24}  # A quarter of the cores: training's data path shares the CPUs
MAXIMUM_MEMORY=17179869184  # 16 GB, Chrome's Memory64 limit
OUT=web/public/ort-memory64  # Vite copies public/ into the site; the app loads it as ort-memory64/ (pipeline.js MEMORY64_PATH)
SRC=$BUILD_DIR/onnxruntime-$ORT_VERSION

[ -d $SRC ] || git clone --progress --depth 1 --branch v$ORT_VERSION $MIRROR $SRC
git -C $SRC submodule update --init --depth 1 --progress cmake/external/onnx cmake/external/emsdk
EMSDK_VERSION=$(grep -oP 'emsdk_version", default="\K[^"]+' $SRC/tools/ci_build/build_args.py)
$SRC/cmake/external/emsdk/emsdk install $EMSDK_VERSION
$SRC/cmake/external/emsdk/emsdk activate $EMSDK_VERSION

WASM_CMAKE=$SRC/cmake/onnxruntime_webassembly.cmake
sed -i 's#wasm/js_post_js\.js#wasm/js_post_js_64.js#g; s#set(MAXIMUM_MEMORY "4294967296")#set(MAXIMUM_MEMORY "'$MAXIMUM_MEMORY'")#' $WASM_CMAKE
grep -q 'js_post_js_64.js' $WASM_CMAKE && grep -q "MAXIMUM_MEMORY \"$MAXIMUM_MEMORY\"" $WASM_CMAKE \
    || { echo "FAILED: the Memory64 edits no longer match $WASM_CMAKE" >&2; exit 1; }
# The external-data loader's JS truncates pointers to 32 bits (Number(p >>> 0)): above 4 GB the weights land at the
# wrong address (all-NaN outputs). Number(p) is exact up to 2^53
LOADER=$SRC/onnxruntime/core/framework/external_data_loader.cc
sed -i 's/Number(\([$a-z_0-9]*\) >>> 0)/Number(\1)/g' $LOADER
! grep -q '>>> 0' $LOADER || { echo "FAILED: 32-bit pointer truncations left in $LOADER" >&2; exit 1; }

# ORT's JS passes pointers and size_t as Numbers; wasm64 wants BigInt, so emscripten wraps the exports listed here
SIGNATURES=$(python3 web/scripts/ort_signature_conversions.py $SRC/onnxruntime/wasm/api.h)
export CFLAGS="-sMEMORY64=1" CXXFLAGS="-sMEMORY64=1 -DORT_WASM64"
nice -n 19 python3 $SRC/tools/ci_build/build.py --build_dir $BUILD_DIR/build_m64 --config Release --parallel $JOBS \
    --skip_submodule_sync --build_wasm --enable_wasm_simd --enable_wasm_threads --use_webgpu \
    --target onnxruntime_webassembly --enable_wasm_api_exception_catching --skip_tests --allow_running_as_root \
    --cmake_extra_defines "CMAKE_EXE_LINKER_FLAGS=-sMEMORY64=1 -sSIGNATURE_CONVERSIONS=$SIGNATURES"

mkdir -p $OUT
cp -v $BUILD_DIR/build_m64/Release/ort-wasm-simd-threaded.asyncify.{mjs,wasm} $OUT/
ls -l $OUT
