#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/sm75-q8-persist"}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda}
BUILD_JOBS=${BUILD_JOBS:-16}
extra=()
python3 "${REPO_ROOT}/tools/apply_engine_patch.py"
if [[ -n "${NCCL_ROOT:-}" ]]; then
  extra+=("-DNCCL_INCLUDE_DIR=${NCCL_ROOT}/include" "-DNCCL_LIBRARY=${NCCL_ROOT}/lib/libnccl.so.2")
fi

cmake -S "${REPO_ROOT}/vendor/llama.cpp" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT}" \
  -DCMAKE_CUDA_ARCHITECTURES=75-real \
  -DGGML_CUDA=ON \
  '-DGGML_CUDA_FA_QUANTS=q8_0-q8_0;f16-f16' \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF "${extra[@]}"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" \
  --target llama-server llama-fit-params llama-quantize llama-bench
