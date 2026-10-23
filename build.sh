#!/usr/bin/env bash
set -euo pipefail

configure_extra=()
build_extra=()
for arg in "$@"; do
  case "${arg}" in
    --rebuild)
      configure_extra=(--fresh)
      build_extra=(--clean-first)
      ;;
    -h|--help)
      printf 'Usage: bash build.sh [--rebuild]\n\nBuild incrementally; refresh stale CMake paths automatically.\n  --rebuild  Reset CMake configuration and rebuild all targets from scratch.\n'
      exit 0
      ;;
    *) printf 'Unknown argument: %s (see --help)\n' "${arg}" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/sm75-q8-persist"}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda}
BUILD_JOBS=${BUILD_JOBS:-16}
extra=()
python3 "${REPO_ROOT}/tools/apply_engine_patch.py"
SOURCE_DIR=$(cd -- "${REPO_ROOT}/vendor/llama.cpp" && pwd -P)
mkdir -p -- "${BUILD_DIR}"
BUILD_DIR=$(cd -- "${BUILD_DIR}" && pwd -P)

# CMake stores absolute source/build paths; moved checkouts need a fresh cache,
# not merely a clean compile. Leave unchanged checkouts incremental.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_source=
  cached_build=
  while IFS='=' read -r key value; do
    case "${key}" in
      CMAKE_HOME_DIRECTORY:INTERNAL) cached_source=${value} ;;
      CMAKE_CACHEFILE_DIR:INTERNAL) cached_build=${value} ;;
    esac
  done < "${BUILD_DIR}/CMakeCache.txt"
  if [[ "${cached_source}" != "${SOURCE_DIR}" || "${cached_build}" != "${BUILD_DIR}" ]]; then
    printf 'Stale CMake paths detected; refreshing configuration for this checkout.\n'
    configure_extra=(--fresh)
  fi
fi

if [[ -n "${NCCL_ROOT:-}" ]]; then
  extra+=("-DNCCL_INCLUDE_DIR=${NCCL_ROOT}/include" "-DNCCL_LIBRARY=${NCCL_ROOT}/lib/libnccl.so.2")
fi

cmake "${configure_extra[@]}" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
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

cmake --build "${BUILD_DIR}" "${build_extra[@]}" --parallel "${BUILD_JOBS}" \
  --target llama-server llama-fit-params llama-quantize llama-bench
