#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${REPO_ROOT}/config.env"}
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi
PROFILE=${PROFILE:-tp-q8kv-262144-q8draft-sharedhead-gpuvision-u1024}
if [[ "${PROFILE}" == */* || ! -f "${REPO_ROOT}/profiles/${PROFILE}.env" ]]; then
  printf 'Unknown profile: %s\n' "${PROFILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${REPO_ROOT}/profiles/${PROFILE}.env"
MODEL_ROOT=${MODEL_ROOT:-"${REPO_ROOT}/../models"}
SERVER_BIN=${SERVER_BIN:-"${REPO_ROOT}/build/${SERVER_BUILD:-sm75}/bin/llama-server"}
TARGET_MODEL=${TARGET_MODEL:-"${MODEL_ROOT}/orcarouter/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf"}
VISION_MODEL=${VISION_MODEL:-"${MODEL_ROOT}/orcarouter/Qwen3.8-27B-Uncensored-GGUF/mmproj-Qwen3.8-27B-Uncensored-f16.gguf"}
DRAFT_QUANT=${DRAFT_QUANT:-Q8_0}
DRAFT_MODEL=${DRAFT_MODEL:-"${MODEL_ROOT}/${DRAFT_MODEL_SUBPATH:-incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-${DRAFT_QUANT}.gguf}"}
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES:-2,3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

args=("${SERVER_BIN}" --model "${TARGET_MODEL}"
  --alias "${SERVED_NAME:-qwen3.8-27b-llamacpp}"
  --host "${HOST:-127.0.0.1}" --port "${PORT:-3847}"
  --ctx-size "${CONTEXT:-262144}" --parallel 1
  --batch-size "${BATCH:-2048}" --ubatch-size "${UBATCH:-256}"
  --n-gpu-layers all --split-mode "${SPLIT_MODE:-layer}"
  --tensor-split "${TENSOR_SPLIT:-35,31}"
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
  --fit off --no-context-shift --metrics --verbosity "${VERBOSITY:-4}"
  --threads "${CPU_THREADS:-16}" --threads-batch "${CPU_THREADS:-16}"
  --reasoning "${REASONING:-off}"
  --mmproj "${VISION_MODEL}"
  --image-min-tokens "${IMAGE_MIN_TOKENS:-64}"
  --image-max-tokens "${IMAGE_MAX_TOKENS:-4096}")

if [[ -n "${TARGET_TENSOR_OVERRIDES:-}" ]]; then
  args+=(--override-tensor "${TARGET_TENSOR_OVERRIDES}")
fi

if [[ "${DRAFT_ENABLED:-1}" == 1 ]]; then
  args+=(--model-draft "${DRAFT_MODEL}" --spec-type draft-dflash
    --spec-draft-n-max "${DRAFT_TOKENS:-3}" --spec-draft-p-min 0
    --spec-draft-ngl all
    --spec-draft-n-ubatch "${DRAFT_UBATCH:-256}"
    --spec-draft-type-k f16 --spec-draft-type-v f16)
  if [[ "${DRAFT_DEVICE:-CUDA1}" != inherit ]]; then
    args+=(--spec-draft-device "${DRAFT_DEVICE:-CUDA1}")
  fi
  if [[ -n "${DRAFT_TENSOR_OVERRIDES:-}" ]]; then
    args+=(--spec-draft-override-tensor "${DRAFT_TENSOR_OVERRIDES}")
  fi
fi
case "${VISION_DEVICE:-gpu}" in
  gpu) args+=(--mmproj-offload --mmproj-device "${VISION_GPU:-CUDA0}") ;;
  cpu) args+=(--no-mmproj-offload) ;;
  *) printf 'VISION_DEVICE must be gpu or cpu\n' >&2; exit 2 ;;
esac

case "${PERSISTENT_CACHE:-0}" in
  1)
    KV_CACHE_DIR=${KV_CACHE_DIR:-"${REPO_ROOT}/../llama-cpp-kv-caches/prefix-v1"}
    KV_CACHE_RAM_MIB=${KV_CACHE_RAM_MIB:-512}
    KV_CACHE_MAX_MIB=${KV_CACHE_MAX_MIB:-163840}
    KV_CACHE_TARGET_MIB=${KV_CACHE_TARGET_MIB:-131072}
    KV_CACHE_MIN_FREE_MIB=${KV_CACHE_MIN_FREE_MIB:-16384}
    args+=(--slots --cache-dir "${KV_CACHE_DIR}" --cache-ram "${KV_CACHE_RAM_MIB}"
      --cache-dir-max "${KV_CACHE_MAX_MIB}" --cache-dir-target "${KV_CACHE_TARGET_MIB}"
      --cache-dir-min-free "${KV_CACHE_MIN_FREE_MIB}")
    ;;
  0) ;;
  *) printf 'PERSISTENT_CACHE must be 0 or 1\n' >&2; exit 2 ;;
esac

if [[ "${1:-}" == --print-config ]]; then
  python3 - "${args[@]}" <<'PY'
import json
import os
import sys
print(json.dumps({"command": sys.argv[1:], "environment": {
    key: os.environ[key] for key in sorted(os.environ)
    if key in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GGML_META_ASYNC_BROADCAST", "GGML_META_LAZY_COMM", "LLAMA_DFLASH_MODE_RESERVE"] or key.startswith("GGML_CUDA_SM75_")
}}, indent=2))
PY
  exit
fi
exec "${args[@]}" "$@"
