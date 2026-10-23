# Airified llama.cpp

[English](README.md) · [简体中文](README.zh-CN.md)

A llama.cpp-based local inference runtime with targeted implementation optimizations, tuned deployment profiles, and automatic GPU/RAM/disk prefix reuse.

The validated profile runs **Qwen3.8-27B** on **two NVLink-connected RTX 2080 Ti 22-GiB cards**, balancing quality (approx. 8-bit weights/KV), full 256K context, DFlash2 decoding, and vision. Configuration and measurements below refer to this setup.

[Performance](#performance) · [Setup](#setup) · [Configuration](#configuration) · [Persistent cache](#automatic-tiered-prefix-cache) · [Validation](#validation)

## Target configuration

| Component | Default |
| --- | --- |
| Target model | [OrcaRouter Qwen3.8-27B-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF), **Q8_0** |
| Target KV cache | **Q8_0 K and V** |
| Context | **262144 tokens**, incl. text input, vision input, and text output |
| Speculative decoding | [Inco Qwen3.8-27B-DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), **Q8_0**, **3 proposals** |
| Vision | GPU; **64–4096 visual tokens/image**; approx. 320×200–2560×1600 at 16:10, with 32-pixel alignment |
| Parallelism | Tensor parallel (TP); **one active sequence**, with requests queued |
| Batching | Target microbatch **1024**, logical batch **2048**, draft microbatch **64** |
| Session reuse | **Automatic persistent prefix cache**, enabled by default |

Other Qwen3.8-27B Q8_0 checkpoints need compatible architecture, tokenizer and projector, plus revalidation; other target quants are untested. All active weights are on GPU except the approx. 1.26-GiB CPU input embedding table. CPU vision offload frees VRAM on the encoder rank, but the drafter rank remains the bottleneck with default placement.

## Performance

**TL;DR: Approx. 85% of the vLLM FP8/FP8-KV reference's prefill throughput and 110% of its decode throughput, with 256K vs 220K context. Stock llama.cpp reaches approx. 70% prefill and 74–82% decode at 128K.**

| Route | Peak VRAM (GiB) | 4.2 MP image | 32K code | 128K code | 32K prose | 128K prose |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| Unmodified llama.cpp, Q8/Q8 KV/256K, MTP3 | 42.0 | OOM | 941.7 / 65.3 | 633.4 / 44.2 | 939.1 / 49.9 | 631.8 / 35.3 |
| Unmodified llama.cpp, Q8/Q8 KV/256K, Q4 DFlash2-3 | 41.4 | PASS | 944.0 / 71.0 | 652.5 / 49.1 | 942.8 / 51.3 | 650.9 / 38.4 |
| **This repo: Q8/Q8 KV/256K, Q8 DFlash2-3** | **41.3** | PASS | **1148.1 / 73.3** | **793.9 / 63.2** | **1144.9 / 55.4** | **793.2 / 49.8** |
| vLLM: FP8/FP8 KV/220K, MTP3 | 41.6 | PASS | 1361.3 / 62.2 | 915.6 / 58.9 | 1368.7 / 49.9 | 915.1 / 48.1 |
| vLLM: FP8/TQK8V4/256K, MTP4 | 42.2 | PASS | 1393.1 / 55.8 | 974.8 / 25.0 | 1389.1 / 42.7 | 973.5 / 19.3 |
| vLLM: NVFP4/FP8 KV/256K, DFlash2-7 | 41.5 | PASS | 1212.3 / 121.1 | 861.8 / 112.9 | 1204.2 / 70.1 | 857.8 / 60.9 |

Cells show **prefill / decode token/s**, medians of two fresh-process runs: **32768 or 131072 input tokens, 1024 natural output tokens**, greedy sampling, normal EOS, no prefix reuse. Prefill = input tokens / client TTFT; decode excludes the first streamed token batch.

Identical prompts and fixed GPU pairs per workload; precision, speculation and settings differ. Stock MTP uses the embedded block; stock DFlash2 uses Q4/private-head, versus this repo's Q8/shared-head draft. vLLM is the [local SM75 fork](https://github.com/airium/vLLM-2080Ti-Definitive) of [weicj's deployment](https://github.com/weicj/vLLM-2080Ti-Definitive), not upstream. FP8/FP8-KV context is 225280; NVFP4 lowers target precision for speed.

### Restart and conversation-switch latency

Matching persisted prefixes avoid almost all prefill after restart:

| Input | Prose TTFT: cold → SSD restore | Code TTFT: cold → SSD restore |
| --- | ---: | ---: |
| 32K | 28.75 s → **2.80 s** | 28.36 s → **2.86 s** |
| 128K | 165.47 s → **7.26 s** | 165.41 s → **6.93 s** |

Fresh-process restores with verified SSD reads, not just warm OS cache. At 128K, **131068 tokens are reused, four recomputed**, with identical output. Startup is excluded; first use fingerprints models. Storage: XFS on two Gen4 NVMe SSDs in RAID1.

## Assumed environment

| Item | Reference configuration |
| --- | --- |
| GPUs | Two **22-GiB** RTX 2080 Ti cards, SM75, NVLink; approx. **43.0 GiB usable** after driver overhead |
| Host | Linux x86-64-v3; Ubuntu 26.04.1 LTS, NVIDIA driver 615.71.09 |
| RAM | Practical minimum: approx. **14 GiB available for inference**, incl. default 512-MiB cache |
| Storage | SSD, preferably NVMe; approx. 32 GB for model files, excluding source/build space and a persistent-cache budget |
| CUDA / compiler | CUDA **13.4**, GCC **15.2** |
| Build tools | CMake **4.2.3**, Ninja, Python 3 |
| NCCL | **2.29.7**; runtime library **and headers** required for TP |
| OpenSSL | **3.5.5** on the reference host; development package enables accelerated cache checksums |
| llama.cpp base | [`5aeac37a35b44fde4bdffaec9159f2fcb253b622`](https://github.com/ggml-org/llama.cpp/tree/5aeac37a35b44fde4bdffaec9159f2fcb253b622), October 18, 2026, plus [seven local patches](upstream.json) |

**Ordinary 11-GiB 2080 Ti cards are not sufficient for this profile.** The build explicitly targets `75-real`. Other GPU architectures and toolchain combinations are not qualified by these results.

## Features and optimizations

| Change | Benefit |
| --- | --- |
| [SM75 Q8 attention](patches/sm75-q8-attention.patch) | Fused multi-query verification and tuned GQA prefill; Q8 KV storage, FP16 attention arithmetic. |
| [Shared TP draft weights](patches/tp-shared-dflash.patch) | Original Q8 draft shares the target head/input table; no duplicate private head. |
| [Split-aware workspace and peer transfers](patches/tp-workspace-and-peer-copy.patch) | Correct TP workspace sizing; direct GPU transfers replace supported CPU-staging paths. |
| [Lazy TP collectives](patches/tp-lazy-collectives.patch) | Avoids unused collective contexts and buffers. |
| [DFlash mode-aware reservation](patches/dflash-mode-reserve.patch) | Separate reservations for generation blocks and full feature injection. |
| [Speculative context accounting](patches/speculative-context-end.patch) | Correct stopping at the context boundary with accepted-but-unemitted tokens. |
| [Persistent prefix cache](patches/persistent-prefix-cache.patch) | Automatic restart/session reuse, deduplication, bounded host buffers, quotas and recovery. |

## Setup

### 1. Install prerequisites and clone

Install CUDA 13.4 and a compatible driver ([NVIDIA guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)), then:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build python3 python3-venv pkg-config libssl-dev

git clone --recurse-submodules https://github.com/airium/llama-cpp-airified.git
cd llama-cpp-airified
```

The submodule is pinned: **do not update it to upstream `main`**. `build.sh` applies patches and rejects unexpected revisions or conflicts.

Tooling environment for downloads and NCCL; no PyTorch needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install huggingface_hub 'nvidia-nccl-cu13==2.29.7'
```

The [NCCL wheel](https://pypi.org/project/nvidia-nccl-cu13/2.29.7/) supplies headers and libraries; a compatible [system NCCL installation](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) also works.

### 2. Download the target, vision projector and drafter

Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to download only these files into the sibling `models/` directory. Run `hf auth login` first if needed.

```bash
MODEL_ROOT="$(pwd)/../models"
mkdir -p "$MODEL_ROOT"

hf download orcarouter/Qwen3.8-27B-Uncensored-GGUF \
  Qwen3.8-27B-Uncensored-Q8_0.gguf \
  mmproj-Qwen3.8-27B-Uncensored-f16.gguf \
  --local-dir "$MODEL_ROOT/orcarouter/Qwen3.8-27B-Uncensored-GGUF"

hf download incoai/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --local-dir "$MODEL_ROOT/incoai/Qwen3.8-27B-DFlash2-GGUF"
```

Use the original drafter—no head attachment, requantization or conversion. The projector retains its supplied precision despite the `f16` filename.

### 3. Build

With the tooling environment active:

```bash
export NCCL_ROOT="$(python3 -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')"
bash build.sh
```

For system NCCL, omit the `export` if autodetected, or point `NCCL_ROOT` to a prefix containing `include/nccl.h` and `lib/libnccl.so.2`. Keep the wheel environment if it provides runtime NCCL.

Build defaults: `CUDA_TOOLKIT=/usr/local/cuda`, `BUILD_DIR=build/sm75-q8-persist`, `BUILD_JOBS=16`. Override as needed, e.g. `BUILD_JOBS=8 bash build.sh`.

Builds are incremental; stale CMake paths are refreshed automatically after a move. Use `bash build.sh --rebuild` for a clean rebuild. Recreate `.venv` after moving the checkout.

### 4. Select your GPUs and launch

```bash
test -f config.env || cp config.example.env config.env
```

**Edit `config.env` first.** Set `GPU_DEVICES` to your NVLink pair—usually `0,1` on a two-card host. The template uses `2,3`.

```dotenv
GPU_DEVICES=0,1
HOST=127.0.0.1
PORT=3847
```

Model paths match the downloads above; set `MODEL_ROOT` here if using another location.

```bash
bash serve.sh --print-config
bash serve.sh
```

Runs in the foreground; Ctrl-C to stop. Automatic fit and context shifting are disabled.

## Configuration

Keep host overrides in the ignored [`config.env`](config.example.env). Explicit assignments there override inherited environment variables.

| Setting | Default / purpose |
| --- | --- |
| `PROFILE` | `tp-q8kv-262144-q8draft-sharedhead-gpuvision-u1024` |
| `GPU_DEVICES` | `2,3` in the template; **set this to your pair** |
| `HOST` / `PORT` | `127.0.0.1` / `3847` |
| `MODEL_ROOT` | Sibling `../models`; individual `TARGET_MODEL`, `VISION_MODEL`, `DRAFT_MODEL` paths can override it |
| `CPU_THREADS` | `16` |
| `UBATCH` / `BATCH` / `DRAFT_UBATCH` | `1024` / `2048` / `64` |
| `DRAFT_TOKENS` | `3` |
| `IMAGE_MIN_TOKENS` / `IMAGE_MAX_TOKENS` | `64` / `4096` |
| `REASONING` | `off`; can be enabled per API request |

### Optional code-focused profile

Three proposals are the default. For the code-focused alternative, change **all four settings**:

```dotenv
UBATCH=1280
BATCH=2560
DRAFT_UBATCH=256
DRAFT_TOKENS=6
```

Approx. **11% faster code decode at 32K, 2% at 128K**, but slower prefill and approx. 20% slower prose decode. Full-context/image validated; not the default.

## Automatic tiered prefix cache

Automatic prefix lookup across GPU, RAM and disk; no client session headers or manual save/restore. Only the uncached suffix needs prefill. **Active attention history stays on GPU**; disk preserves prefixes, not extra active context.

| Setting | Default |
| --- | --- |
| `PERSISTENT_CACHE` | `1`; set `0` to disable the persistent tier |
| `KV_CACHE_DIR` | Sibling `../llama-cpp-kv-caches/prefix-v1` |
| `KV_CACHE_RAM_MIB` | `512`, for managed hot data, queued writes, staging and metadata—not total process RSS |
| `KV_CACHE_MAX_MIB` | `163840`: 160-GiB logical disk upper watermark |
| `KV_CACHE_TARGET_MIB` | `131072`: 128-GiB eviction target; must not exceed the upper watermark |
| `KV_CACHE_MIN_FREE_MIB` | `16384`: preserve 16 GiB of filesystem free space |

Use a dedicated root per engine process; `KV_CACHE_DIR` can select an existing store. Snapshots commit atomically and share deduplicated chunks, with checksums and quota/LRU eviction. Caches contain conversation data and use private permissions.

Only committed snapshots survive restart. Reuse requires matching model/runtime, image identity and valid GDN checkpoints; changed prefixes or incompatible builds may need prefill. `cache_prompt:false` bypasses lookup and saving.

Disk writes are incremental; state capture still pauses at engine boundaries, longer for longer contexts. Linux only; no LoRA persistence.

Inspect `GET /slots` for `persistent_cache` counters; response `timings.cache_n` / `timings.prompt_n` report reused / recomputed tokens.

## API

OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:3847/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-llamacpp","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":128,"temperature":0}'
```

Images use `image_url` content parts. Enable reasoning with `"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}`. Endpoints: `/health`, `/v1/models`, `/props`, `/slots`, `/metrics`.

## Validation

Passed checks:

- **Full 262144 with vision:** 261632 input + 512 output; correct image facts, accepted DFlash proposals, no OOM.
- **Fresh-process full-context restore:** 261628 reused / four new input tokens; 512 identical output tokens and correct image facts.
- **Natural 32K/1K and 128K/1K prose/code:** identical outputs across cold, GPU-hot and verified SSD-restored fresh processes.
- Arithmetic, JSON/instructions, code, reasoning, tool calls/results.
- 64/4096-token images, five maximum-budget images, changed-image cache invalidation.
- Shared-prefix branching, RAM switching, quotas/eviction, corruption fallback, interrupted writes, read-only metadata.
- Native server/store and CUDA/backend/sanitizer checks.

Functional/capacity checks, not broad model-quality evaluation or guaranteed identical output across arbitrary settings.

Optional smoke checks (running server; Python standard library only):

```bash
python3 tools/probe.py --base http://127.0.0.1:3847 --output artifacts/smoke --suite smoke
python3 tools/probe.py --base http://127.0.0.1:3847 --output artifacts/vision --suite vision --image tests/assets/vision-fixture.png
```

Restart check: `tools/test_persistent_cache.py --help`; `produce` / `restore` compare against a GPU-hot reference.

## Upstream and model attribution

Project code and patches: [MIT](LICENSE). Upstream [llama.cpp](https://github.com/ggml-org/llama.cpp)'s [MIT license](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/LICENSE) is retained in the submodule; [upstream.json](upstream.json) records the pin and patch order. Download weights separately from OrcaRouter/Inco; their model cards specify licensing. Results apply to the named checkpoints/settings, not all Qwen variants.
