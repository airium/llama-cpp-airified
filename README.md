# Airified llama.cpp

[English](README.md) · [简体中文](README.zh-CN.md)

Airium's llama.cpp-based local inference runtime, with targeted implementation optimizations, tuned deployment profiles, and automatic GPU/RAM/disk prefix reuse.

The current validated profile runs **Qwen3.8-27B** with approximately eight-bit weights and KV cache, full 256K context, vision, and DFlash2 on **two NVLink-connected RTX 2080 Ti 22-GiB cards**. It favors prose and document analysis without reducing target precision or context to fit speculative decoding. The configuration and measurements below describe this reference deployment.

[Performance](#performance) · [Setup](#setup) · [Configuration](#configuration) · [Persistent cache](#automatic-tiered-prefix-cache) · [Validation](#validation)

## Target configuration

| Component | Default |
| --- | --- |
| Target model | [OrcaRouter Qwen3.8-27B-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF), **Q8_0** |
| Target KV cache | **Q8_0 K and V** |
| Context | **262,144 total tokens**, including input, images, and generated output |
| Speculative decoding | [Inco Qwen3.8-27B-DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), **Q8_0**, **three proposals** |
| Vision | GPU encoder; **64 minimum / 4,096 maximum visual tokens per image** |
| Parallelism | Tensor parallel across both GPUs; **one active sequence**, with requests queued |
| Batching | Target microbatch **1024**, logical batch **2048**, draft microbatch **64** |
| Session reuse | Automatic persistent prefix cache, enabled by default |

All target decoder layers remain on GPU. The vision encoder and draft backbone occupy opposite TP ranks; the drafter shares the target's existing TP-sharded output head. The approximately 1.26-GiB input embedding table stays in CPU RAM. **No CPU vision offload or private draft-head conversion is required.**

The image minimum is a preprocessing floor, not a fixed allocation. A 256×256 image uses 64 visual tokens; a 2048×2048 image uses 4096. Larger images are resized to the configured budget, so this setting does not preserve unlimited source-image detail.

## Performance

Speed cells are **cold prefill / decode, in tokens/s**: medians of two fresh-process runs, with **32,768 or 131,072 input tokens and 1,024 unrestricted natural output tokens**. Sampling is greedy with normal EOS and no prefix reuse. Prefill is input tokens divided by client TTFT; decode excludes the first streamed token batch.

| Route | Peak VRAM (GiB) | 4.2 MP image | 32K code | 128K code | 32K prose | 128K prose |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| Unmodified llama.cpp, Q8/Q8 KV/256K, MTP3 | 42.0 | OOM | 941.7 / 65.3 | 633.4 / 44.2 | 939.1 / 49.9 | 631.8 / 35.3 |
| Unmodified llama.cpp, Q8/Q8 KV/256K, Q4 DFlash2-3 | 41.4 | PASS | 944.0 / 71.0 | 652.5 / 49.1 | 942.8 / 51.3 | 650.9 / 38.4 |
| **This repo: Q8/Q8 KV/256K, Q8 DFlash2-3** | **41.3** | PASS | **1148.1 / 73.3** | **793.9 / 63.2** | **1144.9 / 55.4** | **793.2 / 49.8** |
| vLLM: FP8/FP8 KV/220K, MTP3 | 41.6 | PASS | 1361.3 / 62.2 | 915.6 / 58.9 | 1368.7 / 49.9 | 915.1 / 48.1 |
| vLLM: FP8/TQK8V4/256K, MTP4 | 42.2 | PASS | 1393.1 / 55.8 | 974.8 / 25.0 | 1389.1 / 42.7 | 973.5 / 19.3 |
| vLLM: NVFP4/FP8 KV/256K, DFlash2-7 | 41.5 | PASS | 1212.3 / 121.1 | 861.8 / 112.9 | 1204.2 / 70.1 | 857.8 / 60.9 |

All routes use identical prompts and fixed GPU pairs per workload. These are deployment comparisons: draft precision, speculation and route settings differ. The stock MTP row uses the target's embedded prediction block; the stock DFlash2 row uses a Q4 drafter with a private head. This repository uses the original Q8 drafter with a shared head. vLLM results use the [local SM75-tuned deployment](https://github.com/airium/vLLM-2080Ti-Definitive), based on [weicj's vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive), not stock vLLM. Its FP8/FP8-KV route has 225,280-token context; NVFP4 trades target precision for speed.

### Restart and conversation-switch latency

With a matching persisted prefix, a fresh process avoids almost all long-context prefill:

| Input | Prose TTFT: cold → SSD restore | Code TTFT: cold → SSD restore |
| --- | ---: | ---: |
| 32K | 28.75 s → **2.80 s** | 28.36 s → **2.86 s** |
| 128K | 165.47 s → **7.26 s** | 165.41 s → **6.93 s** |

Each SSD case starts a fresh process and verifies physical storage reads after advising the test cache's clean file pages out of the OS cache. The 128K cases reuse **131,068 tokens**, process **four**, and generate the same 1,024 output-token IDs as their cold/GPU-hot references. Server startup is separate from TTFT; first use also fingerprints the model files.

## Tested environment

| Item | Reference configuration |
| --- | --- |
| GPUs | Two **22-GiB** RTX 2080 Ti cards, SM75, NVLink; about **43 GiB usable** after driver overhead |
| Host | Linux x86-64; Ubuntu 26.04.1 LTS, NVIDIA driver 615.71.09 |
| CPU / RAM | AMD EPYC 7B13, 256 GiB RAM; **not a measured minimum requirement** |
| Storage | NVMe SSD; persistence timings measured on an NVMe RAID1 filesystem |
| CUDA / compiler | CUDA **13.4**, GCC **15.2** |
| Build tools | CMake **4.2.3**, Ninja, Python 3 |
| NCCL | **2.29.7**; runtime library **and headers** required for TP |
| OpenSSL | **3.5.5** on the reference host; development package enables accelerated cache checksums |
| llama.cpp base | [`5aeac37a35b44fde4bdffaec9159f2fcb253b622`](https://github.com/ggml-org/llama.cpp/tree/5aeac37a35b44fde4bdffaec9159f2fcb253b622), October 18, 2026, plus [seven local patches](upstream.json) |

**Ordinary 11-GiB 2080 Ti cards are not sufficient for this profile.** The build explicitly targets `75-real`. Other GPU architectures and toolchain combinations are not qualified by these results.

Allow approximately **32 GB for the three model downloads**, additional build space, and space for your chosen persistent-cache budget. Host RAM must also accommodate CPU embeddings, active recurrent checkpoints, model-loading mappings and ordinary workspaces: the cache's 512-MiB budget is **not total process RAM**.

## Features and optimizations

| Change | Benefit |
| --- | --- |
| [SM75 Q8 attention](patches/sm75-q8-attention.patch) | Fuses Q8 cache reconstruction into small multi-query verification attention; tunes GQA/shared-query cold prefill. Preserves Q8 KV storage and FP16 attention arithmetic. |
| [Shared TP draft weights](patches/tp-shared-dflash.patch) | Uses the original Q8 drafter with the target's existing output head and input table, avoiding a duplicate private head. |
| [Split-aware workspace and peer transfers](patches/tp-workspace-and-peer-copy.patch) | Sizes workspace for actual TP partitions and removes supported vocabulary/hidden-vector CPU staging. |
| [Lazy TP collectives](patches/tp-lazy-collectives.patch) | Avoids allocating unused collective contexts and their buffers. |
| [DFlash mode-aware reservation](patches/dflash-mode-reserve.patch) | Separates generation-block reservation from full feature-injection requirements. |
| [Speculative context accounting](patches/speculative-context-end.patch) | Corrects stopping at the configured context boundary when accepted tokens are cached but not yet emitted. |
| [Persistent prefix cache](patches/persistent-prefix-cache.patch) | Adds automatic restart/session reuse, shared immutable chunks, bounded host buffers, quota eviction and failure recovery. |

These changes do not implement an SM75 chunk-GDN replacement or all-INT8 attention. The selected route does not require lossy cache compression or lower-bit target weights.

## Setup

### 1. Install prerequisites and clone

Install CUDA 13.4 and a compatible NVIDIA driver first; see NVIDIA's [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). The following assumes a Linux development environment comparable to the reference host.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build \
  python3 python3-venv pkg-config libssl-dev

git clone --recurse-submodules \
  https://github.com/airium/llama-cpp-airified.git
cd llama-cpp-airified
```

The submodule is pinned; **do not update it to upstream `main`**. `build.sh` applies the checked-in patch sequence and refuses an unexpected revision or conflicting edits.

Create a small tooling environment for downloads and the tested NCCL distribution. PyTorch is not required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install huggingface_hub 'nvidia-nccl-cu13==2.29.7'
```

The [NVIDIA NCCL wheel](https://pypi.org/project/nvidia-nccl-cu13/2.29.7/) supplies the native headers and shared library. An existing compatible system NCCL development installation can be used instead; see NVIDIA's [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html).

### 2. Download the target, vision projector and drafter

From the repository directory, download only the required files into the sibling `models/` directory. Use `hf auth login` first if your Hub access requires authentication. The official [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) supports these file-specific downloads.

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

Use the original downloaded drafter. No head attachment, re-quantization, safetensors conversion or legacy preparation script is needed. The projector keeps its supplied precision despite the `f16` filename; it is not quantized to Q8 by this setup.

### 3. Build

With the tooling environment still active:

```bash
export NCCL_ROOT="$(python3 -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')"
bash build.sh
```

If using a system NCCL installation, omit that `export` when CMake can discover it, or set `NCCL_ROOT` to a prefix containing `include/nccl.h` and `lib/libnccl.so.2`. Keep the wheel environment if it supplies your runtime NCCL library.

`CUDA_TOOLKIT` defaults to `/usr/local/cuda`; `BUILD_DIR` defaults to `build/sm75-q8-persist`; `BUILD_JOBS` defaults to 16. Set these in the build environment when needed, for example `BUILD_JOBS=8 bash build.sh` on a smaller host.

### 4. Select your GPUs and launch

```bash
test -f config.env || cp config.example.env config.env
```

**Edit `config.env` before launching.** Set `GPU_DEVICES` to your NVLink-connected pair, normally `0,1` on a two-card machine. The supplied template selects `2,3`, so it must be changed on a host with only two GPUs.

```dotenv
GPU_DEVICES=0,1
HOST=127.0.0.1
PORT=3847
```

The default model paths already match the download layout above. If you used another location, set `MODEL_ROOT` in this same file.

```bash
bash serve.sh --print-config
bash serve.sh
```

The launcher runs in the foreground; Ctrl-C stops it. It disables automatic fit and context shifting, so it will not silently reduce context or offload decoder layers to make a changed configuration fit.

## Configuration

Keep host overrides together in the ignored [`config.env`](config.example.env), starting from the example. Explicit assignments there take precedence over inherited environment variables.

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

Three proposals remain the balanced default. The measured code-focused alternative changes **all four coupled settings**:

```dotenv
UBATCH=1280
BATCH=2560
DRAFT_UBATCH=256
DRAFT_TOKENS=6
```

During tuning, this alternative reached about **81.5 tok/s at 32K** and **64.3 tok/s at 128K** for code decode, at the cost of slower prose decoding and prefill. It passed full-context/image qualification but is not the default. Simply increasing proposal length is not guaranteed to improve speed.

## Automatic tiered prefix cache

No client session header or manual save/restore call is required. Send the conversation normally: the server checks the available GPU prefix, restores a better matching saved prefix through bounded RAM buffers, and computes only the uncached suffix. Matching includes image identity, recurrent checkpoints and draft state.

**Normal decoding keeps active attention history on GPU.** Disk storage preserves reusable contexts; it does not extend active context beyond 262144 tokens or stream attention history from SSD for every output token.

| Setting | Default |
| --- | --- |
| `PERSISTENT_CACHE` | `1`; set `0` to disable the persistent tier |
| `KV_CACHE_DIR` | Sibling `../llama-cpp-kv-caches/prefix-v1` |
| `KV_CACHE_RAM_MIB` | `512`, for managed hot data, queued writes, staging and metadata—not total process RSS |
| `KV_CACHE_MAX_MIB` | `163840`: 160-GiB logical disk upper watermark |
| `KV_CACHE_TARGET_MIB` | `131072`: 128-GiB eviction target; must not exceed the upper watermark |
| `KV_CACHE_MIN_FREE_MIB` | `16384`: preserve 16 GiB of filesystem free space |

Use a dedicated cache directory; independent server processes need separate roots. Set `KV_CACHE_DIR` to keep using an existing cache location. Snapshots share immutable chunks and are committed atomically. Checksums, model/runtime identities and quota-based probation/LRU eviction handle damaged, incompatible and old entries. Cache state contains conversation data and uses private directory/file permissions.

Only committed snapshots survive a restart. Edited prefixes or incompatible builds can require prefill again; hybrid GDN reuse needs a valid recurrent boundary. `cache_prompt:false` bypasses persistent lookup and saving for that request.

Disk writes are incremental, but capture still reads/hashes live state at a safe engine boundary. Capture is not fully nonblocking; its pause grows with the saved context size. This Linux implementation does not support persistent caching with LoRA adapters.

`GET /slots` exposes `persistent_cache` hit/miss, disk, hot/queued-buffer and deduplication counters. Response `timings.cache_n` and `timings.prompt_n` show actual reused and recomputed tokens.

## API

The server exposes an OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:3847/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-llamacpp","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":128,"temperature":0}'
```

Images use standard `image_url` content parts. Enable reasoning per request with `"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}`. Health and inspection endpoints include `/health`, `/v1/models`, `/props`, `/slots` and `/metrics`.

## Validation

The Q8 deployment has passed:

- **Exact full context with vision:** 261632 input tokens, including a 4096-token image, plus 512 generated tokens = **262144**; correct image facts, accepted DFlash proposals and no OOM.
- **Fresh-process full-context restore:** 261628 reused / four new input tokens, all 512 output tokens, identical generated text and correct image facts.
- **Natural 32K/1K and 128K/1K prose/code:** identical output-token sequences across cold, GPU-hot and verified SSD-backed fresh-process runs.
- Arithmetic, JSON/instruction following, code, enabled reasoning, tool calls and tool-result continuation.
- 64-token small images, 4096-token images, five maximum-budget images, and rejection of stale reuse for changed images.
- Shared-prefix branching, RAM-tier switching, disk quotas/eviction, corrupt-entry fallback, interrupted-write recovery and read-only access-metadata handling.
- Native server/store component tests and focused CUDA/backend/sanitizer checks for the selected optimizations.

These are functional and capacity checks, not a broad model-quality benchmark or a guarantee of bit-identical output across arbitrary sampling/kernel/configuration changes.

Optional smoke checks against a running server use only Python's standard library:

```bash
python3 tools/probe.py --base http://127.0.0.1:3847 \
  --output artifacts/smoke --suite smoke
python3 tools/probe.py --base http://127.0.0.1:3847 \
  --output artifacts/vision --suite vision --image tests/assets/vision-fixture.png
```

For an automatic restart check, run `tools/test_persistent_cache.py --help`; its `produce` and `restore` stages compare a saved request with a GPU-hot reference across a server restart.

## Upstream and model attribution

Project code and local patches are released under the [MIT License](LICENSE).

This is a pinned, patched deployment of [llama.cpp](https://github.com/ggml-org/llama.cpp), whose [MIT license](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/LICENSE) is retained in the submodule. [upstream.json](upstream.json) records the exact base and ordered patches. Model weights are downloaded separately from OrcaRouter and Inco; their model cards describe the checkpoints and their licenses. Measurements and validation apply to the named checkpoint files and selected settings, not automatically to other Qwen variants.
