# Airified llama.cpp

[English](README.md) · [简体中文](README.zh-CN.md)

基于 llama.cpp 的本地推理运行时，包含针对性的实现优化、调优配置及 GPU／内存／磁盘自动前缀复用。

已验证配置：**两张通过 NVLink 连接的 RTX 2080 Ti 22 GiB 显卡**运行 **Qwen3.8-27B**，兼顾质量（约 8 位权重/KV）、完整 256K 上下文、DFlash2 解码和图像输入。下文配置与实测数据均针对这一部署。

[性能](#性能) · [部署](#部署) · [配置](#配置) · [持久化缓存](#自动分层前缀缓存) · [验证](#验证)

## 目标配置

| 组件 | 默认配置 |
| --- | --- |
| 目标模型 | [OrcaRouter Qwen3.8-27B-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF)，**Q8_0** |
| 目标模型 KV 缓存 | K、V 均为 **Q8_0** |
| 上下文 | **262144 token**，含文本输入、图像输入和文本输出 |
| 推测解码 | [Inco Qwen3.8-27B-DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)，**Q8_0**，**3 个候选 token** |
| 图像支持 | GPU；**64–4096 视觉 token/图像**；16:10 时约 320×200–2560×1600，按 32 像素倍数对齐 |
| 并行方式 | 张量并行（TP）；**一个活跃序列**，其余请求排队 |
| 批处理 | 目标模型物理微批次 **1024**，逻辑批次 **2048**，草稿模型物理微批次 **64** |
| 会话复用 | **自动持久化前缀缓存**，默认开启 |

其他 Qwen3.8-27B Q8_0 检查点需匹配架构、分词器和投影器，并重新验证；其他目标模型量化尚未测试。除 CPU 上约 1.26 GiB 的输入词嵌入表外，活跃权重均在 GPU。CPU 视觉卸载可释放编码器 rank 的显存，但默认布局下瓶颈仍是草稿模型 rank。

## 性能

**简述：达到 vLLM FP8/FP8-KV 参考路线约 85% 的预填充吞吐、110% 的解码吞吐，同时提供 256K 而非 220K 上下文。未修改的 llama.cpp 在 128K 下约为该参考路线的 70% 预填充、74–82% 解码吞吐。**

| 路线 | 峰值显存（GiB） | 4.2 MP 图像 | 32K 代码 | 128K 代码 | 32K 文本 | 128K 文本 |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| 未修改的 llama.cpp，Q8/Q8 KV/256K，MTP3 | 42.0 | OOM | 941.7 / 65.3 | 633.4 / 44.2 | 939.1 / 49.9 | 631.8 / 35.3 |
| 未修改的 llama.cpp，Q8/Q8 KV/256K，Q4 DFlash2-3 | 41.4 | PASS | 944.0 / 71.0 | 652.5 / 49.1 | 942.8 / 51.3 | 650.9 / 38.4 |
| **本仓库：Q8/Q8 KV/256K，Q8 DFlash2-3** | **41.3** | PASS | **1148.1 / 73.3** | **793.9 / 63.2** | **1144.9 / 55.4** | **793.2 / 49.8** |
| vLLM：FP8/FP8 KV/220K，MTP3 | 41.6 | PASS | 1361.3 / 62.2 | 915.6 / 58.9 | 1368.7 / 49.9 | 915.1 / 48.1 |
| vLLM：FP8/TQK8V4/256K，MTP4 | 42.2 | PASS | 1393.1 / 55.8 | 974.8 / 25.0 | 1389.1 / 42.7 | 973.5 / 19.3 |
| vLLM：NVFP4/FP8 KV/256K，DFlash2-7 | 41.5 | PASS | 1212.3 / 121.1 | 861.8 / 112.9 | 1204.2 / 70.1 | 857.8 / 60.9 |

单元格为**预填充 / 解码 token/s**，取两次全新进程测试的中位数：**32768 或 131072 输入 token、1024 自然输出 token**，贪心采样、正常 EOS、无前缀复用。预填充 = 输入 token / 客户端 TTFT；解码不计首批流式 token。

各路线使用相同提示词，每种负载固定使用同一对 GPU；精度、推测解码方式和配置不同。原版 MTP 使用内置模块，原版 DFlash2 使用 Q4/独立输出头，本仓库使用 Q8/共享输出头。vLLM 为 [weicj 部署](https://github.com/weicj/vLLM-2080Ti-Definitive)的[本地 SM75 分支](https://github.com/airium/vLLM-2080Ti-Definitive)，并非上游原版。FP8/FP8-KV 上下文为 225280；NVFP4 降低目标精度以换取速度。

### 重启与切换会话的延迟

命中持久化前缀后，重启可跳过几乎全部预填充：

| 输入 | 文本 TTFT：冷预填充 → SSD 恢复 | 代码 TTFT：冷预填充 → SSD 恢复 |
| --- | ---: | ---: |
| 32K | 28.75 s → **2.80 s** | 28.36 s → **2.86 s** |
| 128K | 165.47 s → **7.26 s** | 165.41 s → **6.93 s** |

已核实全新进程的实际 SSD 读取，并非仅命中 OS 热缓存。128K 下**复用 131068、重算 4 个 token**，输出一致。启动时间另计；首次使用计算模型指纹。存储为两块 Gen4 NVMe SSD 组成的 RAID1，文件系统为 XFS。

## 参考环境

| 项目 | 参考配置 |
| --- | --- |
| GPU | 两张 **22 GiB** RTX 2080 Ti，SM75、NVLink；扣除驱动开销后约 **43.0 GiB 可用显存** |
| 系统 | Linux x86-64-v3；Ubuntu 26.04.1 LTS，NVIDIA 驱动 615.71.09 |
| 内存 | 实用最低预算：约 **14 GiB 推理可用内存**，含默认 512 MiB 缓存 |
| 存储 | SSD，优先 NVMe；模型文件约 32 GB，另需源码/编译与持久缓存空间 |
| CUDA / 编译器 | CUDA **13.4**，GCC **15.2** |
| 构建工具 | CMake **4.2.3**、Ninja、Python 3 |
| NCCL | **2.29.7**；TP 需要运行时库**和头文件** |
| OpenSSL | 参考主机为 **3.5.5**；安装开发包可启用缓存校验和加速 |
| llama.cpp 基础版本 | [`5aeac37a35b44fde4bdffaec9159f2fcb253b622`](https://github.com/ggml-org/llama.cpp/tree/5aeac37a35b44fde4bdffaec9159f2fcb253b622)，2026 年 10 月 18 日，加上[七个本地补丁](upstream.json) |

**普通的 11 GiB 2080 Ti 不足以运行此配置。** 构建明确以 `75-real` 为目标；这些测试不能为其他 GPU 架构或工具链组合提供验证。

## 功能与优化

| 改动 | 作用 |
| --- | --- |
| [SM75 Q8 注意力](patches/sm75-q8-attention.patch) | 融合多查询验证、调优 GQA 预填充；Q8 KV 存储，FP16 注意力运算。 |
| [共享 TP 草稿权重](patches/tp-shared-dflash.patch) | 原始 Q8 草稿共享目标模型输出头/输入表，无需重复的独立输出头。 |
| [分片工作区与 GPU 间传输](patches/tp-workspace-and-peer-copy.patch) | 修正 TP 工作区大小；支持的路径改用 GPU 直传，移除 CPU 中转。 |
| [按需分配 TP 集合通信资源](patches/tp-lazy-collectives.patch) | 避免未使用的集合通信上下文和缓冲区。 |
| [按 DFlash 模式预留资源](patches/dflash-mode-reserve.patch) | 分别为生成块和完整特征注入预留资源。 |
| [推测解码上下文计数](patches/speculative-context-end.patch) | 正确处理已接受但未输出 token 的上下文边界。 |
| [持久化前缀缓存](patches/persistent-prefix-cache.patch) | 自动跨重启/会话复用、去重、有界主机缓冲区、配额及恢复。 |

## 部署

### 1. 安装依赖并克隆仓库

先安装 CUDA 13.4 和兼容驱动（[NVIDIA 指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)），然后：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build python3 python3-venv pkg-config libssl-dev

git clone --recurse-submodules https://github.com/airium/llama-cpp-airified.git
cd llama-cpp-airified
```

子模块版本已固定：**不要更新到上游 `main`**。`build.sh` 会应用补丁，并拒绝不符的版本或冲突。

下载和 NCCL 工具环境，无需 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install huggingface_hub 'nvidia-nccl-cu13==2.29.7'
```

[NCCL wheel](https://pypi.org/project/nvidia-nccl-cu13/2.29.7/) 提供头文件和库；也可使用兼容的[系统 NCCL](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)。

### 2. 下载目标模型、视觉投影器和草稿模型

用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 将以下文件下载到同级 `models/` 目录；需要时先运行 `hf auth login`。

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

直接使用原始草稿，无需附加输出头、重新量化或转换。投影器保留文件原有精度，虽文件名含 `f16`。

### 3. 构建

在已激活的工具环境中：

```bash
export NCCL_ROOT="$(python3 -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')"
bash build.sh
```

系统 NCCL 若能自动检测，可省略 `export`；否则将 `NCCL_ROOT` 指向包含 `include/nccl.h` 和 `lib/libnccl.so.2` 的安装前缀。运行时若使用 wheel 提供的 NCCL，请保留该环境。

构建默认值：`CUDA_TOOLKIT=/usr/local/cuda`、`BUILD_DIR=build/sm75-q8-persist`、`BUILD_JOBS=16`。可按需覆盖，如 `BUILD_JOBS=8 bash build.sh`。

默认增量构建；目录移动后会自动刷新失效的 CMake 路径。需要完全重编译时运行 `bash build.sh --rebuild`。移动仓库后还需重建 `.venv`。

### 4. 选择 GPU 并启动

```bash
test -f config.env || cp config.example.env config.env
```

**先编辑 `config.env`。** 将 `GPU_DEVICES` 设为自己的 NVLink GPU 对；双卡主机通常为 `0,1`，模板使用 `2,3`。

```dotenv
GPU_DEVICES=0,1
HOST=127.0.0.1
PORT=3847
```

模型路径与上述下载布局一致；其他位置可在此设置 `MODEL_ROOT`。

```bash
bash serve.sh --print-config
bash serve.sh
```

前台运行，Ctrl-C 停止。自动适配和上下文滑动均已禁用。

## 配置

主机配置集中放在被 Git 忽略的 [`config.env`](config.example.env)；其中的显式赋值覆盖继承的环境变量。

| 设置 | 默认值 / 用途 |
| --- | --- |
| `PROFILE` | `tp-q8kv-262144-q8draft-sharedhead-gpuvision-u1024` |
| `GPU_DEVICES` | 模板中为 `2,3`；**请改为自己的 GPU 对** |
| `HOST` / `PORT` | `127.0.0.1` / `3847` |
| `MODEL_ROOT` | 同级 `../models`；可分别通过 `TARGET_MODEL`、`VISION_MODEL`、`DRAFT_MODEL` 覆盖文件路径 |
| `CPU_THREADS` | `16` |
| `UBATCH` / `BATCH` / `DRAFT_UBATCH` | `1024` / `2048` / `64` |
| `DRAFT_TOKENS` | `3` |
| `IMAGE_MIN_TOKENS` / `IMAGE_MAX_TOKENS` | `64` / `4096` |
| `REASONING` | `off`；可按 API 请求开启 |

### 可选：偏向代码的配置

默认每轮 3 个候选 token。代码优先配置需**同时修改四项**：

```dotenv
UBATCH=1280
BATCH=2560
DRAFT_UBATCH=256
DRAFT_TOKENS=6
```

代码解码约 **32K 快 11%、128K 快 2%**，但预填充更慢、文本解码约慢 20%。已通过完整上下文/图像验证，非默认配置。

## 自动分层前缀缓存

自动跨 GPU、内存和磁盘查找前缀，无需会话请求头或手动保存/恢复；仅未缓存的后缀需要预填充。**活跃注意力历史保留在 GPU**；磁盘保存可复用前缀，不扩展活跃上下文。

| 设置 | 默认值 |
| --- | --- |
| `PERSISTENT_CACHE` | `1`；设为 `0` 可关闭持久化层 |
| `KV_CACHE_DIR` | 同级 `../llama-cpp-kv-caches/prefix-v1` |
| `KV_CACHE_RAM_MIB` | `512`，用于托管的热数据、排队写入、暂存和元数据，而非进程总 RSS |
| `KV_CACHE_MAX_MIB` | `163840`：逻辑磁盘容量上限水位为 160 GiB |
| `KV_CACHE_TARGET_MIB` | `131072`：淘汰目标为 128 GiB，不得超过上限水位 |
| `KV_CACHE_MIN_FREE_MIB` | `16384`：保留 16 GiB 文件系统可用空间 |

每个引擎进程使用独立根目录；`KV_CACHE_DIR` 可选择已有缓存。快照原子提交、共享去重数据块，并有校验和及配额/LRU 淘汰。缓存包含会话数据，采用私有访问权限。

仅已提交快照可在重启后保留。复用需匹配模型/运行时、图像标识和有效 GDN 检查点；前缀变化或构建不兼容时可能需重新预填充。`cache_prompt:false` 跳过查找和保存。

磁盘增量写入；状态采集仍在引擎边界暂停，上下文越长暂停越久。仅支持 Linux，不支持 LoRA 持久化。

通过 `GET /slots` 查看 `persistent_cache` 计数；响应 `timings.cache_n` / `timings.prompt_n` 为复用 / 重算 token 数。

## API

兼容 OpenAI 的接口：

```bash
curl http://127.0.0.1:3847/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-llamacpp","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":128,"temperature":0}'
```

图像使用 `image_url` 内容项。通过 `"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}` 开启推理。接口：`/health`、`/v1/models`、`/props`、`/slots`、`/metrics`。

## 验证

已通过：

- **含图像完整 262144：** 261632 输入 + 512 输出；图像事实正确、有接受的 DFlash 候选、无 OOM。
- **全新进程满上下文恢复：** 261628 复用 / 4 个新输入 token；512 个输出 token 一致，图像事实正确。
- **自然生成 32K/1K 与 128K/1K 文本/代码：** 冷预填充、GPU 热缓存和已核实 SSD 读取的全新进程恢复，输出一致。
- 算术、JSON/指令、代码、推理、工具调用/结果续接。
- 64/4096-token 图像、五张上限图像、图像变化时缓存失效。
- 共享前缀分支、内存层切换、配额/淘汰、损坏回退、中断写入、只读元数据。
- 原生服务/存储及 CUDA/后端/sanitizer 检查。

功能/容量验证，非全面模型质量评测，也不保证任意配置下输出一致。

可选冒烟检查（服务已运行，仅需 Python 标准库）：

```bash
python3 tools/probe.py --base http://127.0.0.1:3847 --output artifacts/smoke --suite smoke
python3 tools/probe.py --base http://127.0.0.1:3847 --output artifacts/vision --suite vision --image tests/assets/vision-fixture.png
```

重启检查：`tools/test_persistent_cache.py --help`；`produce` / `restore` 与 GPU 热缓存参考比较。

## 上游与模型致谢

项目代码与补丁：[MIT](LICENSE)。上游 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 [MIT 许可证](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/LICENSE) 保留在子模块中；[upstream.json](upstream.json) 记录基础版本和补丁顺序。权重从 OrcaRouter/Inco 单独下载，许可证见各模型卡。结果适用于所列检查点/配置，并非所有 Qwen 变体。
