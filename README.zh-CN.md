# Airified llama.cpp

[English](README.md) · [简体中文](README.zh-CN.md)

Airium 基于 llama.cpp 定制的本地推理运行时，包含针对性的实现优化、经过调优的部署配置，以及 GPU／内存／磁盘之间的自动前缀复用。

目前已验证的配置在**两张通过 NVLink 连接的 RTX 2080 Ti 22 GiB 显卡**上运行 **Qwen3.8-27B**，同时支持约八位权重和 KV 缓存、完整 256K 上下文、图像输入及 DFlash2 推测解码。默认配置偏向通用文本与文档分析，不通过降低目标模型精度或缩短上下文来容纳推测解码。下文的配置与实测数据均针对这一参考部署。

[性能](#性能) · [部署](#部署) · [配置](#配置) · [持久化缓存](#自动分层前缀缓存) · [验证](#验证)

## 目标配置

| 组件 | 默认配置 |
| --- | --- |
| 目标模型 | [OrcaRouter Qwen3.8-27B-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF)，**Q8_0** |
| 目标模型 KV 缓存 | K、V 均为 **Q8_0** |
| 上下文 | **总计 262,144 个 token**，包括文本输入、图像和生成输出 |
| 推测解码 | [Inco Qwen3.8-27B-DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)，**Q8_0**，每轮提出 **3 个候选 token** |
| 图像支持 | GPU 视觉编码器；每张图像**最少 64 / 最多 4,096 个视觉 token** |
| 并行方式 | 双 GPU 张量并行；**一个活跃序列**，其余请求排队 |
| 批处理 | 目标模型物理微批次 **1024**，逻辑批次 **2048**，草稿模型物理微批次 **64** |
| 会话复用 | 自动持久化前缀缓存，默认开启 |

目标模型的所有解码层都保留在 GPU 上。视觉编码器和草稿模型主干分别放在两个 TP rank 上；草稿模型共享目标模型已有的 TP 分片输出头。约 1.26 GiB 的输入词嵌入表保留在 CPU 内存中。**无需将视觉编码卸载到 CPU，也无需为草稿模型转换独立的输出头。**

图像 token 下限是预处理时的尺寸下限，并非固定分配量。256×256 图像使用 64 个视觉 token，2048×2048 图像使用 4096 个。更大的图像会按配置的预算缩放，因此这一设置不会无限保留原始图像的细节。

## 性能

速度单元格表示**冷预填充 / 解码速度，单位为 token/s**：每项取两次全新进程测试的中位数，输入为 **32,768 或 131,072 个 token**，输出为 **1,024 个不受强制约束的自然生成 token**。采用贪心采样，正常处理 EOS，不复用前缀。预填充速度以输入 token 数除以客户端观测到的首 token 延迟（TTFT）计算；解码计时不包含流式返回的首批 token。

| 路线 | 峰值显存（GiB） | 4.2 MP 图像 | 32K 代码 | 128K 代码 | 32K 自然语言文本 | 128K 自然语言文本 |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| 未修改的 llama.cpp，Q8/Q8 KV/256K，MTP3 | 42.0 | OOM | 941.7 / 65.3 | 633.4 / 44.2 | 939.1 / 49.9 | 631.8 / 35.3 |
| 未修改的 llama.cpp，Q8/Q8 KV/256K，Q4 DFlash2-3 | 41.4 | PASS | 944.0 / 71.0 | 652.5 / 49.1 | 942.8 / 51.3 | 650.9 / 38.4 |
| **本仓库：Q8/Q8 KV/256K，Q8 DFlash2-3** | **41.3** | PASS | **1148.1 / 73.3** | **793.9 / 63.2** | **1144.9 / 55.4** | **793.2 / 49.8** |
| vLLM：FP8/FP8 KV/220K，MTP3 | 41.6 | PASS | 1361.3 / 62.2 | 915.6 / 58.9 | 1368.7 / 49.9 | 915.1 / 48.1 |
| vLLM：FP8/TQK8V4/256K，MTP4 | 42.2 | PASS | 1393.1 / 55.8 | 974.8 / 25.0 | 1389.1 / 42.7 | 973.5 / 19.3 |
| vLLM：NVFP4/FP8 KV/256K，DFlash2-7 | 41.5 | PASS | 1212.3 / 121.1 | 861.8 / 112.9 | 1204.2 / 70.1 | 857.8 / 60.9 |

各路线使用相同的提示词，同一种工作负载固定使用同一对 GPU。这是不同部署路线的比较：草稿精度、推测解码方式和配置并不完全相同。未修改的 MTP 路线使用目标模型内置的预测模块；未修改的 DFlash2 路线使用带独立输出头的 Q4 草稿模型；本仓库使用共享输出头的原始 Q8 草稿模型。vLLM 数据来自基于 [weicj 的 vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive) 的[本地 SM75 优化部署](https://github.com/airium/vLLM-2080Ti-Definitive)，并非原版 vLLM。其 FP8/FP8-KV 路线的上下文长度为 225,280；NVFP4 路线以降低目标模型精度换取速度。

### 重启与切换会话的延迟

命中持久化前缀后，全新进程可以跳过几乎全部长上下文预填充：

| 输入 | 自然语言文本 TTFT：冷预填充 → SSD 恢复 | 代码 TTFT：冷预填充 → SSD 恢复 |
| --- | ---: | ---: |
| 32K | 28.75 s → **2.80 s** | 28.36 s → **2.86 s** |
| 128K | 165.47 s → **7.26 s** | 165.41 s → **6.93 s** |

每个 SSD 测试均启动全新进程，先对测试缓存文件的干净页发出逐出操作系统页缓存的建议，再核实实际存储读取。128K 测试复用 **131,068 个 token**，重新处理 **4 个**，并生成与冷预填充、GPU 热缓存参考完全相同的 1,024 个输出 token ID。服务进程的启动时间不计入 TTFT；首次使用时还需要计算模型文件指纹。

## 已测试环境

| 项目 | 参考配置 |
| --- | --- |
| GPU | 两张 **22 GiB** RTX 2080 Ti，SM75、NVLink；扣除驱动开销后约有 **43 GiB 可用显存** |
| 系统 | Linux x86-64；Ubuntu 26.04.1 LTS，NVIDIA 驱动 615.71.09 |
| CPU / 内存 | AMD EPYC 7B13，256 GiB 内存；**并非实测的最低配置要求** |
| 存储 | NVMe SSD；持久化延迟在 NVMe RAID1 文件系统上测得 |
| CUDA / 编译器 | CUDA **13.4**，GCC **15.2** |
| 构建工具 | CMake **4.2.3**、Ninja、Python 3 |
| NCCL | **2.29.7**；TP 需要运行时库**和头文件** |
| OpenSSL | 参考主机为 **3.5.5**；安装开发包可启用缓存校验和加速 |
| llama.cpp 基础版本 | [`5aeac37a35b44fde4bdffaec9159f2fcb253b622`](https://github.com/ggml-org/llama.cpp/tree/5aeac37a35b44fde4bdffaec9159f2fcb253b622)，2026 年 10 月 18 日，加上[七个本地补丁](upstream.json) |

**普通的 11 GiB 2080 Ti 不足以运行此配置。** 构建明确以 `75-real` 为目标；这些测试不能为其他 GPU 架构或工具链组合提供验证。

三项模型文件下载约需 **32 GB** 空间，此外还需预留构建空间和自选的持久化缓存空间。主机内存还需容纳 CPU 词嵌入表、活跃的循环状态检查点、模型加载映射和常规工作区：缓存的 512 MiB 预算**不等于进程总内存占用**。

## 功能与优化

| 改动 | 作用 |
| --- | --- |
| [SM75 Q8 注意力](patches/sm75-q8-attention.patch) | 将 Q8 缓存重建融合到少量多查询 token 的验证注意力中，并优化 GQA／共享查询的冷预填充。保留 Q8 KV 存储和 FP16 注意力运算。 |
| [共享 TP 草稿权重](patches/tp-shared-dflash.patch) | 原始 Q8 草稿模型共享目标模型已有的输出头和输入表，避免重复分配独立输出头。 |
| [按分片预留工作区与 GPU 间传输](patches/tp-workspace-and-peer-copy.patch) | 根据实际 TP 分片大小预留工作区，并在支持的路径中移除词表／隐藏向量的 CPU 中转。 |
| [按需分配 TP 集合通信资源](patches/tp-lazy-collectives.patch) | 避免分配未使用的集合通信上下文及其缓冲区。 |
| [按 DFlash 模式预留资源](patches/dflash-mode-reserve.patch) | 区分生成块与完整特征注入的预留需求。 |
| [推测解码上下文计数](patches/speculative-context-end.patch) | 修正已接受的 token 进入缓存但尚未输出时，在上下文边界处的停止逻辑。 |
| [持久化前缀缓存](patches/persistent-prefix-cache.patch) | 自动跨重启／会话复用，支持共享不可变数据块、有界主机缓冲区、配额淘汰和故障恢复。 |

这些改动未实现 SM75 分块 GDN 替代内核或全 INT8 注意力。所选路线不需要有损缓存压缩，也不需要降低目标模型权重量化位宽。

## 部署

### 1. 安装依赖并克隆仓库

先安装 CUDA 13.4 和兼容的 NVIDIA 驱动，参见 NVIDIA 的 [CUDA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。以下步骤假设使用与参考主机相近的 Linux 开发环境。

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build \
  python3 python3-venv pkg-config libssl-dev

git clone --recurse-submodules \
  https://github.com/airium/llama-cpp-airified.git
cd llama-cpp-airified
```

子模块版本已固定，**不要将它更新到上游 `main`**。`build.sh` 会按顺序应用仓库内的补丁，并拒绝不符的基础版本或存在冲突的修改。

创建一个轻量的工具环境，用于下载模型并安装已测试的 NCCL 发行包。无需安装 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install huggingface_hub 'nvidia-nccl-cu13==2.29.7'
```

[NVIDIA NCCL wheel](https://pypi.org/project/nvidia-nccl-cu13/2.29.7/) 提供原生头文件和共享库。也可以使用系统中已有且兼容的 NCCL 开发环境，参见 NVIDIA 的 [NCCL 安装指南](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)。

### 2. 下载目标模型、视觉投影器和草稿模型

在仓库目录中执行以下命令，将所需文件下载到同级的 `models/` 目录。如果访问需要身份验证，先执行 `hf auth login`。官方 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 支持按文件下载。

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

直接使用下载的原始草稿模型即可，无需附加输出头、重新量化、转换 safetensors，或运行旧版准备脚本。视觉投影器保留下载文件原有的精度，尽管文件名含 `f16`；本配置不会将其量化为 Q8。

### 3. 构建

保持工具环境处于激活状态：

```bash
export NCCL_ROOT="$(python3 -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')"
bash build.sh
```

如果使用系统 NCCL，且 CMake 能自动找到它，可以省略上述 `export`；也可以将 `NCCL_ROOT` 设为包含 `include/nccl.h` 和 `lib/libnccl.so.2` 的安装前缀。如果运行时使用 wheel 提供的 NCCL 库，请保留这个工具环境。

`CUDA_TOOLKIT` 默认为 `/usr/local/cuda`，`BUILD_DIR` 默认为 `build/sm75-q8-persist`，`BUILD_JOBS` 默认为 16。需要时可通过构建环境覆盖，例如在较小的主机上使用 `BUILD_JOBS=8 bash build.sh`。

### 4. 选择 GPU 并启动

```bash
test -f config.env || cp config.example.env config.env
```

**启动前请编辑 `config.env`。** 将 `GPU_DEVICES` 设为通过 NVLink 连接的两张卡；双卡主机上通常为 `0,1`。模板默认选择 `2,3`，因此只有两张 GPU 的主机必须修改此项。

```dotenv
GPU_DEVICES=0,1
HOST=127.0.0.1
PORT=3847
```

默认模型路径与上述下载目录布局一致。如果使用其他位置，请在同一文件中设置 `MODEL_ROOT`。

```bash
bash serve.sh --print-config
bash serve.sh
```

服务在前台运行，按 Ctrl-C 停止。启动器禁用了自动适配和上下文滑动，因此当修改后的配置无法放入显存时，不会静默缩短上下文或将解码层卸载到 CPU。

## 配置

从示例创建被 Git 忽略的 [`config.env`](config.example.env)，将主机配置集中放在其中。该文件中的显式赋值优先于继承的环境变量。

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

兼顾不同负载的默认值仍为每轮 3 个候选 token。经过实测的代码优先配置需要**同时修改以下四个相互关联的参数**：

```dotenv
UBATCH=1280
BATCH=2560
DRAFT_UBATCH=256
DRAFT_TOKENS=6
```

在调优测试中，该配置的代码解码速度约为 **32K 下 81.5 token/s**、**128K 下 64.3 token/s**，代价是自然语言文本解码变慢、预填充速度下降。它已通过完整上下文／图像验证，但不是默认配置。单纯增加候选长度不保证更快。

## 自动分层前缀缓存

客户端无需添加会话请求头，也无需手动调用保存／恢复接口。正常发送会话即可：服务先检查 GPU 上可用的前缀，再通过有界内存缓冲区恢复匹配度更高的已保存前缀，只计算未缓存的后缀。匹配条件包括图像标识、循环状态检查点和草稿状态。

**正常解码时，活跃的注意力历史始终保留在 GPU 上。** 磁盘用于保存可复用的上下文；它既不会把活跃上下文扩展到 262144 个 token 以上，也不会在生成每个 token 时从 SSD 流式读取注意力历史。

| 设置 | 默认值 |
| --- | --- |
| `PERSISTENT_CACHE` | `1`；设为 `0` 可关闭持久化层 |
| `KV_CACHE_DIR` | 同级 `../llama-cpp-kv-caches/prefix-v1` |
| `KV_CACHE_RAM_MIB` | `512`，用于托管的热数据、排队写入、暂存和元数据，而非进程总 RSS |
| `KV_CACHE_MAX_MIB` | `163840`：逻辑磁盘容量上限水位为 160 GiB |
| `KV_CACHE_TARGET_MIB` | `131072`：淘汰目标为 128 GiB，不得超过上限水位 |
| `KV_CACHE_MIN_FREE_MIB` | `16384`：保留 16 GiB 文件系统可用空间 |

请使用专用缓存目录；独立服务进程必须使用不同的根目录。设置 `KV_CACHE_DIR` 可以继续使用已有缓存位置。快照共享不可变数据块，并以原子方式提交。校验和、模型／运行时标识，以及基于配额的 probation／LRU 淘汰策略用于处理损坏、不兼容和过期条目。缓存状态包含会话数据，目录和文件采用仅用户可访问的权限。

只有已提交的快照能在重启后保留。修改前缀或使用不兼容的构建可能需要重新预填充；混合 GDN 模型的复用需要有效的循环状态边界。请求中设置 `cache_prompt:false` 会同时跳过持久化查找和保存。

磁盘写入是增量式的，但快照采集仍需在引擎安全边界读取活跃状态并计算哈希。采集并非完全非阻塞，其暂停时间随保存的上下文大小增长。此 Linux 实现不支持在启用 LoRA 适配器时使用持久化缓存。

`GET /slots` 提供 `persistent_cache` 的命中／未命中、磁盘、热数据／排队缓冲区和去重计数。响应中的 `timings.cache_n` 与 `timings.prompt_n` 表示实际复用和重新计算的 token 数。

## API

服务提供兼容 OpenAI 的接口：

```bash
curl http://127.0.0.1:3847/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-llamacpp","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":128,"temperature":0}'
```

图像使用标准 `image_url` 内容项。若需按请求开启推理，可设置 `"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}`。健康检查和状态接口包括 `/health`、`/v1/models`、`/props`、`/slots` 和 `/metrics`。

## 验证

Q8 部署已通过以下验证：

- **含图像的精确满上下文：** 261632 个输入 token（包含一张 4096-token 图像）加上 512 个生成 token，总计 **262144**；图像事实回答正确，DFlash 候选有被接受，未发生 OOM。
- **全新进程中的满上下文恢复：** 复用 261628 个输入 token，只重新处理 4 个；完整输出 512 个 token，生成文本一致，图像事实回答正确。
- **自然生成的 32K/1K 与 128K/1K 文本／代码：** 冷预填充、GPU 热缓存及经实际 SSD 读取验证的全新进程恢复，均生成完全相同的输出 token 序列。
- 算术、JSON／指令遵循、代码、开启推理、工具调用与工具结果续接。
- 64-token 小图像、4096-token 图像、五张达到单图上限的图像，以及图像变化时拒绝复用旧缓存。
- 共享前缀分支、RAM 层会话切换、磁盘配额／淘汰、损坏条目回退、中断写入恢复和只读访问元数据处理。
- 原生服务／存储组件测试，以及针对所选优化的 CUDA／后端／sanitizer 检查。

这些是功能和容量验证，并非全面的模型质量评测，也不保证任意采样方式、内核或配置变化后仍能生成逐位一致的输出。

可选的运行中服务冒烟检查只依赖 Python 标准库：

```bash
python3 tools/probe.py --base http://127.0.0.1:3847 \
  --output artifacts/smoke --suite smoke
python3 tools/probe.py --base http://127.0.0.1:3847 \
  --output artifacts/vision --suite vision --image tests/assets/vision-fixture.png
```

如需自动重启检查，运行 `tools/test_persistent_cache.py --help`；其 `produce` 和 `restore` 阶段会跨服务重启，将保存的请求与 GPU 热缓存参考进行比较。

## 上游与模型致谢

项目代码和本地补丁采用 [MIT 许可证](LICENSE)。

本项目是基于固定版本并应用补丁的 [llama.cpp](https://github.com/ggml-org/llama.cpp) 部署，上游的 [MIT 许可证](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/LICENSE) 保留在子模块中。[upstream.json](upstream.json) 记录了精确的基础版本和补丁顺序。模型权重从 OrcaRouter 和 Inco 单独下载；具体检查点及其许可证以各自模型卡为准。实测和验证适用于文中列出的检查点文件及所选配置，并不自动适用于其他 Qwen 变体。
