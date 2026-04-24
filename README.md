<div align="center">

# Kaiwu · 开物

**让本地大模型跑出最佳状态，零配置。**

**Run local LLMs at their best. Zero configuration.**

[English](#english) · [中文](#中文)

</div>

---

<a name="english"></a>
# Kaiwu

> *"装上就跑，参数全自动"*

Kaiwu is a local LLM deployment tool that automatically finds the optimal configuration for your hardware and model — so you don't have to.

While LM Studio and Ollama make models *run*, Kaiwu makes them run *well*.

## Why Kaiwu?

Most local LLM tools give you a model that works, then leave you to figure out context length, KV cache type, batch size, thread count, and a dozen other parameters. Get it wrong and you're slow, out of memory, or wasting hardware you paid for.

Kaiwu figures all of that out for you — once — and caches the result. Second launch takes 2 seconds.

## Real Benchmarks

### 30B MoE on 8GB GPU — the hard case

Model: Qwen3-30B-A3B · RTX 5060 Laptop 8GB · Windows 11

| | LM Studio | Kaiwu |
|---|---|---|
| Speed | 3 tok/s | **21 tok/s** |
| VRAM used | 7,549 MB (93%) | 2,603 MB (32%) |
| Config required | Manual | **None** |

LM Studio fills VRAM and saturates the GPU. Kaiwu keeps attention layers on GPU and routes MoE experts through CPU — 7× faster, 65% less VRAM.

### 8B dense — everyday use

Model: Llama 3.1 8B Q5_K_M · RTX 5060 8GB

| | LM Studio | Kaiwu |
|---|---|---|
| Speed (8K ctx) | 46.5 tok/s | **51.7 tok/s** |
| Context window | 4–8K (default) | **64K (auto)** |

Same speed, 8× more context. Kaiwu automatically selects f16 KV cache when it fits in VRAM — matching LM Studio's speed while running a much larger context window.

### Dual 4090 — high-end

Model: Qwen3.6-35B-A3B · 2× RTX 4090 24GB

- **115 tok/s** · **256K context** · fully automatic tensor split

## How It Works

```
kaiwu run Qwen3-30B-A3B
```

That's it. Kaiwu:

1. **Probes your hardware** — GPU, VRAM, CPU cores, RAM
2. **Reads the model** — architecture, layer count, KV heads, native context limit
3. **Runs warmup benchmark** — probes ctx from native max downward, finds the largest window that keeps speed ≥ 20 tok/s
4. **Tunes parameters** — KV cache type, ubatch size, thread count, mlock — all measured, not guessed
5. **Caches the result** — next launch skips warmup entirely (2s startup)

On subsequent runs, you see:

```
✓ Using last config  (64K ctx · 26.2 tok/s · 3 days ago)
```

## Installation

One command, no dependencies:

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/val1813/kaiwu/main/install.ps1 | iex
```

**Linux / macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/val1813/kaiwu/main/install.sh | sh
```

Or download manually from [Releases](https://github.com/val1813/kaiwu/releases).

## Quick Start

```bash
# Run a model (auto-downloads if needed)
kaiwu run Qwen3-30B-A3B

# Run a local GGUF file
kaiwu run /path/to/model.gguf

# Connect your IDE (Continue, Cursor, Claude Code)
# Point it to: http://localhost:11435/v1

# Check what's running
kaiwu status

# Stop
kaiwu stop
```

The API is OpenAI-compatible. Any tool that works with the OpenAI API works with Kaiwu.

## Advanced Usage

```bash
# Override context size (any value, not just powers of 2)
kaiwu run Qwen3-8B --ctx-size 12000

# Force re-run warmup (after hardware change, or just to re-tune)
kaiwu run Qwen3-8B --reset

# Fast start — skip warmup, use cached config only
kaiwu run Qwen3-8B --fast

# List available models
kaiwu list

# Inject IDE config automatically
kaiwu inject
```

## What Gets Auto-Tuned

Everything you'd have to configure manually in LM Studio:

| Parameter | How Kaiwu decides |
|---|---|
| Context length | Probes from model's native max down; stops where speed ≥ 20 tok/s |
| KV cache type | Calculates f16 footprint; uses f16 if it fits, q8_0+q4_0 otherwise |
| MoE expert placement | Detects `.ffn_.*_exps.` tensors; routes to CPU automatically |
| ubatch size | Benchmarks 128 vs 512; picks the faster one |
| Thread count | 2 for full-GPU, physical_cores/2 for MoE offload |
| mlock | Enabled when RAM headroom > 30% |
| GPU tensor split | Proportional to VRAM when multiple GPUs detected |

## Requirements

- **GPU**: NVIDIA (CUDA) — 4GB+ VRAM recommended
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **RAM**: 8GB+ (16GB+ for 30B MoE models)
- **Model format**: GGUF

CPU-only inference is supported but not the focus.

## Commands

| Command | What it does |
|---|---|
| `run <model>` | Start a model. Downloads if needed. |
| `stop` | Stop the running model. |
| `status` | Show running model, speed, VRAM usage. |
| `list` | List available and downloaded models. |
| `probe` | Show detected hardware. |
| `inject` | Configure Continue/Cursor to use Kaiwu. |
| `version` | Show version. |

---

<a name="中文"></a>
# 开物 (Kaiwu)

> *"开物成务，利用厚生"* — 明·宋应星《天工开物》

Kaiwu 是一个本地大模型部署工具。它自动找到适合你硬件和模型的最优配置，你什么都不用调。

其他工具让模型"能跑"，Kaiwu 让模型"跑好"。

## 为什么选 Kaiwu？

用 LM Studio 或 Ollama 跑模型，你还需要自己搞定：上下文长度、KV cache 类型、批处理大小、线程数……一大堆参数。搞错了就慢、OOM 或者浪费了买来的显卡。

Kaiwu 替你把这些全部算好——只算一次——然后缓存起来。下次启动只需要 2 秒。

## 真实数据

### 8GB 显卡跑 30B 模型——最难的场景

模型：Qwen3-30B-A3B · RTX 5060 笔记本 8GB · Windows 11

| | LM Studio | Kaiwu |
|---|---|---|
| 速度 | 3 tok/s | **21 tok/s** |
| 显存占用 | 7,549 MB（93%） | 2,603 MB（32%） |
| 需要手动配置 | 是 | **不需要** |

LM Studio 把显存塞满，GPU 跑满。Kaiwu 只把 attention 层放 GPU，MoE expert 层走 CPU——快 7 倍，省 65% 显存。

### 8B 模型——日常使用

模型：Llama 3.1 8B Q5_K_M · RTX 5060 8GB

| | LM Studio | Kaiwu |
|---|---|---|
| 速度（8K 上下文） | 46.5 tok/s | **51.7 tok/s** |
| 上下文窗口 | 4–8K（默认） | **64K（自动）** |

速度持平甚至更快，上下文多 8 倍。Kaiwu 自动判断显存够不够装 f16 KV cache，够就用——速度匹配 LM Studio，同时开更大的上下文。

### 双 4090——高端配置

模型：Qwen3.6-35B-A3B · 2× RTX 4090 24GB

- **115 tok/s** · **256K 上下文** · 自动多卡分配

## 怎么用

```
kaiwu run Qwen3-30B-A3B
```

就这一句。Kaiwu 会：

1. **探测硬件**——GPU、显存、CPU 核数、内存
2. **读模型信息**——架构、层数、KV heads、原生上下文限制
3. **跑 warmup 基准测试**——从模型最大上下文往下探，找到在速度 ≥ 20 tok/s 前提下能跑的最大值
4. **调整参数**——KV cache 类型、ubatch、线程数、mlock——全部实测，不靠猜
5. **缓存结果**——下次启动跳过 warmup，2 秒就绪

第二次启动你会看到：

```
✓ 使用上次配置  (64K ctx · 26.2 tok/s · 3 天前)
```

## 安装

一行命令，无需任何依赖：

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/val1813/kaiwu/main/install.ps1 | iex
```

**Linux / macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/val1813/kaiwu/main/install.sh | sh
```

也可以从 [Releases](https://github.com/val1813/kaiwu/releases) 手动下载。

## 快速开始

```bash
# 运行模型（没有会自动下载）
kaiwu run Qwen3-30B-A3B

# 运行本地 GGUF 文件
kaiwu run /path/to/model.gguf

# 接入 IDE（Continue、Cursor、Claude Code）
# API 地址：http://localhost:11435/v1

# 查看运行状态
kaiwu status

# 停止
kaiwu stop
```

API 兼容 OpenAI 格式。任何支持 OpenAI API 的工具都可以直接用。

## 进阶用法

```bash
# 指定上下文大小（任意值，不必是 2 的幂次）
kaiwu run Qwen3-8B --ctx-size 12000

# 强制重新调参（换了硬件，或者想重新优化）
kaiwu run Qwen3-8B --reset

# 快速启动——跳过 warmup，直接用缓存
kaiwu run Qwen3-8B --fast

# 列出可用模型
kaiwu list

# 自动配置 IDE
kaiwu inject
```

## 自动调整的参数

所有你在 LM Studio 里需要手动配的东西：

| 参数 | Kaiwu 怎么决定 |
|---|---|
| 上下文长度 | 从模型最大值往下探，找速度 ≥ 20 tok/s 的最大值 |
| KV cache 类型 | 算 f16 占多少显存，够就用 f16，不够降到 q8_0+q4_0 |
| MoE expert 位置 | 自动识别 `.ffn_.*_exps.` 张量，路由到 CPU |
| ubatch 大小 | 实测 128 vs 512，取快的 |
| 线程数 | 全 GPU 用 2，MoE offload 用物理核 /2 |
| mlock | 内存余量 > 30% 时自动开，防止模型被换出到磁盘 |
| 多卡分配 | 按显存比例自动切分 |

## 硬件要求

- **显卡**：NVIDIA（CUDA）——建议 4GB+ 显存
- **系统**：Windows 10/11，Linux（Ubuntu 20.04+）
- **内存**：8GB+（30B MoE 模型建议 16GB+）
- **模型格式**：GGUF

支持纯 CPU 推理，但不是主要使用场景。

## 命令列表

| 命令 | 说明 |
|---|---|
| `run <模型>` | 启动模型，没有会自动下载 |
| `stop` | 停止运行中的模型 |
| `status` | 显示当前模型、速度、显存占用 |
| `list` | 列出可用和已下载的模型 |
| `probe` | 显示检测到的硬件信息 |
| `inject` | 自动配置 Continue/Cursor 接入 Kaiwu |
| `version` | 显示版本号 |

---

## For Developers / 贡献者

Build from source (requires Go 1.22+):

```bash
git clone https://github.com/val1813/kaiwu.git
cd kaiwu
make build-windows   # or build-linux
```

---

<div align="center">

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) · by [llmbbs.ai](https://llmbbs.ai)

</div>
