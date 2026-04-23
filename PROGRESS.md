# Kaiwu 上下文优化项目进度

## 目标
解决本地模型上下文太短的问题，让 Kaiwu 能做真实的长项目开发。

## 三步方案

### 第一步：上下文扩容（✅ 已完成）

**改动文件：**
- `internal/optimizer/params.go` - ctx 从 4K/8K/16K 改为 32K/64K/128K，V cache 改 q4_0
- `internal/engine/runner.go` - ctx 同步更新，V cache 改 q4_0
- `internal/engine/oom.go` - KV cache 估算修正（GQA 用 4 kv_heads），system reserve 降到 300MB
- `internal/model/matcher.go` - 本地文件优先，放宽 VRAM 阈值
- `internal/proxy/context.go` - 上下文提示措辞改为不建议"新开对话"

**效果验证：**
- ✅ 8GB 显存：4K → 32K 上下文（8倍提升）
- ✅ KV cache：2448 MB → 1872 MB（省 576 MB 显存）
- ✅ 推理速度：29.1 → 39.3 tok/s（提升 35%）
- ✅ 日志确认：`n_ctx = 32768`, `K (q8_0): 1224 MiB, V (q4_0): 648 MiB`

**对比 LM Studio：**
- LM Studio 默认：2048-4096 tokens
- Kaiwu 1.2.0：32768 tokens（8倍）

**版本：** Kaiwu 1.2.0

---

### 第二步：对话压缩（✅ 已完成）

**目标：**
上下文达到 75% 时自动压缩历史对话，避免用户手动"新开对话"。

**方案：** 纯算法压缩（extractive summary）
- 头部保护：系统提示 + 首轮对话
- 尾部保护：最近 8K tokens
- 中间压缩：提取关键信息（代码路径、函数名、TODO、命令等）

**实现细节：**
- 触发阈值：75% of 32K = 24576 tokens
- 压缩算法：保留每条消息的首行 + 代码块 + 文件路径 + 关键词
- 零延迟：不调用模型，纯算法提取，避免单 slot 阻塞

**效果验证：**
- ✅ 触发正常：36361 tokens → 9843 tokens（省 26518 tokens）
- ✅ 压缩率：73% 压缩率，保留关键信息
- ✅ 零延迟：压缩耗时 <1ms，不影响请求响应

**关键文件：**
- `internal/proxy/handler.go` - 请求拦截和压缩触发
- `internal/proxy/compressor.go` - 压缩逻辑（纯算法）
- `internal/monitor/monitor.go` - 仪表盘显示压缩统计

**版本：** Kaiwu 1.2.1

**踩坑记录：**
- ❌ 最初方案：调用本地模型生成摘要 → 单 slot 阻塞，超时
- ✅ 最终方案：纯算法提取关键信息 → 零延迟，效果好
- 测试方法：8GB VRAM 下 10K+ tokens 的 prompt processing 需要数分钟，测试时用合理超时（300s+）

---

### 第三步：文件 RAG（📋 计划中）

**目标：**
本地 json 存储 + 关键词检索，解决大文件/代码库场景。

**方案：**
- 对话历史 → 用压缩方案（第二步）
- 文件/代码库 → 用 RAG（本步骤）

**工程量：** 约 1-2 周

---

## 当前状态

**已完成：**
- ✅ 第一步：上下文扩容 4K→32K + 非对称 KV cache
- ✅ 第二步：对话压缩（纯算法，零延迟，73% 压缩率）
- ✅ 仪表盘增强：上下文行显示 xK/32K + 压缩统计
- ✅ 动态 ctx 计算：根据 VRAM/模型大小/层数自动算最优值
- ✅ 微调模式：`--ctx-size` 手动指定上下文大小
- ✅ 编译部署 Kaiwu 1.3.0

**下一步：**
- 🔄 第三步：文件 RAG

**技术债：**
- matcher 的本地文件检查逻辑可以优化（现在每次都扫描目录）
- warmup 缓存的 profile 格式需要版本号（避免参数变更后用旧缓存）

---

## IsoQuant 涡轮量化集成（🔄 进行中）

**Spec：** `D:\program\ollama\kaiwu-v4\KAIWU_TWO_VERSION_SPEC.md`
**Fork：** `johndpope/llama-cpp-turboquant@feature/planarquant-kv-cache`
**GitHub：** `https://github.com/val1813/kaiwu`

### 代码改动（✅ 已完成）

| 文件 | 改动 |
|------|------|
| `model/matcher.go` | DeployProfile 加 `HasIsoQuant` 字段，默认 true |
| `engine/runner.go` | buildArgs 改用 `-ctk iso3 -ctv iso3`，不支持时回退 q8_0/q4_0 |
| `engine/binary.go` | 新增 bundled binary 优先级：本地打包 iso3 > 缓存 > 下载官方 |
| `engine/oom.go` | OOM 预检用 iso3 压缩系数 0.75 + 600MB 固定开销 |
| `optimizer/params.go` | DynamicCtxSize 同步 iso3，上限扩到 512K |
| `optimizer/warmup.go` | BuildArgs 同步 iso3 参数 |
| `model/autodetect.go` | MoE 模型 VRAM/RAM 估算修复（shared layers 占比计算） |
| `go.mod` + 全部 import | module path 改为 `github.com/val1813/kaiwu` |

### GitHub Actions CI（✅ 已完成）

**Workflow：** `.github/workflows/build-llama-server.yml`

自动编译 turboquant fork 的 llama-server，支持 iso3 KV cache。

| 平台 | 状态 | Runner | CUDA |
|------|------|--------|------|
| Linux | ✅ 编译成功 | ubuntu-22.04 | apt cuda-toolkit-12-4 |
| Windows | ✅ 编译成功 | windows-2022 | Jimver/cuda-toolkit@v0.2.16 |

**CUDA 架构：** sm_75/80/86/89（Turing/Ampere/Ada Lovelace）
- RTX 50 系列（sm_120）通过 PTX JIT 编译支持，运行时自动处理

**MSVC 编译修复（3 处）：**
1. `ops.cpp`: `extern "C" GGML_API int turbo3_cpu_wht_group_size;` → `int turbo3_cpu_wht_group_size = 1;`（声明改定义）
2. `ggml-turbo-quant.c`: 添加 `#define _USE_MATH_DEFINES`（MSVC 不自带 M_PI）
3. `llama-kv-cache.cpp`: 添加 `float * g_innerq_scale_inv_host = nullptr;`（链接符号）

**产物：**
- `llama-server-turboquant-win-cuda-12.4`（Windows: llama-server-cuda.exe + DLLs）
- `llama-server-turboquant-linux-cuda-12.4`（Linux: llama-server-cuda + .so）
- 保留 90 天，手动触发 workflow_dispatch

**踩坑记录：**
- ❌ 手动 curl 下载 CUDA redist zip → 路径/解压/合并全部出问题，连续失败 4 次
- ❌ PowerShell Move-Item 合并嵌套目录 → cuda_runtime.h 丢失
- ❌ CUDA 12.4 不支持 compute_120 → 去掉 sm_120，用 PTX JIT 代替
- ❌ `extern "C"` 只改声明不改定义 → MSVC LNK2019 链接错误
- ✅ 最终方案：Jimver/cuda-toolkit Action + 3 处 MSVC fix，一次通过

### VPS 实测（✅ 双 4090 24GB×2, 128GB RAM）

模型：Qwen3.6-35B MoE Q5_K_M (25GB), MoE offload

```
配置                     ctx     速度        GPU（双卡合计）
─────────────────────────────────────────────────────────
基线 q8_0+q4_0          128K    48.3 tok/s    5.2 GB
iso3+iso3               128K    50.9 tok/s    5.9 GB
iso3+iso3               256K    47.2 tok/s    8.1 GB
iso3+iso3               512K    50.7 tok/s   12.4 GB
```

关键发现：
- iso3 KV 增量压缩比实测 0.73，和理论 0.75 吻合
- iso3 有 ~600MB 固定解码 buffer 开销（已加入 OOM 预检）
- 512K 上下文速度和 128K 基线持平（50.7 vs 48.3）

### 本地实测（RTX 5060 Laptop 8GB, 16GB RAM）

模型：Qwen3-30B-A3B Q3_K_XL (13GB), MoE offload

```
配置                     ctx     速度        GPU      RAM
──────────────────────────────────────────────────────────
q8_0+q4_0 (官方b8851)   4K     11.1 tok/s   5.0/8.1  —
iso3+iso3 (turboquant)   待测    待测         待测     待测
```

### 待解决问题

1. **Windows 编译 turboquant fork** — ✅ 已完成
   - GitHub Actions CI 编译成功（Windows + Linux）
   - 使用 `Jimver/cuda-toolkit@v0.2.16` 安装 CUDA（不要手动下载）
   - 3 处 MSVC 编译 fix 已集成到 workflow
   - **下一步：** 下载 artifact，部署到本地，测试 iso3 速度

2. **8GB 笔记本跑 30B 内存不足** — ✅ 已修复
   - autodetect.go 新增 `estimateMinVRAM()` 和 `estimateMinRAM()`
   - MoE offload 模式：MinVRAM = shared layers (~10%) + 1.5GB，MinRAM = 80% model + 2GB
   - Qwen3-30B-A3B 13GB：5GB VRAM + 12GB RAM（原 14.4GB + 15.9GB）
   - 16GB RAM 机器现在可以正常跑 30B MoE 模型

3. **中文 prompt UTF-8 编码问题** — ✅ 已修复
   - warmup.go 的 benchmark prompt 已使用纯 ASCII 英文
   - 避免 Windows bash 环境下 UTF-8 编码错误

---

## 参考资料

**前沿研究：**
- [TurboQuant 3-bit KV cache](https://www.thenextgentechinsider.com/pulse/turboquant-integration-enables-3-bit-kv-cache-compression-in-llamacpp) - llama.cpp 已集成，零精度损失
- [Qwen3 MoE GQA 架构](https://www.ikangai.com/the-kv-cache-the-hidden-bottleneck-holding-back-local-ai/) - KV cache 极小，8GB 机器可跑 100+ tok/s
- [Hermes 压缩算法](https://medium.com/@nousresearch/hermes-atlas) - 头尾保护 + 中间摘要

**社区实测：**
- Qwen3.5-35B-A3B 在 8GB 机器上 100+ tok/s
- Qwen3-30B-A3B 用非对称 q8_0-K + turbo3-V 配置，49K tokens NIAH 测试 25/25 全通过

---

## 部署信息

**GitHub：** `https://github.com/val1813/kaiwu`

**二进制位置：** `C:\Users\15488\bin\kaiwu.exe`
**源码位置：** `D:\program\ollama\kaiwu-release\`
**模型目录：** `C:\Users\15488\.kaiwu\models\`
**配置目录：** `C:\Users\15488\.kaiwu\`

**当前版本：** Kaiwu dev (2026-04-23 编译)
- Go module: `github.com/val1813/kaiwu`
- llama-server: 官方 b8851 → 待替换为 turboquant iso3 版本
- 已部署模型：Qwen3-8B Q5_K_M, Qwen3-30B-A3B Q3_K_XL
