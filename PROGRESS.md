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

### 本地实测 2（2026-04-23/24, RTX 5060 Laptop 8GB, 16GB RAM）

#### Qwen3-8B Q5_K_M (5.6GB), dense, iso3+FA+MTP

```
模式          ctx     速度         VRAM       备注
──────────────────────────────────────────────────────────
No Think      32K     22.0 tok/s   7.2/8 GB   iso3 生效
Thinking      32K     15.5 tok/s   7.2/8 GB   1024 tok 未 think 完
Prompt(261t)  32K     261.6 tok/s  7.2/8 GB   oobabooga 公式零重试
```

#### Qwen3-30B-A3B Q3_K_XL (13GB), MoE offload, iso3+FA

```
模式          ctx     速度         VRAM       备注
──────────────────────────────────────────────────────────
No Think      4K      14.2 tok/s   2.4/8 GB   RAM 不足(5GB/11.6GB)，warmup 失败
Thinking      4K      11.2 tok/s   2.4/8 GB   experts on CPU
```

#### Llama 3.1 8B Q5_K_M (5.4GB), dense, iso3+FA（无 MTP）

```
模式          ctx     速度         VRAM       备注
──────────────────────────────────────────────────────────
Generation    4K      41.5 tok/s   6.9/8 GB   不带 thinking，速度快
Prompt(542t)  4K      ~218 tok/s   6.9/8 GB   warmup: 37.2 tok/s
```

**iso3 适配结论：**
- iso3 对所有模型默认启用（`HasIsoQuant: true`），运行时检测 llama-server 是否支持
- Llama 3.1 的 iso3 确实生效（启动参数含 `-ctk iso3 -ctv iso3`）
- ctx 只有 4K 不是 iso3 不支持，而是 VRAM 不足（oobabooga 公式反解）
- Qwen3-8B 能拿到 32K 是因为启动时 free VRAM 更多（~6.5GB vs ~5.9GB）

### VPS AB 测试（2026-04-23, 双 RTX 4090 24GB, 128GB RAM）

模型：Qwen3.6-35B-A3B UD-Q4_K_XL, ctx=512K, 4 slots

```
指标                A组: Kaiwu (iso3+MTP)    B组: llama-server 默认
──────────────────────────────────────────────────────────────────
No Think 速度       126.5 tok/s              136.1 tok/s
Thinking 速度       125.5 tok/s (217 tok)    138.5 tok/s (232 tok)
Prompt(524t)        0.72s                    0.65s
GPU0 VRAM           18,546 MiB               23,016 MiB
GPU1 VRAM           17,216 MiB               20,362 MiB
总 VRAM             35,762 MiB               43,378 MiB
RAM                 4,089 MB                 3,736 MB
```

**AB 测试结论：**
- 速度：默认参数反而快 ~8%（iso3 在 VRAM 充裕时有额外解码开销）
- VRAM：Kaiwu iso3 省 7.6 GB（35.7 vs 43.4 GB）
- iso3 的价值在 VRAM 紧张设备上（8GB 5060），不在双 4090 场景

### 体验优化：速度阈值 + 多参数调优（✅ 已完成, 2026-04-24）

**问题：** warmup 只找"能启动的最大 ctx"，不管速度。64K ctx 启动成功但只有 7.8 tok/s，体验极差。

**改动：**
1. **速度感知 ctx 选择** — Phase 1 循环加 `minAcceptableTPS = 20.0` 检查，速度不够就 ctx 减半继续探测
2. **threads 修正** — full_gpu: 2 线程（GPU 推理不需要 CPU），moe_offload: max(Cores/2, 4)
3. **RAM 安全检查** — 启动前检查 RAM 余量，不足时警告（reserve = max(Total×20%, 2GB)）
4. **mlock 支持** — RAM 余量 > 30% 时自动加 `--mlock`，防止模型被 swap 到磁盘

**改动文件：** `optimizer/warmup.go`, `optimizer/params.go`, `engine/runner.go`

**Llama 3.1 8B 实测（RTX 5060 8GB, 16GB RAM）：**

```
探测过程                    改动前          改动后
──────────────────────────────────────────────────────
Probe 1: ctx=64K           7.8 tok/s ✅    7.0 tok/s (< 20, too slow)
Probe 2: ctx=32K           —               10.5 tok/s (< 20, too slow)
Probe 3: ctx=16K           —               13.8 tok/s (< 20, too slow)
Probe 4: ctx=8K            —               19.4 tok/s (fallback, best)
最终结果                    64K / 7.8 tok/s  8K / 19.4 tok/s
```

**结论：** 速度从 7.8 → 19.4 tok/s（2.5 倍提升），ctx 从 64K 降到 8K 但体验大幅改善。8K ctx 对普通对话足够，长文档场景可手动 `--ctx-size 32768` 覆盖。

### iso3 性能排查 + LM Studio 对比（2026-04-24）

**问题：** Kaiwu 19.4 tok/s vs LM Studio 46.5 tok/s，同模型同 VRAM 速度差一倍。

**排查过程：** 逐个参数排查，找到三个瓶颈。

**Llama 3.1 8B Q5_K_M @ RTX 5060 8GB 完整对比：**

```
配置                      KV cache    parallel  ctx    速度(实际)   VRAM
──────────────────────────────────────────────────────────────────────────
起点 (iso3+4slot)         iso3+iso3   4         8K     19.4 tok/s   7.2 GB
关 iso3 (q8_0+4slot)      q8_0+q4_0   4        64K     18.5 tok/s   7.2 GB
关 iso3+1slot             q8_0+q4_0   1         8K     38.2 tok/s   6.4 GB
f16 KV+1slot              f16+f16     1         8K     51.7 tok/s   7.0 GB
f16 KV+1slot (自动)       f16+f16     1        64K     26.2 tok/s   7.3 GB  ← 最终版
LM Studio                 未知        未知       8K     46.5 tok/s   7.2 GB
```

**三个瓶颈及影响：**
1. **iso3 解码开销** — 19.4 → 38.2 tok/s（+97%），8GB 机器 iso3 不划算
2. **4 slot → 1 slot** — 实际推理 18.5 → 38.2 tok/s（+106%），单用户不需要多 slot
3. **q4_0 V cache → f16** — 38.2 → 51.7 tok/s（+35%），量化解码有代价

**最终版本效果：**
- 64K ctx / 26.2 tok/s（起点 7.8 tok/s，提升 3.4 倍）
- 同 8K ctx 对比 LM Studio：51.7 vs 46.5 tok/s（Kaiwu 快 11%）
- ctx 大 8 倍（64K vs 8K），速度只慢 43%，trade-off 合理

**代码改动（最终版 v1）：**

KV cache 动态策略（`optimizer/warmup.go` + `engine/runner.go`）：
- VRAM ≤ 8GB：f16+f16（速度优先）
- VRAM 8-16GB：q8_0+q4_0（平衡）
- VRAM > 16GB：iso3+iso3（ctx 优先）

其他改动：
- `--parallel 1`（单用户模式，所有 VRAM 档位）
- `--ctx-size` 用户覆盖 bug 修复（warmup 读取 `profile.CtxOverride`）
- threads: full_gpu=2, moe_offload=max(Cores/2, 4)
- RAM 安全检查 + mlock 自动启用

### 参数自适应优化 v2（2026-04-24）

**目标：** 消除硬编码档位，改为基于 VRAM 计算的自适应策略。

**改动内容：**

1. **ubatch 探测**（`warmup.go` Phase 2）
   - 实测 128 vs 512，选速度快的
   - 不写死，让实测决定
   - 修复 bug：Phase 2 结果总是覆盖 Phase 1（独立比较）

2. **KV cache 计算驱动**（新文件 `model/kv_cache.go`）
   - 公式：`KV_MB = 2 * layers * kv_heads * head_dim * ctx * bytes_per_element / (1024*1024)`
   - 策略：算 f16 KV cache 占多少显存，`freeVRAM - modelSize - f16KV > 1024MB` → 用 f16，否则降到 q8_0+q4_0
   - 不再按 8/16/24GB 档位硬编码，完全基于计算

3. **新增参数**
   - `--kv-unified`（多 slot 共享 KV cache，减少碎片）
   - `--fit on`（llama.cpp 自动适配未设置参数）

**测试结果（Llama 3.1 8B Q5_K_M @ RTX 5060 8GB）：**

```
配置                ctx     KV cache    ubatch  速度(warmup)  速度(实际)   VRAM
────────────────────────────────────────────────────────────────────────────
v1 (档位逻辑)       64K     f16+f16     128     26.2 tok/s    ~20 tok/s    7.3 GB
v2 (计算驱动)       64K     f16+f16     512     28.7 tok/s    ~20 tok/s    7.6 GB
v2 (计算驱动)       8K      f16+f16     512     51.1 tok/s    56.7 tok/s   7.5 GB
LM Studio           8K      未知        未知    —             46.5 tok/s   7.2 GB
```

**关键发现：**

1. **ubatch=512 比 128 快**
   - 8K ctx: 51.1 vs 47.5 tok/s (+7.6%)
   - 64K ctx: 28.7 vs 23.6 tok/s (+21.6%)

2. **ctx 对速度影响巨大**
   - 8K ctx: 56.7 tok/s（VRAM 92%）
   - 64K ctx: 20.0 tok/s（VRAM 94%）
   - ctx 大 8 倍，速度慢 2.8 倍

3. **Kaiwu 8K 超过 LM Studio 22%**
   - Kaiwu: 56.7 tok/s
   - LM Studio: 46.5 tok/s
   - 同配置（8K ctx, f16 KV, Q5_K_M）

4. **KV cache 自适应生效**
   - 8GB 机器自动选 f16（计算判断装得下）
   - 64K ctx 下 f16 KV 占 ~2.3 GB（模型 5.3 + KV 2.3 + 余量 = 7.6 GB）

**代码改动（v2）：**

- `model/kv_cache.go`：新增 `EstimateKVCacheMB()` 和 `SelectKVCacheType()` 方法
- `optimizer/warmup.go`：Phase 2 改为 ubatch 探测（128 vs 512），修复覆盖逻辑
- `engine/runner.go`：同步 KV cache 自适应策略
- `optimizer/params.go`：统一 KV cache 逻辑，删除旧的 `kvCacheVByVRAM()`
- 三处 BuildArgs/buildArgs 加 `--kv-unified` 和 `--fit on`

**下一步待调：**
- warmup 测速 vs 实际推理速度差异（64K ctx 下 warmup 28.7 但实际 20）
- 多用户/IDE 场景的 parallel 策略（kaiwu serve 模式）
- 冷启动优化（首次请求 4.7-10.9 tok/s）

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

4. **autodetect 模型 ctx 过小 (4K)** — ✅ 已修复 (2026-04-24)
   - 根因：warmup 用 oobabooga 公式预测 ctx，公式用 q8_0/f16 拟合，严重高估 iso3 VRAM
   - Llama 3.1 8B 只拿到 4K ctx，而同参数 Qwen3-8B 拿到 32K（取决于启动时 free VRAM 波动）
   - 修复：warmup 改为二分探测 — 从 min(nativeCtx, 65536) 开始，OOM 就减半，最多 5 次
   - 让 llama-server 自己决定能跑多少，不再依赖公式预测
   - 效果：Llama 3.1 8B ctx 从 4K → 64K（16 倍），iso3 全模型统一行为
   - 改动文件：`internal/optimizer/warmup.go`（BuildArgs 加 ctxSize 参数，Warmup 改为 probe+tune 两阶段）

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
- llama-server: turboquant iso3 版本 (b8864)，iso3 全模型默认启用
- 已部署模型：Qwen3-8B Q5_K_M, Llama-3.1-8B Q5_K_M
- VPS 模型：Qwen3.6-35B-A3B UD-Q4_K_XL, UD-Q5_K_XL
