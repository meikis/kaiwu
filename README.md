# Kaiwu

**Run 30B models on 8GB GPUs. No tuning required.**

Kaiwu is a local LLM deployment tool that makes large Mixture-of-Experts models practical on consumer hardware. It intelligently offloads MoE experts between GPU and CPU, delivering usable inference speeds where other tools fall short — without any manual configuration.

Built in Go with a llama.cpp backend, Kaiwu targets the gap between "fits in VRAM" and "too slow to use."

## Features

- **Zero-config auto-optimization** — Kaiwu profiles your hardware and automatically determines the optimal GPU/CPU split for expert offloading. No layer counts to guess, no config files to edit.

- **MoE expert offload for 8GB GPUs** — Run 30B+ parameter MoE models (e.g. Qwen3-30B-A3B, Qwen3.6-35B-A3B) on a single 8GB GPU by selectively keeping hot experts in VRAM and routing the rest through CPU.

- **7x faster than naive offload** — Benchmarked at **21 tok/s** on a 30B MoE model with an RTX 5060 Laptop 8GB, compared to **3 tok/s** with LM Studio's default layer-based offload strategy.

- **Cross-platform** — Native support for Windows and Linux. Single binary, no Python environment or dependency management needed.

- **OpenAI-compatible API** — Drop-in `/v1/chat/completions` endpoint. Works with any client that speaks the OpenAI API — Continue, Cursor, Open WebUI, or your own scripts.

## Installation

### Download Pre-built Binaries

Download the latest release for your platform:

**Windows:**
```bash
# Download kaiwu.exe from releases
# Add to PATH or run directly
kaiwu.exe version
```

**Linux:**
```bash
# Download kaiwu binary from releases
chmod +x kaiwu
sudo mv kaiwu /usr/local/bin/
kaiwu version
```

### Build from Source

Requirements: Go 1.22+

```bash
git clone https://github.com/val1813/kaiwu.git
cd kaiwu

# Windows
make build-windows

# Linux
make build-linux

# Binary will be in dist/ directory
```

## Quick Start

### Run a Model

Start a model server:

```bash
kaiwu run Qwen3-30B-A3B
```

The model will be available at `http://localhost:11435/v1` (OpenAI-compatible API).

### Check Status

```bash
kaiwu status
```

Output shows running models, ports, and resource usage.

### Stop a Model

```bash
kaiwu stop
```

### List Available Models

```bash
kaiwu list
```

## Available Commands

| Command | Description |
|---------|-------------|
| `run` | Start a model server |
| `stop` | Stop running model(s) |
| `status` | Show running models and system status |
| `list` | List available models |
| `probe` | Output hardware fingerprint |
| `inject` | Inject IDE configuration (Continue, Cursor) |
| `bench` | Run performance benchmarks |
| `config` | Manage configuration |
| `version` | Show version information |

Use `kaiwu <command> --help` for detailed usage.

## Benchmark Results

Tested on a laptop with RTX 5060 8GB / i7-13700HX / 16GB RAM, Windows 11.  
Kaiwu runs zero-config; LM Studio uses best manual settings (`--gpu max`, GPU offload=48).

### 30B MoE on 8GB GPU — Core Advantage

Model: Qwen3-30B-A3B (MoE, 30B total / 3B active), quantization: UD-Q3_K_XL (13.8GB)

| Metric | LM Studio | Kaiwu | Delta |
|--------|-----------|-------|-------|
| Speed (tok/s) | 3.0 | 21.0 | **7x faster** |
| VRAM | 7,549 MB (93%) | 2,603 MB (32%) | **-65%** |
| GPU Utilization | 100% | 14% | |
| RAM | 11.5 GB | 15.7 GB | |
| Config required | Manual (GPU offload layers, parallel, batch size) | None | |

LM Studio loads 7.5GB into VRAM and saturates the GPU. Kaiwu places only attention layers on GPU and runs MoE experts on CPU, cutting VRAM by 65% while running 7x faster.

### 8B Dense — Everyday Use

Model: Qwen3-8B-Q5_K_M (dense, 5.7GB)

| Metric | LM Studio | Kaiwu | Delta |
|--------|-----------|-------|-------|
| Speed (tok/s) | 44.3 | 42.4 | ~equal |
| VRAM | 7,476 MB | 7,473 MB | equal |

When the model fits entirely in VRAM, both tools perform the same. Kaiwu does not slow you down on standard models.

### Dual 4090 — High-End Scaling

Qwen3.6-35B-A3B on dual RTX 4090 24GB: **115 tok/s**, linear GPU scaling.

> Full data, methodology, and reproduction steps: [`benchmark/BENCHMARK_REPORT.md`](benchmark/BENCHMARK_REPORT.md)  
> Comparison charts: [`benchmark/comparison_chart.png`](benchmark/comparison_chart.png) | [`benchmark/30b_moe_highlight.png`](benchmark/30b_moe_highlight.png)

## How It Works

1. **Hardware detection** — On first run, Kaiwu probes GPU model, VRAM, CPU cores, and available RAM to build a hardware profile.

2. **Automatic parameter optimization** — Based on the hardware profile and model architecture (dense vs MoE), Kaiwu calculates GPU offload layers, MoE expert placement, batch size, context length, and thread count. For MoE models, it offloads only the `.ffn_.*_exps.` expert weights to CPU, keeping attention and routing layers on GPU.

3. **Warmup benchmark with caching** — After loading, Kaiwu runs a short warmup pass and caches the optimal configuration. Subsequent launches skip the tuning step.

4. **Real-time monitoring** — During inference, a proxy layer tracks token throughput, VRAM pressure, and detects degenerate output loops, restarting the backend if needed.

## Architecture

```
┌─────────────────────────────────────────────┐
│  CLI (Go)                                   │
│  kaiwu run <model> [--fast] [--bench]     │
├─────────────────────────────────────────────┤
│  Proxy Layer (Go)                           │
│  - OpenAI-compatible API (/v1/chat/...)     │
│  - Token throughput monitoring              │
│  - Output loop detection & auto-restart     │
│  - Request logging                          │
├─────────────────────────────────────────────┤
│  llama.cpp backend (llama-server)           │
│  - Model loading with computed offload map  │
│  - MoE expert CPU/GPU split                 │
│  - Multi-GPU tensor parallelism             │
└─────────────────────────────────────────────┘
```

- **Go CLI** — Single binary, parses flags, orchestrates hardware detection and server lifecycle.
- **llama-server** — The inference engine. Kaiwu generates launch parameters (layer offload, thread count, batch size) and manages the process.
- **Proxy layer** — Sits between the client and llama-server. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Monitors throughput and catches degenerate loops (repeated token sequences) by killing and restarting the backend.
- **API endpoint** — Any OpenAI-compatible client (Continue, Cursor, Open WebUI, custom scripts) connects to `http://localhost:11435/v1/chat/completions`.

## Configuration

Kaiwu stores its configuration in `~/.kaiwu/config.toml`. On first run, it generates a default config with sensible defaults.

Key settings:

- `proxy_port` — Port for the OpenAI-compatible API (default: 11435)
- `llama_port` — Port for the llama.cpp backend (default: 11434)

Model storage and cache:

- `~/.kaiwu/models/` — Downloaded model files
- `~/.kaiwu/profiles/` — Hardware profiles and optimization cache

Most users won't need to edit the config. Kaiwu auto-detects hardware and optimizes accordingly.

## Development

### Project Structure

```
cmd/kaiwu/          # Main entry point
internal/
  config/           # Configuration management
  download/         # Model download and verification
  engine/           # llama.cpp process management
  hardware/         # GPU/CPU detection and profiling
  ide/              # IDE integration (Continue, Cursor)
  model/            # Model store and matching
  monitor/          # Performance monitoring
  optimizer/        # Expert offload optimization
  proxy/            # OpenAI API proxy layer
```

### Building

```bash
# Windows
make build-windows

# Linux
make build-linux

# Cross-platform verification
make verify
```

### Testing

```bash
make test
```

All new features should include tests. Run the full test suite before submitting PRs.

## Contributing

Issues and pull requests are welcome. Before submitting:

- Follow standard Go conventions (gofmt, golint)
- Add tests for new features
- Ensure `make test` and `make verify` pass

For major changes, open an issue first to discuss the approach.

## License

See LICENSE file.

## Acknowledgments

Kaiwu is built on [llama.cpp](https://github.com/ggerganov/llama.cpp) and inspired by [Ollama](https://ollama.ai) and [LM Studio](https://lmstudio.ai).
