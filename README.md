# go-llama

A lightweight llama.cpp instance manager with full parameter control. Spin up any model on any port with any llama-server flag.

## Install

```bash
curl -fsSL https://go-llama.dev/install.sh | sh
```

Or build from source:

```bash
git clone https://github.com/majidkorai/go-llama
cd go-llama
go build -o go-llama .
```

## Quick Start

```bash
# 1. Download the latest llama-server binary
#    Detects your GPU and offers CPU/CUDA/Vulkan options
go-llama update

# 2. Pull a model from HuggingFace
go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M

# 3. Run it with custom flags
go-llama run Qwopus3.6-27B-v2-Q4_K_M.gguf --tensor-split 12,8 --flash-attn on --cont-batching

# 4. Start the web UI manager
go-llama serve
```

## Commands

| Command | Description |
|---------|-------------|
| `go-llama update` | Interactive install — detects GPU, lets you choose CPU/CUDA/Vulkan build |
| `go-llama pull <model>` | Download a GGUF model from HuggingFace |
| `go-llama list` | List downloaded models |
| `go-llama run <model> [flags]` | Run a model on port 8081 with custom flags |
| `go-llama serve [port]` | Web UI + REST API on :9080 |
| `go-llama ps` | List running instances (via server) |
| `go-llama stop <port>` | Stop an instance (via server) |

## Model Pull

Pull any GGUF model from HuggingFace:

```bash
go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M
go-llama pull hf.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q6_K
go-llama pull hf.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF:Balanced
```

Models are stored in `~/.go-llama/models/`.

## Custom Flags

Pass any llama-server flag directly:

```bash
go-llama run Qwopus3.6-27B-v2-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --tensor-split 12,8 \
  --ctx-size 4096 \
  --flash-attn on \
  --cont-batching \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp
```

### Common Flags

| Flag | Description |
|------|-------------|
| `--n-gpu-layers 99` | Offload all layers to GPU |
| `--tensor-split 12,8` | Manual GPU split (3060:12GB, 2080:8GB) |
| `--ctx-size N` | Context window |
| `--flash-attn on` | Flash attention |
| `--cont-batching` | Continuous batching |
| `--cache-type-k q4_0` | KV cache quantization |
| `--spec-type draft-mtp` | MTP speculative decoding |
| `-np N` | Parallel slots |

## Install Options

The `go-llama update` command interactively detects:

- **CUDA** — NVIDIA GPUs via `nvidia-smi`
- **ROCm** — AMD GPUs via `/opt/rocm`
- **Vulkan** — Cross-platform GPU acceleration (works with NVIDIA too)
- **CPU** — No GPU required

On Linux, CUDA requires building from source (instructions provided). Vulkan and CPU are pre-built.

## Web UI

Start the web UI to manage everything from your browser:

```bash
go-llama serve
```

Open `http://localhost:9080` to pull models, launch instances, set flags, and monitor running servers.

## Architecture

```
~/.go-llama/
├── bin/
│   └── llama-server        # Auto-downloaded inference engine
├── models/
│   └── *.gguf              # Downloaded models
└── index.json              # Model registry
```

## Why go-llama?

Ollama hides llama.cpp flags and hardcodes defaults. go-llama exposes every llama-server parameter while keeping the convenience of model management. Perfect for multi-GPU setups, MTP testing, or when you need precise control over inference.

## Building from Source

```bash
git clone https://github.com/majidkorai/go-llama
cd go-llama
go build -o go-llama .
```
