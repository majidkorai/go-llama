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
# 1. Download the latest llama-server binary (detects GPU)
go-llama update

# 2. Pull a model from HuggingFace
go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M

# 3. Run it with custom flags (blocking mode)
go-llama run Qwopus3.6-27B-v2-Q4_K_M.gguf --tensor-split 12,8 --flash-attn on

# 4. Start the web UI manager
go-llama serve
```

## Commands

| Command | Description |
|---------|-------------|
| `go-llama update` | Interactive install — detects GPU, choose CPU/CUDA/Vulkan build |
| `go-llama pull <model>` | Download a GGUF model from HuggingFace |
| `go-llama list` | List downloaded models |
| `go-llama run <model> [flags]` | Run a model on port 8081 (blocking, shows output) |
| `go-llama serve [port]` | Web UI + REST API |
| `go-llama ps` | List running instances |
| `go-llama stop <port>` | Stop an instance |
| `go-llama --version` | Show version |

## Web UI

Start the web UI:

```bash
go-llama serve
```

Open **http://localhost:9080** in your browser. You can:

- **Pull models** from HuggingFace directly in the browser
- **Launch instances** with custom flags from the UI
- **Chat** with running instances via the built-in chat panel
- **Monitor** running instances with port/PID/status
- **Stop** instances with one click

## Custom Flags

Pass any llama-server flag directly. Common ones:

| Flag | Description |
|------|-------------|
| `--n-gpu-layers 99` | Offload all layers to GPU |
| `--tensor-split 12,8` | Manual GPU split for multi-GPU |
| `--ctx-size 4096` | Context window |
| `--flash-attn on` | Flash attention |
| `--cont-batching` | Continuous batching |
| `--cache-type-k q4_0` | KV cache quantization |
| `--spec-type draft-mtp` | MTP speculative decoding |
| `-np N` | Parallel slots |

## Architecture

```
~/.go-llama/
├── bin/
│   └── llama-server        # Inference engine
├── models/
│   └── *.gguf              # Downloaded models
└── index.json              # Model registry
```

Models are pulled from HuggingFace directly. llama-server is downloaded from GitHub releases (or built from source for CUDA on Linux).

## Why go-llama?

Ollama hides llama.cpp flags and hardcodes defaults. go-llama exposes every parameter while keeping convenience — model management, web UI, multi-instance, and chat. Perfect for multi-GPU setups, MTP testing, or when you need precise control over inference.

## Development Roadmap

### Phase 1 — Done
- [x] CLI: `pull`, `list`, `run`, `serve`, `ps`, `stop`, `update`
- [x] Model pull from HuggingFace
- [x] llama-server auto-download with GPU detection
- [x] Web UI: pull models, launch instances, stop, chat
- [x] CORS-free chat proxy
- [x] Interactive install (CPU/CUDA/Vulkan choice)

### Phase 2 — Next
- [ ] Presets: save/load flag configurations
- [ ] Token/s metrics display in web UI
- [ ] Auto-download CUDA builds for Linux from custom CI
- [ ] Model info display (architecture, quantization, context length)
- [ ] Log streaming to web UI

### Phase 3 — Future
- [ ] `go-llama push` — share presets
- [ ] Multi-host support (distribute across machines)
- [ ] REST API documentation
- [ ] VS Code extension integration
- [ ] Install script (`curl ... | sh`)

## Building from Source

```bash
git clone https://github.com/majidkorai/go-llama
cd go-llama
go build -o go-llama .
```
