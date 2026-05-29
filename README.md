# gollama

A lightweight llama.cpp instance manager with full parameter control.
Spin up any model on any port with any llama-server flag.

## Install

Build from source:

```bash
git clone https://github.com/majidkorai/gollama
cd gollama
go build -o gollama .
sudo cp gollama /usr/local/bin/
```

## Quick Start

```bash
# 1. Interactive install — detects GPU, choose CPU/CUDA/Vulkan
gollama update

# 2. Pull a model from HuggingFace
gollama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M

# 3. Run with custom flags (blocking, shows output)
gollama run Qwopus3.6-27B-v2-Q4_K_M.gguf --tensor-split 12,8 --flash-attn on

# 4. Start the web UI manager
gollama serve
```

## Commands

| Command | Description |
|---------|-------------|
| `gollama update` | Interactive install — detects GPU, choose CPU/CUDA/Vulkan/ROCm |
| `gollama pull <model>` | Download a GGUF model from HuggingFace |
| `gollama list` | List downloaded models |
| `gollama run <model> [flags]` | Run a model on port 8081 (blocking, shows output) |
| `gollama serve [port]` | Web UI + REST API on :9080 |
| `gollama ps` | List running instances |
| `gollama stop <port>` | Stop an instance |
| `gollama --version` | Show version |

## Web UI

Open **http://localhost:9080** in your browser.

- **Pull models** — paste a HuggingFace URL and download from the browser
- **Launch instances** — select a model, set flags, pick a port
- **Chat** — built-in chat panel with running instances (proxied, no CORS)
- **Monitor** — running instances with port/PID/status badge
- **Stop** — one-click instance termination
- **Logs** — view llama-server stderr directly in the browser

## Custom Flags

Any llama-server flag works. Common ones:

| Flag | Description |
|------|-------------|
| `--n-gpu-layers 99` | Offload all layers to GPU |
| `--tensor-split 12,8` | Manual GPU split (e.g. 3060:12GB, 2080:8GB) |
| `--ctx-size 4096` | Context window |
| `--flash-attn on` | Flash attention |
| `--cont-batching` | Continuous batching |
| `--cache-type-k q4_0` | KV cache quantization (reduces VRAM) |
| `--spec-type draft-mtp` | MTP speculative decoding |
| `-np N` | Parallel slots |

## Architecture

```
~/.gollama/
├── bin/
│   └── llama-server        # Inference engine (auto-downloaded or built)
├── models/
│   └── *.gguf              # Downloaded models
├── logs/
│   └── port-NNNN.log       # Instance logs (viewable from web UI)
└── index.json              # Model registry
```

## Notes

- **VRAM**: gollama needs free GPU memory. Stop Ollama (`systemctl stop ollama`)
  before launching instances if both use the same GPUs.
- **Linux CUDA**: pre-built CUDA binaries are not available for Linux on the
  llama.cpp release page. Build from source or use the Vulkan build instead.
  See `gollama update` for build instructions.
- **Multi-instance**: each instance runs on its own port (8081, 8082, ...).
  Chat with any running instance from the web UI.

## Why gollama?

Ollama hides llama.cpp flags and hardcodes defaults. gollama exposes every
parameter while keeping convenience — model management, web UI, multi-instance,
built-in chat, and full llama-server control. Perfect for multi-GPU setups,
MTP testing, or when you need precise control over inference.

## Development Roadmap

### Phase 1 — Done
- [x] CLI: `pull`, `list`, `run`, `serve`, `ps`, `stop`, `update`, `--version`
- [x] Model pull from HuggingFace (direct download, no Ollama dependency)
- [x] llama-server auto-download with interactive GPU detection
- [x] Web UI: pull models, launch/stop instances, chat, log viewer
- [x] CORS-free chat proxy
- [x] Instance log capture (stderr written to `~/.gollama/logs/`)

### Phase 2 — Next
- [ ] Presets: save/load flag configurations
- [ ] Token/s metrics display in web UI
- [ ] Auto-download CUDA builds for Linux from custom CI
- [ ] Model info display (architecture, quantization, context length)
- [ ] Health checks and auto-restart for crashed instances

### Phase 3 — Future
- [ ] Multi-host support (distribute across machines)
- [ ] REST API documentation
- [ ] VS Code extension integration
- [ ] Install script (`curl ... | sh`)

## Building from Source

```bash
git clone https://github.com/majidkorai/gollama
cd gollama
go build -o gollama .
```
## User interface 

<img width="1393" height="709" alt="Screenshot 2026-05-29 at 3 19 03 PM" src="https://github.com/user-attachments/assets/62b986c2-4c6b-40e7-be4e-9ef40140ed86" />

