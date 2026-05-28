# go-llama

A lightweight llama.cpp instance manager with full parameter control.

Single binary — start a model with any flags on any port.

## Install

```bash
go install github.com/majidkorai/go-llama@latest
```

Or build from source:

```bash
git clone https://github.com/majidkorai/go-llama
cd go-llama
go build -o go-llama .
```

## Usage

```bash
# Start manager with web UI on :9080
./go-llama serve

# Quick-run a model on port 8081
./go-llama run qwen3.6:35b --tensor-split 12,8

# List running instances
./go-llama ps

# Stop an instance
./go-llama stop 8081
```

## Web UI

Open `http://localhost:9080` in your browser to manage instances, select models, and configure flags through a GUI.

## Flags

Any llama-server flag can be passed. Common ones:

| Flag | Description |
|------|-------------|
| `--n-gpu-layers 99` | Offload all layers to GPU |
| `--tensor-split 12,8` | Split across GPUs by VRAM |
| `--ctx-size 4096` | Context window |
| `--flash-attn on` | Flash attention |
| `--cont-batching` | Continuous batching |
| `--spec-type draft-mtp` | Enable MTP speculative decoding |
| `--cache-type-k q4_0` | KV cache quantization |

## Why?

Ollama hides llama.cpp flags. go-llama exposes them all while keeping convenience — multi-instance, presets, web UI, and full llama-server control.
