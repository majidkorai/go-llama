package main

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// ── Types ──────────────────────────────────────────────────────────────

type Instance struct {
	Port         int      `json:"port"`
	Model        string   `json:"model"`
	PID          int      `json:"pid"`
	Status       string   `json:"status"`
	TokensPerSec float64  `json:"tokens_per_sec,omitempty"`
	Flags        []string `json:"flags,omitempty"`
}

type Preset struct {
	Name  string   `json:"name"`
	Model string   `json:"model"`
	Port  int      `json:"port"`
	Flags []string `json:"flags"`
}

type ModelInfo struct {
	Name         string `json:"name"`
	BlobPath     string `json:"blob_path"`
	Size         int64  `json:"size"`
	Architecture string `json:"architecture,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	Source       string `json:"source,omitempty"` // "ollama" or "local"
}

// ── Helpers ────────────────────────────────────────────────────────────

func goLLamaDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".go-llama")
}

func modelsDir() string {
	return filepath.Join(goLLamaDir(), "models")
}

func binDir() string {
	return filepath.Join(goLLamaDir(), "bin")
}

func indexFile() string {
	return filepath.Join(goLLamaDir(), "index.json")
}

func versionFile() string {
	return filepath.Join(goLLamaDir(), "llama-server-version.txt")
}

func ensureDir(dir string) {
	os.MkdirAll(dir, 0755)
}

func loadIndex() map[string]ModelInfo {
	ensureDir(goLLamaDir())
	data, err := os.ReadFile(indexFile())
	if err != nil {
		return make(map[string]ModelInfo)
	}
	var idx map[string]ModelInfo
	if json.Unmarshal(data, &idx) != nil {
		return make(map[string]ModelInfo)
	}
	return idx
}

func saveIndex(idx map[string]ModelInfo) {
	data, _ := json.MarshalIndent(idx, "", "  ")
	os.WriteFile(indexFile(), data, 0644)
}

// downloadFile downloads a URL to a local path with progress.
func downloadFile(url, dest string) error {
	fmt.Printf("Downloading %s ...\n", url)
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Show progress for large files
	written, err := io.Copy(out, resp.Body)
	if err != nil {
		return err
	}
	_ = written
	fmt.Printf("Downloaded %d bytes\n", written)
	return nil
}

// ── Runner ─────────────────────────────────────────────────────────────

type Manager struct {
	mu        sync.Mutex
	instances map[int]*Instance
	nextPort  int
}

func NewManager() *Manager {
	return &Manager{
		instances: make(map[int]*Instance),
		nextPort:  8081,
	}
}

func findLlamaServer() string {
	// 1. Check go-llama's own bin directory first
	self := filepath.Join(binDir(), "llama-server")
	if _, err := os.Stat(self); err == nil {
		return self
	}
	// 2. Check system PATH
	if path, err := exec.LookPath("llama-server"); err == nil {
		return path
	}
	// 3. Check common locations
	candidates := []string{
		"/usr/local/lib/ollama/llama-server",
		"/usr/local/bin/llama-server",
		"/home/ollama/llama.cpp/build/bin/llama-server",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "llama-server"
}

func resolveModelBlob(model string) (string, error) {
	// Check go-llama local models
	idx := loadIndex()
	if info, ok := idx[model]; ok {
		if _, err := os.Stat(info.BlobPath); err == nil {
			return info.BlobPath, nil
		}
		// Stale index entry — remove it
		delete(idx, model)
		saveIndex(idx)
	}

	// Check direct file path
	if _, err := os.Stat(model); err == nil {
		return model, nil
	}

	return model, nil
}

func (m *Manager) Start(model string, port int, extraArgs []string) (*Instance, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if port == 0 {
		port = m.nextPort
		m.nextPort++
	}

	if _, exists := m.instances[port]; exists {
		return nil, fmt.Errorf("port %d is already in use", port)
	}

	llamaBin := findLlamaServer()
	blob, err := resolveModelBlob(model)
	if err != nil {
		return nil, fmt.Errorf("resolving model: %w", err)
	}

	args := []string{
		"-m", blob,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(port),
		"--no-webui",
	}
	hasNGL, hasTS := false, false
	for _, a := range extraArgs {
		if a == "--n-gpu-layers" || a == "-ngl" {
			hasNGL = true
		}
		if a == "--tensor-split" || a == "-ts" {
			hasTS = true
		}
	}
	if !hasNGL {
		args = append(args, "--n-gpu-layers", "99")
	}
	if !hasTS {
		args = append(args, "--tensor-split", "12,8")
	}
	args = append(args, extraArgs...)

	cmd := exec.Command(llamaBin, args...)
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}

	inst := &Instance{
		Port:   port,
		Model:  model,
		PID:    cmd.Process.Pid,
		Status: "running",
		Flags:  args,
	}
	m.instances[port] = inst

	go func() {
		cmd.Wait()
		m.mu.Lock()
		if inst.Status == "running" {
			inst.Status = "stopped"
		}
		m.mu.Unlock()
	}()

	return inst, nil
}

func (m *Manager) Stop(port int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	inst, ok := m.instances[port]
	if !ok {
		return fmt.Errorf("no instance on port %d", port)
	}

	proc, err := os.FindProcess(inst.PID)
	if err == nil {
		proc.Signal(os.Interrupt)
		proc.Kill()
	}

	inst.Status = "stopped"
	delete(m.instances, port)
	return nil
}

func (m *Manager) List() []*Instance {
	m.mu.Lock()
	defer m.mu.Unlock()

	result := make([]*Instance, 0, len(m.instances))
	for _, inst := range m.instances {
		result = append(result, inst)
	}
	return result
}

// ── Model DB ───────────────────────────────────────────────────────────

func listModels() ([]ModelInfo, error) {
	var models []ModelInfo

	// Scan go-llama's local model storage
	idx := loadIndex()
	for _, info := range idx {
		if _, err := os.Stat(info.BlobPath); err == nil {
			info.Source = "local"
			models = append(models, info)
		}
	}

	return models, nil
}

// ── API Server ─────────────────────────────────────────────────────────

func startServer(mgr *Manager, port string) {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/models", func(w http.ResponseWriter, r *http.Request) {
		models, err := listModels()
		if err != nil {
			jsonError(w, err.Error(), 500)
			return
		}
		if models == nil {
			models = []ModelInfo{}
		}
		jsonResponse(w, models)
	})

	mux.HandleFunc("/api/v1/models/pull", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			jsonError(w, err.Error(), 400)
			return
		}
		if err := pullModel(req.Model); err != nil {
			jsonError(w, err.Error(), 500)
			return
		}
		jsonResponse(w, map[string]string{"status": "ok", "model": req.Model})
	})

	mux.HandleFunc("/api/v1/instances", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			jsonResponse(w, mgr.List())
		case http.MethodPost:
			var req struct {
				Model string   `json:"model"`
				Port  int      `json:"port"`
				Flags []string `json:"flags"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				jsonError(w, err.Error(), 400)
				return
			}
			inst, err := mgr.Start(req.Model, req.Port, req.Flags)
			if err != nil {
				jsonError(w, err.Error(), 500)
				return
			}
			jsonResponse(w, inst)
		default:
			http.Error(w, "method not allowed", 405)
		}
	})

	mux.HandleFunc("/api/v1/instances/stop", func(w http.ResponseWriter, r *http.Request) {
		portStr := r.URL.Query().Get("port")
		port, err := strconv.Atoi(portStr)
		if err != nil {
			jsonError(w, "port query parameter is required", 400)
			return
		}
		if err := mgr.Stop(port); err != nil {
			jsonError(w, err.Error(), 500)
			return
		}
		jsonResponse(w, map[string]string{"status": "stopped"})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, uiPage)
	})

	addr := fmt.Sprintf(":%s", port)
	log.Printf("go-llama listening on %s", addr)
	log.Printf("Web UI: http://localhost%s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func jsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// ── Pull Model ─────────────────────────────────────────────────────────

func pullModel(ref string) error {
	// Parse hf.co/user/repo:quant
	if !strings.HasPrefix(ref, "hf.co/") {
		// Support bare HuggingFace IDs too
		ref = "hf.co/" + ref
	}

	parts := strings.SplitN(ref, ":", 2)
	modelID := strings.TrimPrefix(parts[0], "hf.co/")
	quant := "Q4_K_M"
	if len(parts) > 1 && parts[1] != "" {
		quant = parts[1]
	}

	// Query HuggingFace API for GGUF files
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s", modelID)
	resp, err := http.Get(apiURL)
	if err != nil {
		return fmt.Errorf("fetching model info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("model %s not found on HuggingFace (HTTP %d)", modelID, resp.StatusCode)
	}

	var modelData struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
			Size     int64  `json:"size"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&modelData); err != nil {
		return fmt.Errorf("parsing model info: %w", err)
	}

	// Find the best matching GGUF file
	var targetFile string
	var targetSize int64
	for _, s := range modelData.Siblings {
		if strings.HasSuffix(s.Filename, ".gguf") && strings.Contains(s.Filename, quant) {
			targetFile = s.Filename
			targetSize = s.Size
			break
		}
	}
	if targetFile == "" {
		// Fallback: pick first GGUF
		for _, s := range modelData.Siblings {
			if strings.HasSuffix(s.Filename, ".gguf") {
				targetFile = s.Filename
				targetSize = s.Size
				break
			}
		}
	}
	if targetFile == "" {
		return fmt.Errorf("no GGUF file found in %s", modelID)
	}

	// Download the GGUF
	ensureDir(modelsDir())
	dest := filepath.Join(modelsDir(), filepath.Base(targetFile))

	downloadURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelID, targetFile)
	fmt.Printf("Downloading %s (%s)...\n", targetFile, formatSize(targetSize))

	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("creating file: %w", err)
	}
	defer out.Close()

	dlResp, err := http.Get(downloadURL)
	if err != nil {
		return fmt.Errorf("downloading: %w", err)
	}
	defer dlResp.Body.Close()

	if dlResp.StatusCode != 200 {
		os.Remove(dest)
		return fmt.Errorf("download failed (HTTP %d)", dlResp.StatusCode)
	}

	written, err := io.Copy(out, dlResp.Body)
	if err != nil {
		os.Remove(dest)
		return fmt.Errorf("downloading: %w", err)
	}

	// Register in index
	idx := loadIndex()
	modelName := fmt.Sprintf("hf.co/%s:%s", modelID, quant)
	idx[modelName] = ModelInfo{
		Name:     modelName,
		BlobPath: dest,
		Size:     written,
	}
	saveIndex(idx)

	fmt.Printf("Downloaded %s (%s)\n", modelName, formatSize(written))
	return nil
}

// ── llama-server Download ──────────────────────────────────────────────

func llamaServerDownloadURL() (string, string, error) {
	// Fetch latest release
	resp, err := http.Get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
	if err != nil {
		return "", "", fmt.Errorf("fetching latest release: %w", err)
	}
	defer resp.Body.Close()

	var release struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name               string `json:"name"`
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return "", "", fmt.Errorf("parsing release: %w", err)
	}

	// Determine the right binary for this platform
	osName := runtime.GOOS
	arch := runtime.GOARCH
	if arch == "x86_64" || arch == "amd64" {
		arch = "x64"
	}

	var prefix string
	switch osName {
	case "linux":
		prefix = "llama-" + release.TagName + "-bin-ubuntu-" + arch
	case "darwin":
		prefix = "llama-" + release.TagName + "-bin-macos-" + arch
	case "windows":
		prefix = "llama-" + release.TagName + "-bin-win-" + arch
	default:
		return "", "", fmt.Errorf("unsupported platform: %s/%s", osName, arch)
	}

	// Try to find a matching asset
	for _, asset := range release.Assets {
		if strings.HasPrefix(asset.Name, prefix) {
			return asset.BrowserDownloadURL, release.TagName, nil
		}
	}

	// For Linux CUDA, there's no pre-built CUDA binary on the releases page.
	// Return the CPU build as fallback.
	for _, asset := range release.Assets {
		if strings.HasPrefix(asset.Name, prefix) && !strings.Contains(asset.Name, "openvino") && !strings.Contains(asset.Name, "rocm") && !strings.Contains(asset.Name, "vulkan") {
			return asset.BrowserDownloadURL, release.TagName, nil
		}
	}

	return "", "", fmt.Errorf("no pre-built llama-server found for %s/%s", osName, arch)
}

func ensureLlamaServer() error {
	self := filepath.Join(binDir(), "llama-server")
	if _, err := os.Stat(self); err == nil {
		// Check version
		cmd := exec.Command(self, "--version")
		out, _ := cmd.Output()
		_ = out
		return nil // already installed
	}

	fmt.Println("llama-server not found. Downloading...")
	url, tag, err := llamaServerDownloadURL()
	if err != nil {
		return fmt.Errorf("%w\n\nTry building from source:\n  git clone https://github.com/ggml-org/llama.cpp\n  cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server\n  cp build/bin/llama-server ~/.go-llama/bin/", err)
	}

	ensureDir(binDir())

	// Download archive
	tmpFile := filepath.Join(binDir(), "llama-server.tar.gz")
	defer os.Remove(tmpFile)

	fmt.Printf("Downloading %s ...\n", url)
	dlResp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("downloading: %w", err)
	}
	defer dlResp.Body.Close()

	out, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, dlResp.Body)
	out.Close()
	if err != nil {
		return fmt.Errorf("downloading: %w", err)
	}

	// Extract
	gzr, err := gzip.NewReader(tmpFileReader(tmpFile))
	if err != nil {
		return fmt.Errorf("extracting: %w", err)
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)
	found := false
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("extracting: %w", err)
		}
		// Look for llama-server or llama-server.exe in the archive
		name := filepath.Base(header.Name)
		if name == "llama-server" || name == "llama-server.exe" {
			outFile := filepath.Join(binDir(), name)
			outF, err := os.Create(outFile)
			if err != nil {
				return err
			}
			if _, err := io.Copy(outF, tr); err != nil {
				outF.Close()
				return err
			}
			outF.Close()
			os.Chmod(outFile, 0755)
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("llama-server not found in downloaded archive")
	}

	// Save version
	os.WriteFile(versionFile(), []byte(tag), 0644)
	fmt.Printf("llama-server %s installed to %s\n", tag, self)
	fmt.Println("Note: CPU-only build. For CUDA, build from source and copy to ~/.go-llama/bin/llama-server")
	return nil
}

func tmpFileReader(path string) io.ReadCloser {
	r, _ := os.Open(path)
	return r
}

func formatSize(bytes int64) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(1<<20))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// ── CLI ────────────────────────────────────────────────────────────────

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	if os.Args[1] == "help" || os.Args[1] == "--help" || os.Args[1] == "-h" {
		printUsage()
		return
	}

	mgr := NewManager()

	switch os.Args[1] {
	case "update":
		if err := ensureLlamaServer(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

	case "pull":
		if len(os.Args) < 3 {
			fmt.Println("Usage: go-llama pull hf.co/user/repo:quant")
			fmt.Println("  e.g. go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M")
			os.Exit(1)
		}
		modelRef := os.Args[2]
		if err := pullModel(modelRef); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

	case "list":
		models, err := listModels()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if len(models) == 0 {
			fmt.Println("No models found. Use 'go-llama pull <model>' to download one.")
			return
		}
		fmt.Printf("%-40s %-10s %s\n", "Name", "Size", "Source")
		for _, m := range models {
			fmt.Printf("%-40s %-10s %s\n", m.Name, formatSize(m.Size), m.Source)
		}

	case "serve":
		ensureLlamaServer() // best-effort
		port := "9080"
		if len(os.Args) > 2 {
			port = os.Args[2]
		}
		fmt.Printf("go-llama starting on :%s\n", port)
		startServer(mgr, port)

	case "run":
		if len(os.Args) < 3 {
			fmt.Println("Usage: go-llama run <model> [flags...]")
			os.Exit(1)
		}
		// Auto-ensure llama-server is available
		if err := ensureLlamaServer(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: %v\n", err)
		}
		model := os.Args[2]
		extraArgs := os.Args[3:]

		// Check if server is running, if not start one
		if len(mgr.List()) == 0 {
			inst, err := mgr.Start(model, 8081, extraArgs)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Started %s on port %d (PID %d)\n", inst.Model, inst.Port, inst.PID)
			fmt.Printf("Chat: http://localhost:%d\n", inst.Port)
			fmt.Println("Press Ctrl+C to stop")
			// Block until signal
			select {}
		}

	case "ps":
		instances := mgr.List()
		if len(instances) == 0 {
			fmt.Println("No running instances")
			return
		}
		fmt.Printf("%-5s %-20s %-5s %-8s\n", "Port", "Model", "PID", "Status")
		for _, inst := range instances {
			fmt.Printf("%-5d %-20s %-5d %-8s\n", inst.Port, inst.Model, inst.PID, inst.Status)
		}

	case "stop":
		if len(os.Args) < 3 {
			fmt.Println("Usage: go-llama stop <port>")
			os.Exit(1)
		}
		port, _ := strconv.Atoi(os.Args[2])
		if err := mgr.Stop(port); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Stopped instance on port %d\n", port)

	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`go-llama — llama.cpp instance manager

Usage:
  go-llama update                 Download/update llama-server binary
  go-llama pull <model>           Download model from HuggingFace
  go-llama list                   List available models
  go-llama serve [port]           Start manager with web UI (default :9080)
  go-llama run <model> [flags]    Quick-start a model on port 8081
  go-llama ps                     List running instances
  go-llama stop <port>            Stop an instance

Examples:
  go-llama update
  go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M
  go-llama run Qwopus3.6-27B-v2-Q4_K_M.gguf --tensor-split 12,8 --flash-attn on
  go-llama serve

Tip:
  Models are stored in ~/.go-llama/models/
  llama-server binary in ~/.go-llama/bin/ (auto-downloaded via 'go-llama update')
  For CUDA on Linux: build from source and copy the binary to ~/.go-llama/bin/`)
}

// ── Web UI ─────────────────────────────────────────────────────────────

const uiPage = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>go-llama</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; max-width: 1200px; margin: 0 auto; }
h1 { color: #a78bfa; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 14px; margin-bottom: 20px; }
h2 { color: #c4b5fd; margin: 16px 0 8px; font-size: 16px; }
.card { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155; }
.card-row { display: flex; gap: 16px; }
.card-row .card { flex: 1; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.instance { border-left: 3px solid #22c55e; }
.instance.stopped { border-left-color: #ef4444; }
label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
select, input, button { width: 100%; padding: 8px; background: #0f172a; border: 1px solid #334155; border-radius: 4px; color: #e2e8f0; font-size: 14px; margin-bottom: 8px; }
button { background: #7c3aed; border: none; cursor: pointer; font-weight: 600; transition: background .2s; }
button:hover { background: #6d28d9; }
button.secondary { background: #334155; }
button.secondary:hover { background: #475569; }
button.danger { background: #dc2626; }
button.danger:hover { background: #b91c1c; }
button.success { background: #16a34a; }
button.success:hover { background: #15803d; }
button.small { width: auto; padding: 4px 12px; font-size: 12px; }
.flag-row { display: flex; gap: 8px; margin-bottom: 4px; }
.flag-row input { flex: 1; margin-bottom: 0; }
.flag-row button { width: auto; padding: 8px 16px; background: #ef4444; }
.mt-8 { margin-top: 8px; }
.text-sm { font-size: 12px; color: #64748b; }
.text-xs { font-size: 11px; color: #475569; }
.flex { display: flex; justify-content: space-between; align-items: center; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-left: 6px; }
.tag-local { background: #1e3a5f; color: #60a5fa; }
.tag-ollama { background: #3b1f3b; color: #c084fc; }
.progress { width: 100%; height: 4px; background: #0f172a; border-radius: 2px; margin-top: 8px; overflow: hidden; }
.progress-bar { height: 100%; background: #7c3aed; border-radius: 2px; transition: width .5s; }
.pull-status { font-size: 12px; color: #94a3b8; margin-top: 6px; }
</style>
</head>
<body>
<h1>go-llama</h1>
<div class="subtitle">llama.cpp instance manager</div>

<div class="card-row">
  <div class="card">
    <h2>Pull Model</h2>
    <input type="text" id="pullInput" placeholder="hf.co/user/repo:Q4_K_M" value="hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M">
    <button onclick="pullModel()" id="pullBtn">Pull</button>
    <div id="pullStatus" class="pull-status"></div>
  </div>

  <div class="card">
    <h2>New Instance</h2>
    <label>Model</label>
    <select id="modelSelect"><option value="">Loading...</option></select>
    <label>Port</label>
    <input type="number" id="portInput" value="8081" min="8081" max="8099">
    <label>Extra Flags</label>
    <div id="flagsContainer">
      <div class="flag-row">
        <input type="text" placeholder="e.g. --tensor-split 12,8" class="flag-input">
        <button class="small danger" onclick="this.parentElement.remove()">x</button>
      </div>
    </div>
    <button class="secondary small" onclick="addFlag()">+ Add Flag</button>
    <button class="mt-8" onclick="launchInstance()">Launch</button>
  </div>
</div>

<div class="card">
  <h2>Running Instances</h2>
  <div id="instances" class="grid"><div class="text-sm">Loading...</div></div>
</div>

<script>
function addFlag(){
  var c=document.getElementById('flagsContainer'),r=document.createElement('div');
  r.className='flag-row';
  r.innerHTML='<input type="text" placeholder="e.g. --tensor-split 12,8" class="flag-input"><button class="small danger" onclick="this.parentElement.remove()">x</button>';
  c.appendChild(r);
}

async function loadModels(){
  var r=await fetch('/api/v1/models'),m=await r.json(),s=document.getElementById('modelSelect'),seen={};
  s.innerHTML='<option value="">Select model...</option>';
  m.forEach(function(x){
    if(!seen[x.Name]){seen[x.Name]=1;
      var tag=x.Source=='local'?'<span class="tag tag-local">local</span>':'<span class="tag tag-ollama">ollama</span>';
      s.innerHTML+='<option value="'+x.Name+'">'+x.Name+' '+tag+'</option>';
    }
  });
}

async function loadInstances(){
  var r=await fetch('/api/v1/instances'),list=await r.json(),c=document.getElementById('instances');
  if(!list.length){c.innerHTML='<div class="text-sm">No running instances</div>';return;}
  c.innerHTML=list.map(function(i){
    return '<div class="card instance"><div class="flex"><strong>'+i.Model+'</strong><span>:'+i.Port+'</span></div>'+
      '<div class="text-sm mt-8">PID: '+i.PID+' | Status: <strong>'+i.Status+'</strong></div>'+
      '<button class="danger small mt-8" onclick="stopInstance('+i.Port+')">Stop</button></div>';
  }).join('');
}

async function pullModel(){
  var ref=document.getElementById('pullInput').value.trim();
  if(!ref){alert('Enter a model reference like hf.co/user/repo:Q4_K_M');return;}
  var btn=document.getElementById('pullBtn'),status=document.getElementById('pullStatus');
  btn.disabled=true;btn.textContent='Pulling...';status.textContent='Starting download...';
  try{
    var r=await fetch('/api/v1/models/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:ref})});
    var d=await r.json();
    if(d.error){status.textContent='Error: '+d.error;alert(d.error);}
    else{status.textContent='Downloaded '+ref;loadModels();}
  }catch(e){status.textContent='Error: '+e;alert(e);}
  btn.disabled=false;btn.textContent='Pull';
}

async function launchInstance(){
  var m=document.getElementById('modelSelect').value,p=parseInt(document.getElementById('portInput').value),f=[];
  document.querySelectorAll('.flag-input').forEach(function(el){var v=el.value.trim();if(v)f.push(v);});
  if(!m){alert('Select a model');return;}
  var r=await fetch('/api/v1/instances',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m,port:p,flags:f})});
  if(!r.ok){var e=await r.text();alert('Error: '+e);return;}
  var i=await r.json();
  document.getElementById('portInput').value=i.Port+1;
  loadInstances();
}

async function stopInstance(p){await fetch('/api/v1/instances/stop?port='+p,{method:'POST'});loadInstances();}

loadModels();loadInstances();setInterval(loadInstances,3000);
</script>
</body>
</html>`
