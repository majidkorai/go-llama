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
	"syscall"
	"time"
)

// ── Types ──────────────────────────────────────────────────────────────

type progressReader struct {
	reader io.Reader
	total  int64
	done   int64
	start  time.Time
	name   string
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	pr.done += int64(n)
	elapsed := time.Since(pr.start).Seconds()
	var speed string
	if elapsed > 0 {
		rate := float64(pr.done) / (1024 * 1024) / elapsed
		speed = fmt.Sprintf("%.1f MB/s", rate)
	}
	if pr.total > 0 {
		pct := float64(pr.done) * 100 / float64(pr.total)
		fmt.Printf("\r  %s  %.1f%%  (%s / %s)  %s    ",
			pr.name, pct, formatSize(pr.done), formatSize(pr.total), speed)
	} else {
		fmt.Printf("\r  %s  %s  %s       ",
			pr.name, formatSize(pr.done), speed)
	}
	if err == io.EOF {
		fmt.Println()
	}
	return n, err
}

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
	return filepath.Join(home, ".gollama")
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
func backendFile() string {
	return filepath.Join(goLLamaDir(), "llama-server-backend.txt")
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
	m := &Manager{
		instances: make(map[int]*Instance),
		nextPort:  8081,
	}
	m.recoverOrphans()
	return m
}

// recoverOrphans scans for running llama-server processes and adopts them.
func (m *Manager) recoverOrphans() {
	cmd := exec.Command("pgrep", "-a", "llama-server")
	out, err := cmd.Output()
	if err != nil {
		return // no orphans
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		pidStr := parts[0]
		pid, _ := strconv.Atoi(pidStr)
		if pid == 0 {
			continue
		}

		// Extract port and model from command line
		var port int
		var model string
		args := parts[1:]
		for i, a := range args {
			if a == "--port" && i+1 < len(args) {
				port, _ = strconv.Atoi(args[i+1])
			}
			if a == "-m" && i+1 < len(args) {
				model = args[i+1]
				// Shorten model path to just filename
				if idx := strings.LastIndex(model, "/"); idx >= 0 {
					model = model[idx+1:]
				}
			}
		}
		if port == 0 {
			port = m.nextPort
			m.nextPort++
		}
		if port >= m.nextPort {
			m.nextPort = port + 1
		}
		if _, exists := m.instances[port]; !exists && port > 0 {
			m.instances[port] = &Instance{
				Port:   port,
				Model:  model,
				PID:    pid,
				Status: "running",
			}
			log.Printf("recovered orphan instance: port=%d pid=%d model=%s", port, pid, model)
		}
	}
}

func findLlamaServer() string {
	// 1. Check gollama's own bin directory first
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
	// Check gollama local models
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
	}
	isCPU := func() bool {
		d, _ := os.ReadFile(backendFile())
		return strings.TrimSpace(string(d)) == "CPU"
	}()
	if !isCPU {
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
	}
	args = append(args, extraArgs...)

	log.Printf("launching llama-server: %s %s", llamaBin, strings.Join(args, " "))
	logDir := filepath.Join(goLLamaDir(), "logs")
	ensureDir(logDir)
	logFile := filepath.Join(logDir, fmt.Sprintf("port-%d.log", port))
	logF, logErr := os.Create(logFile)
	if logErr != nil {
		log.Printf("warning: could not create log file %s: %v", logFile, logErr)
	}

	cmd := exec.Command(llamaBin, args...)
	cmd.Stdout = logF
	cmd.Stderr = logF

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

	log.Printf("instance started: model=%s port=%d pid=%d", model, port, cmd.Process.Pid)

	go func() {
		err := cmd.Wait()
		m.mu.Lock()
		if inst.Status == "running" {
			inst.Status = "stopped"
			if err != nil {
				log.Printf("instance stopped with error: port=%d err=%v", port, err)
			} else {
				log.Printf("instance stopped: port=%d", port)
			}
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

func (m *Manager) UpdateTokens(port int, tps float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if inst, ok := m.instances[port]; ok {
		inst.TokensPerSec = tps
	}
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

	// Scan gollama's local model storage
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

	mux.HandleFunc("/api/v1/models/delete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", 405)
			return
		}
		var req struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			jsonError(w, err.Error(), 400)
			return
		}
		if req.Name == "" {
			jsonError(w, "model name is required", 400)
			return
		}

		idx := loadIndex()
		info, ok := idx[req.Name]
		if !ok {
			jsonError(w, "model not found", 404)
			return
		}

		// Delete the file
		if err := os.Remove(info.BlobPath); err != nil && !os.IsNotExist(err) {
			jsonError(w, fmt.Sprintf("error deleting file: %v", err), 500)
			return
		}

		// Remove from index
		delete(idx, req.Name)
		saveIndex(idx)

		log.Printf("model deleted: %s", req.Name)
		jsonResponse(w, map[string]string{"status": "deleted", "model": req.Name})
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

	mux.HandleFunc("/api/v1/instances/logs", func(w http.ResponseWriter, r *http.Request) {
		portStr := r.URL.Query().Get("port")
		logDir := filepath.Join(goLLamaDir(), "logs")
		logFile := filepath.Join(logDir, fmt.Sprintf("port-%s.log", portStr))
		data, err := os.ReadFile(logFile)
		if err != nil {
			jsonError(w, "log not found", 404)
			return
		}
		lines := strings.Split(string(data), "\n")
		if len(lines) > 100 {
			lines = lines[len(lines)-100:]
		}
		jsonResponse(w, map[string]interface{}{
			"port":  portStr,
			"lines": lines,
		})
	})

	mux.HandleFunc("/api/v1/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", 405)
			return
		}
		portStr := r.URL.Query().Get("port")
		port, err := strconv.Atoi(portStr)
		if err != nil {
			jsonError(w, "port query parameter is required", 400)
			return
		}

		// Read the incoming request body
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			jsonError(w, "invalid request body", 400)
			return
		}

		// Proxy to llama-server
		target := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
		resp, err := http.Post(target, "application/json", strings.NewReader(string(body)))
		if err != nil {
			jsonError(w, fmt.Sprintf("proxy error: %v", err), 502)
			return
		}
		defer resp.Body.Close()

		// Read response to extract timing
		respBody, _ := io.ReadAll(resp.Body)
		var timingData struct {
			Timings *struct {
				PredictedPerSecond float64 `json:"predicted_per_second"`
			} `json:"timings"`
		}
		if json.Unmarshal(respBody, &timingData) == nil && timingData.Timings != nil {
			mgr.UpdateTokens(port, timingData.Timings.PredictedPerSecond)
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write(respBody)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		fmt.Fprint(w, uiPage)
	})

	addr := fmt.Sprintf(":%s", port)
	log.Printf("gollama listening on %s", addr)
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
	fmt.Printf("Downloading %s (%s)\n", targetFile, formatSize(targetSize))

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

	pr := &progressReader{
		reader: dlResp.Body,
		total:  targetSize,
		name:   "▸",
		start:  time.Now(),
	}
	written, err := io.Copy(out, pr)
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

	log.Printf("model downloaded: %s (%s) → %s", modelName, formatSize(written), dest)
	fmt.Printf("Downloaded %s (%s)\n", modelName, formatSize(written))
	return nil
}

// ── llama-server Download ──────────────────────────────────────────────

type backendOption struct {
	Name   string
	Suffix string // URL suffix for the asset (e.g. "", "-vulkan", "-rocm-7.2")
	GPU    bool
}

func detectGPUBackends() []backendOption {
	var options []backendOption
	options = append(options, backendOption{Name: "CPU", Suffix: "", GPU: false})

	// Check for CUDA via nvidia-smi
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		cmd := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
		if out, err := cmd.Output(); err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			label := "CUDA"
			if len(lines) == 1 {
				label = fmt.Sprintf("CUDA (%s)", strings.TrimSpace(lines[0]))
			} else if len(lines) > 1 {
				label = fmt.Sprintf("CUDA (%d GPUs)", len(lines))
			}
			options = append(options, backendOption{
				Name:   label,
				Suffix: "-cuda",
				GPU:    true,
			})
		}
	}

	// Check for ROCm
	if _, err := os.Stat("/opt/rocm"); err == nil {
		options = append(options, backendOption{
			Name:   "ROCm",
			Suffix: "-rocm-7.2",
			GPU:    true,
		})
	}

	// Vulkan is always available as a fallback
	options = append(options, backendOption{Name: "Vulkan", Suffix: "-vulkan", GPU: true})

	return options
}

func getReleaseData() (string, map[string]string, error) {
	resp, err := http.Get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
	if err != nil {
		return "", nil, fmt.Errorf("fetching latest release: %w", err)
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
		return "", nil, fmt.Errorf("parsing release: %w", err)
	}

	assets := make(map[string]string)
	for _, a := range release.Assets {
		assets[a.Name] = a.BrowserDownloadURL
	}
	return release.TagName, assets, nil
}

func findAsset(tagName, kind string, assets map[string]string) (string, error) {
	osName := runtime.GOOS
	arch := runtime.GOARCH
	if arch == "x86_64" || arch == "amd64" {
		arch = "x64"
	}

	// Build expected filename prefixes
	var candidates []string
	base := fmt.Sprintf("llama-%s-bin", tagName)

	switch osName {
	case "linux":
		// Assets: llama-b9374-bin-ubuntu-x64.tar.gz (CPU)
		//         llama-b9374-bin-ubuntu-vulkan-x64.tar.gz (Vulkan)
		//         llama-b9374-bin-ubuntu-rocm-7.2-x64.tar.gz (ROCm)
		candidates = []string{
			base + "-ubuntu" + kind + "-" + arch,
			base + "-ubuntu-" + arch + kind,
			base + "-ubuntu-" + arch,
		}
	case "darwin":
		candidates = []string{base + "-macos-" + arch}
	case "windows":
		// Assets: llama-b9374-bin-win-x64.zip (CPU)
		//         llama-b9374-bin-win-cuda-13.3-x64.zip (CUDA)
		//         llama-b9374-bin-win-vulkan-x64.zip (Vulkan)
		candidates = []string{
			base + "-win" + kind + "-" + arch,
			base + "-win-" + arch + kind,
			base + "-win-" + arch,
		}
	}

	for _, c := range candidates {
		for name, url := range assets {
			if name == c+".tar.gz" || name == c+".zip" || strings.HasPrefix(name, c+".") {
				return url, nil
			}
		}
	}
	return "", fmt.Errorf("no matching asset for %s/%s (kind=%s)", osName, arch, kind)
}

func ensureLlamaServer() error {
	self := filepath.Join(binDir(), "llama-server")
	if _, err := os.Stat(self); err == nil {
		fmt.Printf("llama-server already installed at %s\n", self)
		cmd := exec.Command(self, "--version")
		out, _ := cmd.Output()
		if len(out) > 0 {
			fmt.Printf("Version: %s", out)
		}
		return nil
	}

	fmt.Println("llama-server not found.")
	tagName, assets, err := getReleaseData()
	if err != nil {
		return fmt.Errorf("fetching release info: %w", err)
	}

	backends := detectGPUBackends()

	fmt.Println("\nAvailable builds for llama.cpp " + tagName + ":")
	for i, b := range backends {
		mark := ""
		if b.GPU {
			mark = " 🚀 (recommended)"
		}
		if i == 0 {
			mark = " (fallback)"
		}
		fmt.Printf("  [%d] %s%s\n", i, b.Name, mark)
	}
	fmt.Printf("\nChoose (0-%d): ", len(backends)-1)

	var choice int
	fmt.Scanf("%d", &choice)
	if choice < 0 || choice >= len(backends) {
		choice = 0
	}

	selected := backends[choice]
	fmt.Printf("Selected: %s\n", selected.Name)

	// Handle CUDA on Linux (no pre-built binary in releases)
	if selected.Suffix == "-cuda" {
		if runtime.GOOS == "linux" {
			fmt.Println("\nCUDA pre-built binaries are not available for Linux on the release page.")
			fmt.Println("However, we can download the Vulkan build which supports NVIDIA GPUs.")
			fmt.Println("Options:")
			fmt.Println("  [1] Download Vulkan build (recommended for NVIDIA)")
			fmt.Println("  [2] Show CUDA build instructions")
			fmt.Println("  [3] Cancel")
			fmt.Print("Choose: ")
			var c int
			fmt.Scanf("%d", &c)
			switch c {
			case 1:
				selected = backends[len(backends)-1] // Vulkan
			case 2:
				fmt.Println(`
To build llama-server with CUDA:

  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real"
  cmake --build build -j --target llama-server
  cp build/bin/llama-server ~/.gollama/bin/

Then run 'gollama update' again.`)
				return nil
			default:
				return fmt.Errorf("installation cancelled")
			}
		}
		// Windows/macOS CUDA builds are available via pre-built binaries
	}

	url, err := findAsset(tagName, selected.Suffix, assets)
	if err != nil {
		return fmt.Errorf("build not found: %w", err)
	}

	ensureDir(binDir())
	tmpFile := filepath.Join(binDir(), "llama-server.tar.gz")
	defer os.Remove(tmpFile)

	fmt.Printf("Downloading ...\n")
	dlResp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("downloading: %w", err)
	}
	defer dlResp.Body.Close()

	pr := &progressReader{
		reader: dlResp.Body,
		total:  dlResp.ContentLength,
		name:   "▸",
		start:  time.Now(),
	}

	out, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, pr)
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

	extractDir := binDir()
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
		name := filepath.Base(header.Name)
		if name == "" || name == "." || name == ".." {
			continue
		}
		outFile := filepath.Join(extractDir, name)
		switch header.Typeflag {
		case tar.TypeDir:
			os.MkdirAll(outFile, 0755)
		case tar.TypeSymlink:
			os.Symlink(header.Linkname, outFile)
		default:
			outF, err := os.Create(outFile)
			if err != nil {
				return fmt.Errorf("creating %s: %w", name, err)
			}
			if _, err := io.Copy(outF, tr); err != nil {
				outF.Close()
				return err
			}
			outF.Close()
			os.Chmod(outFile, os.FileMode(header.Mode))
		}
		if name == "llama-server" || name == "llama-server.exe" {
			found = true
		}
	}

	if !found {
		return fmt.Errorf("llama-server not found in downloaded archive")
	}
	os.WriteFile(versionFile(), []byte(tagName), 0644)
	os.WriteFile(backendFile(), []byte(selected.Name), 0644)
	log.Printf("llama-server installed: version=%s backend=%s path=%s", tagName, selected.Name, self)
	fmt.Printf("\nllama-server %s (%s) installed to %s\n", tagName, selected.Name, self)
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

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 || os.Args[1] == "--version" || os.Args[1] == "-v" {
		if len(os.Args) > 1 && (os.Args[1] == "--version" || os.Args[1] == "-v") {
			fmt.Printf("gollama %s\n", version)
			return
		}
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
			fmt.Println("Usage: gollama pull hf.co/user/repo:quant")
			fmt.Println("  e.g. gollama pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
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
			fmt.Println("No models found. Use 'gollama pull <model>' to download one.")
			return
		}
		fmt.Printf("%-40s %-10s %s\n", "Name", "Size", "Source")
		for _, m := range models {
			fmt.Printf("%-40s %-10s %s\n", m.Name, formatSize(m.Size), m.Source)
		}

	case "serve":
		ensureLlamaServer() // best-effort
		// Periodic cleanup of orphaned/dead instances
		go func() {
			for {
				time.Sleep(30 * time.Second)
				// Check if tracked instances still have live processes
				for _, inst := range mgr.List() {
					proc, err := os.FindProcess(inst.PID)
					if err != nil {
						mgr.Stop(inst.Port)
						continue
					}
					// Signal 0 tests if process exists
					if err := proc.Signal(syscall.Signal(0)); err != nil {
						mgr.Stop(inst.Port)
					}
				}
				// Try to adopt any orphaned llama-server processes
				mgr.recoverOrphans()
			}
		}()
		port := "9080"
		if len(os.Args) > 2 {
			port = os.Args[2]
		}
		fmt.Printf("gollama starting on :%s\n", port)
		startServer(mgr, port)

	case "run":
		if len(os.Args) < 3 {
			fmt.Println("Usage: gollama run <model> [flags...]")
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
			fmt.Println("Usage: gollama stop <port>")
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
	fmt.Println(`gollama — llama.cpp instance manager

Usage:
  gollama update                 Download/update llama-server binary
  gollama pull <model>           Download model from HuggingFace
  gollama list                   List available models
  gollama serve [port]           Start manager with web UI (default :9080)
  gollama run <model> [flags]    Quick-start a model on port 8081
  gollama ps                     List running instances
  gollama stop <port>            Stop an instance

Examples:
  gollama update
  gollama pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
  gollama run Qwopus3.6-27B-v2-Q4_K_M.gguf --tensor-split 12,8 --flash-attn on
  gollama serve

Tip:
  Models are stored in ~/.gollama/models/
  llama-server binary in ~/.gollama/bin/ (auto-downloaded via 'gollama update')
  For CUDA on Linux: build from source and copy the binary to ~/.gollama/bin/`)
}

// ── Web UI ─────────────────────────────────────────────────────────────

const uiPage = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>gollama</title>
<style>
:root { --bg:#0a0a0a; --surface:#1a1a1a; --border:#2a2a2a; --text:#e5e5e5; --muted:#888; --accent:#7c3aed; --accent-hover:#6d28d9; --green:#22c55e; --red:#ef4444; --header-text:#a78bfa; --card-title:#c4b5fd; --input-bg:#0a0a0a; --select-bg:#1a1a1a; --hover-bg:#222; --badge-green-bg:#064e3b; --badge-green-text:#34d399; --badge-red-bg:#450a0a; --badge-red-text:#f87171; --badge-blue-bg:#1a1a3a; --badge-blue-text:#818cf8; --chat-user-bg:#1e293b; }
.light { --bg:#f5f5f5; --surface:#fff; --border:#ddd; --text:#1a1a1a; --muted:#777; --accent:#7c3aed; --accent-hover:#6d28d9; --green:#16a34a; --red:#dc2626; --header-text:#6d28d9; --card-title:#5b21b6; --input-bg:#f5f5f5; --select-bg:#fff; --hover-bg:#eee; --badge-green-bg:#dcfce7; --badge-green-text:#166534; --badge-red-bg:#fee2e2; --badge-red-text:#991b1b; --badge-blue-bg:#e0e7ff; --badge-blue-text:#4338ca; --chat-user-bg:#e0e7ff; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); padding:20px; max-width:1400px; margin:0 auto; transition:background .2s,color .2s; }
h1 { color:var(--header-text); margin-bottom:4px; display:inline-block; font-size:22px; }
.subtitle { color:var(--muted); font-size:13px; margin-bottom:20px; }
h2 { color:var(--card-title); margin:0 0 12px 0; font-size:14px; display:flex; align-items:center; gap:8px; }
.card { background:var(--surface); border-radius:10px; padding:16px; margin-bottom:16px; border:1px solid var(--border); }
.card-row { display:flex; gap:16px; flex-wrap:wrap; }
.card-row .card { flex:1; min-width:300px; }
.label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
select, input, button { width:100%; padding:8px 12px; background:var(--input-bg); border:1px solid var(--border); border-radius:8px; color:var(--text); font-size:13px; margin-bottom:8px; outline:none; transition:border-color .2s; }
select:focus, input:focus { border-color:var(--accent); }
select option { background:var(--select-bg); }
button { background:linear-gradient(135deg,var(--accent),var(--accent-hover)); border:none; cursor:pointer; font-weight:600; font-size:13px; transition:all .2s; padding:8px 16px; border-radius:8px; color:#fff; }
button:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(124,58,237,.3); }
button.secondary { background:var(--border); color:var(--text); box-shadow:none; }
button.secondary:hover { transform:none; background:var(--hover-bg); }
button.danger { background:linear-gradient(135deg,var(--red),#b91c1c); }
button.danger:hover { box-shadow:0 4px 14px rgba(220,38,38,.3); }
button.small { width:auto; padding:4px 10px; font-size:11px; border-radius:6px; }
#themeToggle { width:36px; height:36px; padding:0; font-size:16px; border-radius:50%; display:flex; align-items:center; justify-content:center; float:right; cursor:pointer; background:var(--border); color:var(--text); border:1px solid var(--border); }
#themeToggle:hover { background:var(--hover-bg); transform:none; box-shadow:none; }
.flag-row { display:flex; gap:8px; margin-bottom:4px; }
.flag-row input { flex:1; margin-bottom:0; }
.flag-row button { width:auto; }
.mt-8 { margin-top:8px; }
.text-sm { font-size:12px; color:var(--muted); }
.text-xs { font-size:11px; color:var(--muted); }
.flex { display:flex; justify-content:space-between; align-items:center; }
.badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.3px; }
.badge-green { background:var(--badge-green-bg); color:var(--badge-green-text); }
.badge-red { background:var(--badge-red-bg); color:var(--badge-red-text); }
.badge-blue { background:var(--badge-blue-bg); color:var(--badge-blue-text); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }
.inst-card { border-left:4px solid var(--green); padding:14px; background:var(--surface); border-radius:0 10px 10px 0; border:1px solid var(--border); border-left-width:4px; transition:all .2s; }
.inst-card:hover { border-color:var(--hover-bg); }
.inst-card.stopped { border-left-color:var(--red); opacity:.6; }
.inst-card .title { font-weight:600; font-size:13px; word-break:break-all; }
.inst-card .meta { font-size:11px; color:var(--muted); margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.inst-card .actions { margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; }
.chat-msgs { flex:1; overflow-y:auto; padding:12px; background:var(--bg); border-radius:8px; margin-bottom:8px; font-size:13px; line-height:1.6; }
.chat-msgs .msg { margin-bottom:10px; padding:8px 12px; border-radius:10px; max-width:85%; line-height:1.5; }
.chat-msgs .user { background:var(--chat-user-bg); margin-left:auto; border-bottom-right-radius:4px; }
.chat-msgs .assistant { background:var(--surface); border:1px solid var(--border); border-bottom-left-radius:4px; }
.chat-msgs .system { background:transparent; color:var(--muted); font-style:italic; font-size:11px; text-align:center; max-width:100%; }
.chat-input-row { display:flex; gap:8px; }
.chat-input-row input { flex:1; margin-bottom:0; }
.chat-input-row button { width:auto; padding:8px 20px; }
.empty-state { text-align:center; padding:40px 20px; color:var(--muted); }
.empty-state .icon { font-size:36px; margin-bottom:10px; }
.chat-panel { display:none; }
.chat-panel.active { display:flex; flex-direction:column; height:400px; }
.instance-selector { display:flex; gap:8px; align-items:center; margin-bottom:12px; }
.instance-selector select { margin-bottom:0; }
.model-row { display:flex; justify-content:space-between; align-items:center; padding:8px 10px; border-radius:6px; transition:background .2s; }
.model-row:hover { background:var(--hover-bg); }
.model-row .name { font-size:13px; color:var(--text); }
.model-row .info { font-size:11px; color:var(--muted); margin-top:2px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
.loading { animation:pulse 1.5s infinite; }
.chat-loading { animation:pulse 1.2s infinite; display:inline-block; letter-spacing:4px; font-size:18px; line-height:1; color:var(--muted); }
@media(max-width:768px){.card-row{flex-direction:column}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="flex"><h1>gollama</h1> <button id="themeToggle" onclick="toggleTheme()" title="Toggle theme">🌙</button></div>
<div class="subtitle">llama.cpp instance manager</div>

<div class="card-row">
  <div class="card">
    <h2>📥 Pull Model</h2>
    <input type="text" id="pullInput" placeholder="hf.co/user/repo:Q4_K_M" value="hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M">
    <button onclick="pullModel()" id="pullBtn">Pull</button>
    <div id="pullStatus" class="text-sm" style="margin-top:4px"></div>
  </div>

  <div class="card">
    <h2>🚀 New Instance</h2>
    <div class="label">Model</div>
    <select id="modelSelect"><option value="">Loading models...</option></select>
    <div class="label">Port</div>
    <input type="number" id="portInput" value="8081" min="8081" max="8099">
    <div class="label">Flags</div>
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

<div class="card-row">
  <div class="card" style="flex:1">
    <h2>📦 Models <span id="modelCount" class="text-sm" style="font-weight:400"></span></h2>
    <div id="modelList"><div class="text-sm">Loading...</div></div>
  </div>

  <div class="card" style="flex:1">
    <h2>🟢 Running <span id="instanceCount" class="text-sm" style="font-weight:400"></span></h2>
    <div id="instances" class="grid"><div class="text-sm">Loading...</div></div>
  </div>

  <div class="card" style="flex:1">
    <h2>💬 Chat</h2>
    <div class="instance-selector">
      <select id="chatInstanceSelect" onchange="selectChatInstance()"><option value="">— select running instance —</option></select>
      <button class="small secondary" onclick="refreshChat()">↻</button>
    </div>
    <div id="chatPanel" class="chat-panel">
      <div id="chatMsgs" class="chat-msgs"></div>
      <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="Type a message..." onkeydown="if(event.key=='Enter')sendChat()">
        <button onclick="sendChat()">Send</button>
      </div>
    </div>
    <div id="chatEmpty" class="empty-state">
      <div class="icon">💬</div>
      <div>Launch an instance to start chatting</div>
    </div>
  </div>
</div>

<script>
// ── Model Selector ──
async function loadModels(){
  var r=await fetch('/api/v1/models'),m=await r.json(),s=document.getElementById('modelSelect'),seen={};
  s.innerHTML='<option value="">— Select model —</option>';
  if(!m||!m.length){s.innerHTML+='<option value="" disabled>No models found. Use gollama pull.</option>';return;}
  m.forEach(function(x){
    var n=x.name||'(unnamed)',src=x.source||'unknown';
    if(!seen[n]){seen[n]=1;s.innerHTML+='<option value="'+n+'">'+n+' ['+src+']</option>';}
  });
  loadModelList();
}

async function loadModelList(){
  var r=await fetch('/api/v1/models'),m=await r.json(),c=document.getElementById('modelList');
  document.getElementById('modelCount').textContent='('+m.length+')';
  if(!m.length){c.innerHTML='<div class="text-sm">No models downloaded</div>';return;}
  c.innerHTML=m.map(function(x){
    var name=x.name||'?',size=x.size?fmtSize(x.size):'?',q=x.name.match(/[Qq][0-9]_[A-Z0-9_]+|[Bb][Ff]16|[Ff][Pp]16/);
    return '<div class="model-row"><div><div class="name">'+(name.length>50?name.slice(0,50)+'...':name)+'</div><div class="info">'+size+(q?' | <span class="badge badge-blue">'+q[0].toUpperCase()+'</span>':'')+' ['+(x.source||'?')+']</div></div>'+
      '<button class="small danger" onclick="deleteModel(\''+name.replace(/\'/g,'')+'\')">🗑</button></div>';
  }).join('');
}

function fmtSize(b){
  if(!b)return'?';
  if(b>1073741824)return(b/1073741824).toFixed(1)+' GB';
  if(b>1048576)return(b/1048576).toFixed(0)+' MB';
  return(b/1024).toFixed(0)+' KB';
}

async function deleteModel(name){
  if(!confirm('Delete model "'+name+'"?'))return;
  var r=await fetch('/api/v1/models/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})});
  var d=await r.json();
  if(d.error){alert('Error: '+d.error);return;}
  loadModels();loadModelList();
}

// ── Instances ──
async function loadInstances(){
  var r=await fetch('/api/v1/instances'),list=await r.json(),c=document.getElementById('instances');
  document.getElementById('instanceCount').textContent='('+list.length+')';
  if(!list.length){c.innerHTML='<div class="text-sm">No running instances</div>';return;}
  c.innerHTML=list.map(function(i){
    var sc=i.status=='running'?'':' stopped';
    var bc=i.status=='running'?'badge-green':'badge-red';
    var mn=i.model||'?';
    var tps=i.tokens_per_sec?'<span style="color:#22c55e">⚡ '+i.tokens_per_sec.toFixed(1)+' t/s</span>':'';
    return '<div class="inst-card'+sc+'"><div class="title">'+(mn.length>40?mn.slice(0,40)+'...':mn)+'</div>'+
      '<div class="meta">Port: '+i.port+' | PID: '+i.pid+' | <span class="badge '+bc+'">'+i.status+'</span> '+tps+'</div>'+
      '<div class="actions"><button class="small danger" onclick="stopInstance('+i.port+')">⏹ Stop</button>'+
      '<button class="small secondary" onclick="selectChatFor('+i.port+',\''+mn.replace(/\'/g,'')+'\')">💬 Chat</button>'+
      '<button class="small secondary" onclick="window.open(\'http://\'+location.hostname+\':'+i.port+'\',\'_blank\')">🌐 UI</button>'+
      '<button class="small secondary" onclick="viewLogs('+i.port+')">📋 Logs</button></div></div>';
  }).join('');
}

async function launchInstance(){
  var m=document.getElementById('modelSelect').value,p=parseInt(document.getElementById('portInput').value),f=[];
  document.querySelectorAll('.flag-input').forEach(function(el){(el.value.trim().split(/\s+/)).forEach(function(v){if(v)f.push(v);});});
  if(!m){alert('Select a model');return;}
  var r=await fetch('/api/v1/instances',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m,port:p,flags:f})});
  if(!r.ok){var e=await r.text();alert('Error: '+e);return;}
  var i=await r.json();
  document.getElementById('portInput').value=(i.port||0)+1;
  loadInstances();refreshChatSelector();
}

async function stopInstance(p){
  if(!confirm('Stop instance on port '+p+'?'))return;
  await fetch('/api/v1/instances/stop?port='+p,{method:'POST'});
  loadInstances();refreshChatSelector();
  if(chatPort==p){document.getElementById('chatPanel').classList.remove('active');document.getElementById('chatEmpty').style.display='block';}
}

function addFlag(){
  var c=document.getElementById('flagsContainer'),r=document.createElement('div');
  r.className='flag-row';
  r.innerHTML='<input type="text" placeholder="e.g. --tensor-split 12,8" class="flag-input"><button class="small danger" onclick="this.parentElement.remove()">x</button>';
  c.appendChild(r);
}

// ── Chat ──
var chatPort=0,chatHistory=[];

async function refreshChatSelector(){
  var r=await fetch('/api/v1/instances'),list=await r.json(),s=document.getElementById('chatInstanceSelect');
  s.innerHTML='<option value="">— select running instance —</option>';
  list.forEach(function(i){var mn=i.model||'?';s.innerHTML+='<option value="'+i.port+'"'+(chatPort==i.port?' selected':'')+'>'+i.port+' - '+(mn.length>35?mn.slice(0,35)+'...':mn)+'</option>';});
  if(!list.length){document.getElementById('chatPanel').classList.remove('active');document.getElementById('chatEmpty').style.display='block';}
}

function selectChatInstance(){
  var s=document.getElementById('chatInstanceSelect');chatPort=parseInt(s.value)||0;
  if(chatPort){chatHistory=[];document.getElementById('chatMsgs').innerHTML='';document.getElementById('chatPanel').classList.add('active');document.getElementById('chatEmpty').style.display='none';addSystemMsg('Connected');}
}

function selectChatFor(port,model){
  chatPort=port;chatHistory=[];
  document.getElementById('chatInstanceSelect').value=port;
  document.getElementById('chatMsgs').innerHTML='';
  document.getElementById('chatPanel').classList.add('active');
  document.getElementById('chatEmpty').style.display='none';
  addSystemMsg('Chatting with '+(model||'port '+port));
}

function addSystemMsg(t){var c=document.getElementById('chatMsgs');c.innerHTML+='<div class="msg system">'+t+'</div>';c.scrollTop=c.scrollHeight;}
function addMsg(r,t,re){
  var c=document.getElementById('chatMsgs');
  var el=document.createElement('div');el.className='msg '+r;
  if(re){var r2=re.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');var rd=document.createElement('div');rd.style.cssText='color:var(--muted);font-style:italic;font-size:12px;border-left:2px solid var(--border);padding-left:8px;margin-bottom:4px';rd.innerHTML=r2;c.appendChild(rd);}
  if(t.indexOf('<')===-1&&t.indexOf('&')===-1){el.textContent=t;}else{el.innerHTML=t;}
  c.appendChild(el);c.scrollTop=c.scrollHeight;return el;
}

async function sendChat(){
  var input=document.getElementById('chatInput'),msg=input.value.trim();
  if(!msg||!chatPort)return;
  input.value='';addMsg('user',msg);chatHistory.push({role:'user',content:msg});
  var li=addMsg('assistant','<span class="chat-loading">● ● ●</span>');
  try{
    var r=await fetch('/api/v1/chat?port='+chatPort,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'default',messages:chatHistory.slice(-20),max_tokens:256,stream:false})});
    var d=await r.json(),msg=d.choices&&d.choices[0]&&d.choices[0].message?d.choices[0].message:{},reply=msg.content||'(no response)',reasoning=msg.reasoning_content||'';
    chatHistory.push({role:'assistant',content:reply});
    li.innerHTML=reply;li.className='msg assistant';
    if(reasoning){var r2=reasoning.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');li.insertAdjacentHTML('beforebegin','<div style="color:var(--muted);font-style:italic;font-size:12px;border-left:2px solid var(--border);padding-left:8px;margin-bottom:4px">'+r2+'</div>');}
  }catch(e){li.innerHTML='Error: '+e.message;li.className='msg system';}
}

// ── Logs ──
async function viewLogs(port){
  var r=await fetch('/api/v1/instances/logs?port='+port),d=await r.json();
  if(d.error){alert('No logs');return;}
  document.getElementById('logContent').textContent=d.lines&&d.lines.length?d.lines.slice(-50).join('\\n'):'(empty)';
  document.getElementById('logModal').style.display='block';
}
function closeLogs(){document.getElementById('logModal').style.display='none';}

async function pullModel(){
  var ref=document.getElementById('pullInput').value.trim();
  if(!ref){alert('Enter a model reference');return;}
  var btn=document.getElementById('pullBtn'),st=document.getElementById('pullStatus');
  btn.disabled=true;btn.textContent='Pulling...';st.textContent='Downloading...';
  try{
    var r=await fetch('/api/v1/models/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:ref})});
    var d=await r.json();
    if(d.error){st.textContent='Error: '+d.error;alert(d.error);}else{st.textContent='✅ '+ref;loadModels();}
  }catch(e){st.textContent='Error: '+e;alert(e);}
  btn.disabled=false;btn.textContent='Pull';
}

function refreshChat(){if(chatPort)selectChatFor(chatPort,'');}

// ── Theme ──
function toggleTheme(){
  var b=document.body;
  b.classList.toggle('light');
  document.getElementById('themeToggle').textContent=b.classList.contains('light')?'☀️':'🌙';
  localStorage.setItem('gollama-theme',b.classList.contains('light')?'light':'dark');
}
(function(){
  if(localStorage.getItem('gollama-theme')==='light'){document.body.classList.add('light');document.getElementById('themeToggle').textContent='☀️';}
})();

// ── Init ──
loadModels();loadInstances();refreshChatSelector();
setInterval(function(){loadInstances();refreshChatSelector();},3000);
setInterval(function(){loadModelList();},10000);
</script>

<div id="logModal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.7);z-index:1000">
  <div style="background:var(--surface);margin:5% auto;padding:20px;width:80%;max-width:700px;max-height:70vh;border-radius:10px;overflow:auto;border:1px solid var(--border)">
    <div class="flex"><h2 style="margin-bottom:0">📋 Logs</h2><button class="small danger" onclick="closeLogs()">Close</button></div>
    <pre id="logContent" style="background:var(--input-bg);padding:12px;border-radius:6px;margin-top:12px;font-size:11px;line-height:1.4;overflow:auto;max-height:55vh;white-space:pre-wrap;color:var(--muted)"></pre>
  </div>
</div>
</body>
</html>`
