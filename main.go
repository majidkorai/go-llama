package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
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

func indexFile() string {
	return filepath.Join(goLLamaDir(), "index.json")
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
	if path, err := exec.LookPath("llama-server"); err == nil {
		return path
	}
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
	// 1. Check go-llama local models
	idx := loadIndex()
	if info, ok := idx[model]; ok {
		if _, err := os.Stat(info.BlobPath); err == nil {
			return info.BlobPath, nil
		}
		// Stale index entry — remove it
		delete(idx, model)
		saveIndex(idx)
	}

	// 2. Check direct file path
	if _, err := os.Stat(model); err == nil {
		return model, nil
	}

	// 3. Check Ollama blobs (optional compatibility)
	home, _ := os.UserHomeDir()
	blobDir := filepath.Join(home, ".ollama", "models", "blobs")
	manifestDir := filepath.Join(home, ".ollama", "models", "manifests")

	var found string
	filepath.Walk(manifestDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || found != "" {
			return nil
		}
		data, _ := os.ReadFile(path)
		var m struct {
			Config struct {
				Digest string `json:"digest"`
			} `json:"config"`
		}
		json.Unmarshal(data, &m)

		if strings.Contains(path, model) && m.Config.Digest != "" {
			blobFile := strings.Replace(m.Config.Digest, ":", "-", 1)
			bp := filepath.Join(blobDir, blobFile)
			if _, err := os.Stat(bp); err == nil {
				found = bp
			}
		}
		return nil
	})

	if found != "" {
		return found, nil
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
	seen := make(map[string]bool)

	// 1. go-llama local models
	idx := loadIndex()
	for name, info := range idx {
		if _, err := os.Stat(info.BlobPath); err == nil {
			info.Source = "local"
			models = append(models, info)
			seen[name] = true
		}
	}

	// 2. Ollama models (optional)
	home, _ := os.UserHomeDir()
	manifestDir := filepath.Join(home, ".ollama", "models", "manifests")
	blobDir := filepath.Join(home, ".ollama", "models", "blobs")

	filepath.Walk(manifestDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		data, _ := os.ReadFile(path)
		var manifest struct {
			Config struct{ Digest string } `json:"config"`
			Layers []struct {
				MediaType string `json:"mediaType"`
				Digest    string `json:"digest"`
				Size      int64  `json:"size"`
			} `json:"layers"`
		}
		if json.Unmarshal(data, &manifest) != nil {
			return nil
		}

		var blobDigest string
		var size int64
		for _, layer := range manifest.Layers {
			if strings.Contains(layer.MediaType, "gguf") {
				blobDigest = layer.Digest
				size = layer.Size
				break
			}
		}
		if blobDigest == "" {
			return nil
		}

		blobFile := strings.Replace(blobDigest, ":", "-", 1)
		blobPath := filepath.Join(blobDir, blobFile)

		relPath, _ := filepath.Rel(manifestDir, path)
		modelName := strings.ReplaceAll(relPath, string(filepath.Separator), ":")
		modelName = strings.TrimSuffix(modelName, filepath.Ext(modelName))
		if idx := strings.LastIndex(modelName, ":"); idx > 0 {
			modelName = modelName[:idx] + ":" + modelName[idx+1:]
		}

		if !seen[modelName] {
			models = append(models, ModelInfo{
				Name:     modelName,
				BlobPath: blobPath,
				Size:     size,
				Source:   "ollama",
			})
			seen[modelName] = true
		}
		return nil
	})

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
		model := os.Args[2]
		extraArgs := os.Args[3:]
		inst, err := mgr.Start(model, 8081, extraArgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Started %s on port %d (PID %d)\n", inst.Model, inst.Port, inst.PID)

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
  go-llama pull <model>           Download model from HuggingFace
  go-llama list                   List available models
  go-llama serve [port]           Start manager with web UI (default :9080)
  go-llama run <model> [flags]    Quick-start a model on port 8081
  go-llama ps                     List running instances
  go-llama stop <port>            Stop an instance

Examples:
  go-llama pull hf.co/Jackrong/Qwopus3.6-27B-v2-GGUF:Q4_K_M
  go-llama list
  go-llama run qwen3.6:35b --tensor-split 12,8
  go-llama serve`)
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
h1 { color: #a78bfa; margin-bottom: 20px; }
h2 { color: #c4b5fd; margin: 20px 0 10px; }
.card { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
.instance { border-left: 3px solid #22c55e; }
.instance.stopped { border-left-color: #ef4444; }
label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
select, input, button { width: 100%; padding: 8px; background: #0f172a; border: 1px solid #334155; border-radius: 4px; color: #e2e8f0; font-size: 14px; margin-bottom: 12px; }
button { background: #7c3aed; border: none; cursor: pointer; font-weight: 600; }
button:hover { background: #6d28d9; }
button.danger { background: #dc2626; }
button.danger:hover { background: #b91c1c; }
.flag-row { display: flex; gap: 8px; margin-bottom: 8px; }
.flag-row input { flex: 1; }
.flag-row button { width: auto; padding: 8px 16px; background: #ef4444; }
.mt-10 { margin-top: 10px; }
.text-sm { font-size: 12px; color: #64748b; }
.flex { display: flex; justify-content: space-between; align-items: center; }
</style>
</head>
<body>
<h1>go-llama</h1>

<div class="card">
  <h2>New Instance</h2>
  <label>Model</label>
  <select id="modelSelect"><option value="">Loading...</option></select>
  <label>Port</label>
  <input type="number" id="portInput" value="8081" min="8081" max="8099">
  <label>Extra Flags</label>
  <div id="flagsContainer">
    <div class="flag-row">
      <input type="text" placeholder="--flag value" class="flag-input">
      <button onclick="this.parentElement.remove()">x</button>
    </div>
  </div>
  <button onclick="addFlag()">+ Add Flag</button>
  <button class="mt-10" onclick="launchInstance()">Launch</button>
</div>

<div class="card">
  <h2>Running Instances</h2>
  <div id="instances" class="grid"></div>
</div>

<script>
function addFlag(){
  var c=document.getElementById('flagsContainer'),r=document.createElement('div');
  r.className='flag-row';
  r.innerHTML='<input type="text" placeholder="--flag value" class="flag-input"><button onclick="this.parentElement.remove()">x</button>';
  c.appendChild(r);
}
async function loadModels(){
  var r=await fetch('/api/v1/models'),m=await r.json(),s=document.getElementById('modelSelect'),seen={};
  s.innerHTML='<option value="">Select model...</option>';
  m.forEach(function(x){if(!seen[x.Name]){seen[x.Name]=1;s.innerHTML+='<option value="'+x.Name+'">'+x.Name+'</option>';}});
}
async function loadInstances(){
  var r=await fetch('/api/v1/instances'),list=await r.json(),c=document.getElementById('instances');
  if(!list.length){c.innerHTML='<div class="text-sm">No running instances</div>';return;}
  c.innerHTML=list.map(function(i){return'<div class="card instance"><div class="flex"><strong>'+i.Model+'</strong><span>:'+i.Port+'</span></div><div class="text-sm mt-10">PID: '+i.PID+' | Status: '+i.Status+'</div><button class="danger mt-10" onclick="stopInstance('+i.Port+')">Stop</button></div>';}).join('');
}
async function launchInstance(){
  var m=document.getElementById('modelSelect').value,p=parseInt(document.getElementById('portInput').value),f=[];
  document.querySelectorAll('.flag-input').forEach(function(el){var v=el.value.trim();if(v)f.push(v);});
  var r=await fetch('/api/v1/instances',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m,port:p,flags:f})});
  if(!r.ok){alert('Error: '+await r.text());return;}
  var i=await r.json();
  document.getElementById('portInput').value=i.Port+1;
  loadInstances();
}
async function stopInstance(p){await fetch('/api/v1/instances/stop?port='+p,{method:'POST'});loadInstances();}
loadModels();loadInstances();setInterval(loadInstances,5000);
</script>
</body>
</html>`
