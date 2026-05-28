package main

import (
	"encoding/json"
	"fmt"
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
	home, _ := os.UserHomeDir()
	blobDir := filepath.Join(home, ".ollama", "models", "blobs")
	manifestDir := filepath.Join(home, ".ollama", "models", "manifests")

	// Walk manifests to find model
	filepath.Walk(manifestDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		data, _ := os.ReadFile(path)
		var m struct {
			Config struct {
				Digest string `json:"digest"`
			} `json:"config"`
		}
		json.Unmarshal(data, &m)

		// Simple match: check if manifest path contains model name
		if strings.Contains(path, model) && m.Config.Digest != "" {
			blobFile := strings.Replace(m.Config.Digest, ":", "-", 1)
			blobPath := filepath.Join(blobDir, blobFile)
			if _, err := os.Stat(blobPath); err == nil {
				return fmt.Errorf("FOUND:%s", blobPath) // hack to break walk
			}
		}
		return nil
	})

	// Direct file path
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

func listOllamaModels() ([]ModelInfo, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	manifestDir := filepath.Join(home, ".ollama", "models", "manifests")
	blobDir := filepath.Join(home, ".ollama", "models", "blobs")

	var models []ModelInfo

	filepath.Walk(manifestDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		data, _ := os.ReadFile(path)
		var manifest struct {
			Config struct {
				Digest string `json:"digest"`
			} `json:"config"`
			Layers []struct {
				MediaType string `json:"mediaType"`
				Digest    string `json:"digest"`
				Size      int64  `json:"size"`
			} `json:"layers"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
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
		// Remove last segment if it's the tag
		if idx := strings.LastIndex(modelName, ":"); idx > 0 {
			modelName = modelName[:idx] + ":" + modelName[idx+1:]
		}

		models = append(models, ModelInfo{
			Name:     modelName,
			BlobPath: blobPath,
			Size:     size,
		})
		return nil
	})

	return models, nil
}

// ── API Server ─────────────────────────────────────────────────────────

func startServer(mgr *Manager, port string) {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/models", func(w http.ResponseWriter, r *http.Request) {
		models, err := listOllamaModels()
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

// ── CLI ────────────────────────────────────────────────────────────────

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	mgr := NewManager()

	switch os.Args[1] {
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
  go-llama serve [port]           Start manager with web UI (default :9080)
  go-llama run <model> [flags]    Quick-start a model
  go-llama ps                     List running instances
  go-llama stop <port>            Stop an instance`)
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
