package optimizer

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/kaiwu-ai/kaiwu/internal/config"
	"github.com/kaiwu-ai/kaiwu/internal/engine"
	"github.com/kaiwu-ai/kaiwu/internal/hardware"
	"github.com/kaiwu-ai/kaiwu/internal/model"
)

// OptimizedProfile is the result of warmup benchmark
type OptimizedProfile struct {
	ModelID     string   `json:"model_id"`
	HardwareFP  string   `json:"hardware_fp"`
	Quant       string   `json:"quant"`
	Mode        string   `json:"mode"`
	MeasuredTPS float64  `json:"measured_tps"`
	VRAMUsed_MB int      `json:"vram_used_mb"`
	LaunchArgs  []string `json:"launch_args"`
	CreatedAt   string   `json:"created_at"`
}

// Warmup runs the warmup benchmark and returns optimized parameters
func Warmup(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, fast bool) (*OptimizedProfile, error) {
	fingerprint := hw.Fingerprint()
	profilePath := filepath.Join(config.ProfileDir(), fmt.Sprintf("%s_%s.json", profile.ModelID, fingerprint))

	// Always check cache first (spec: second launch should be 2s)
	if cached, err := loadCachedProfile(profilePath); err == nil {
		if isCacheValid(cached, fingerprint) {
			fmt.Printf("      Using cached profile (%.1f tok/s)\n", cached.MeasuredTPS)
			return cached, nil
		}
		fmt.Printf("      Cache expired, re-running warmup\n")
	}

	// --fast with no cache: skip warmup entirely
	if fast {
		fmt.Printf("      No cached profile, using defaults\n")
		return nil, fmt.Errorf("no cached profile available")
	}

	cfg, _ := config.Load()
	port := cfg.LlamaPort + 10 // Use offset port to avoid conflicts

	// Round 1: Start with recommended params
	fmt.Printf("      Round 1: ")
	args1 := BuildArgs(profile, modelPath, port, hw, 512, 128)
	tps1, vram1, err := runBenchmarkRound(binaryPath, args1, port)
	if err != nil {
		return nil, fmt.Errorf("warmup round 1 failed: %w", err)
	}
	fmt.Printf("%.1f tok/s (VRAM: %d MB)\n", tps1, vram1)

	bestTPS := tps1
	bestArgs := args1
	bestVRAM := vram1

	// Round 2: Adjust batch size based on VRAM usage
	gpu := hw.PrimaryGPU()
	vramTotal := 0
	if gpu != nil {
		vramTotal = gpu.VRAM_MB
	}

	var batchSize, ubatchSize int
	if profile.Mode == "moe_offload" {
		batchSize = 4096
		ubatchSize = 512
	} else {
		batchSize = 512
		ubatchSize = 128
	}

	if vramTotal > 0 {
		vramPct := float64(vram1) / float64(vramTotal) * 100
		if vramPct < 80 {
			// Room to grow: increase batch size
			batchSize *= 2
			ubatchSize *= 2
		} else if vramPct > 95 {
			// Too tight: reduce batch size
			batchSize /= 2
			ubatchSize /= 2
		}
	}

	fmt.Printf("      Round 2: ")
	args2 := BuildArgs(profile, modelPath, port, hw, batchSize, ubatchSize)
	tps2, vram2, err := runBenchmarkRound(binaryPath, args2, port)
	if err != nil {
		fmt.Printf("failed, keeping Round 1 params\n")
	} else {
		fmt.Printf("%.1f tok/s", tps2)
		if tps2 > bestTPS {
			improvement := (tps2 - bestTPS) / bestTPS * 100
			fmt.Printf(" (+%.0f%%)\n", improvement)
			bestTPS = tps2
			bestArgs = args2
			bestVRAM = vram2
		} else {
			fmt.Printf(" (no improvement)\n")
		}
	}

	// Round 3: Fine-tune if Round 2 improved
	if tps2 > tps1 {
		fmt.Printf("      Round 3: ")
		// Try slightly different batch size
		batchSize3 := batchSize * 3 / 4
		ubatchSize3 := ubatchSize * 3 / 4
		args3 := BuildArgs(profile, modelPath, port, hw, batchSize3, ubatchSize3)
		tps3, vram3, err := runBenchmarkRound(binaryPath, args3, port)
		if err != nil {
			fmt.Printf("failed, keeping Round 2 params\n")
		} else {
			fmt.Printf("%.1f tok/s", tps3)
			if tps3 > bestTPS {
				fmt.Printf(" (improved)\n")
				bestTPS = tps3
				bestArgs = args3
				bestVRAM = vram3
			} else {
				fmt.Printf(" (no improvement, keeping Round 2)\n")
				_ = vram3
			}
		}
	} else {
		fmt.Printf("      Round 3: %.1f tok/s (no improvement, keeping Round 1)\n", bestTPS)
	}

	// Save profile
	optimized := &OptimizedProfile{
		ModelID:     profile.ModelID,
		HardwareFP:  fingerprint,
		Quant:       profile.Quant,
		Mode:        profile.Mode,
		MeasuredTPS: bestTPS,
		VRAMUsed_MB: bestVRAM,
		LaunchArgs:  bestArgs,
		CreatedAt:   time.Now().Format(time.RFC3339),
	}

	if err := saveProfile(optimized, profilePath); err != nil {
		fmt.Printf("      Warning: failed to save profile: %v\n", err)
	} else {
		fmt.Printf("      Saved profile: %s\n", profilePath)
	}

	return optimized, nil
}

// BuildArgs constructs llama-server arguments with specific batch sizes
func BuildArgs(profile *model.DeployProfile, modelPath string, port int, hw *hardware.HardwareProbe, batchSize, ubatchSize int) []string {
	gpu := hw.PrimaryGPU()
	vramGB := 0
	if gpu != nil {
		vramGB = gpu.VRAM_MB / 1024
	}

	// Use params rules for ctx-size and KV cache
	ctxSize := DynamicCtxSize(hw, profile)

	// KV cache 量化：iso3 优先
	kvK := "q8_0"
	kvV := kvCacheVByVRAM(vramGB)
	if profile.HasIsoQuant {
		kvK = "iso3"
		kvV = "iso3"
	}

	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", "999",
		"--flash-attn", "on",
		"--cont-batching",
		"--metrics",
		"--no-webui",
		"--cache-reuse", "256",
		"-ctk", kvK,
		"-ctv", kvV,
		"--ctx-size", strconv.Itoa(ctxSize),
		"--threads", strconv.Itoa(hw.CPU.Cores),
		"--batch-size", strconv.Itoa(batchSize),
		"--ubatch-size", strconv.Itoa(ubatchSize),
	}

	// MoE-specific parameters
	if profile.Mode == "moe_offload" && profile.OTFlags != "" {
		// Parse OT flags: '-ot ".ffn_.*_exps.=CPU"'
		otValue := strings.TrimPrefix(profile.OTFlags, "-ot ")
		otValue = strings.Trim(otValue, "\"'")
		args = append(args, "-ot", otValue)
	}

	// MTP speculative decoding — disabled until llama-server supports it
	// if profile.NativeMTP {
	// 	args = append(args,
	// 		"--speculative-algo", "NEXTN",
	// 		"--speculative-num-steps", "3",
	// 		"--speculative-num-draft-tokens", "4",
	// 	)
	// }

	return args
}

// runBenchmarkRound starts llama-server, runs a warmup prompt, measures tok/s
func runBenchmarkRound(binaryPath string, args []string, port int) (tps float64, vramMB int, err error) {
	// Start llama-server
	eng, err := startBenchServer(binaryPath, args, port)
	if err != nil {
		return 0, 0, err
	}
	defer stopBenchServer(eng)

	// Send warmup prompt and measure
	tps, err = measureTPS(port)
	if err != nil {
		return 0, 0, err
	}

	// Get VRAM usage from metrics
	vramMB = getVRAMFromMetrics(port)

	return tps, vramMB, nil
}

// startBenchServer starts llama-server for benchmarking
func startBenchServer(binaryPath string, args []string, port int) (*os.Process, error) {
	// Create log file for debugging
	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("warmup-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		logFile = nil
	}

	cmd := exec.Command(binaryPath, args...)
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	engine.SetProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			logFile.Close()
		}
		return nil, fmt.Errorf("failed to start benchmark server: %w", err)
	}

	// Close log file when process exits
	go func() {
		cmd.Wait()
		if logFile != nil {
			logFile.Close()
		}
	}()

	// Wait for health endpoint
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return cmd.Process, nil
			}
		}
		time.Sleep(1 * time.Second)
	}

	cmd.Process.Kill()
	return nil, fmt.Errorf("benchmark server failed to start within 60s (log: %s)", logPath)
}

// stopBenchServer stops the benchmark server
func stopBenchServer(proc *os.Process) {
	if proc != nil {
		proc.Kill()
		proc.Wait()
	}
}

// measureTPS sends a prompt and measures decode tokens per second
func measureTPS(port int) (float64, error) {
	reqBody := `{
		"model": "test",
		"messages": [{"role": "user", "content": "Write a Python quicksort implementation with detailed comments and edge case handling."}],
		"max_tokens": 200,
		"stream": false
	}`

	start := time.Now()
	resp, err := http.Post(
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		return 0, fmt.Errorf("benchmark request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read response: %w", err)
	}
	elapsed := time.Since(start)

	// Parse response to get token count
	var result struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("failed to parse response: %w", err)
	}

	tokens := result.Usage.CompletionTokens
	if tokens == 0 {
		return 0, fmt.Errorf("no tokens generated")
	}

	tps := float64(tokens) / elapsed.Seconds()
	return tps, nil
}

// getVRAMFromMetrics reads VRAM usage from llama-server metrics endpoint
func getVRAMFromMetrics(port int) int {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	// Look for VRAM metric in Prometheus format
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "llama_vram_usage_bytes") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				val, _ := strconv.ParseFloat(parts[1], 64)
				return int(val / (1024 * 1024))
			}
		}
	}
	return 0
}

func loadCachedProfile(path string) (*OptimizedProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile OptimizedProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// isCacheValid checks if cached profile is still valid (fingerprint match + < 30 days old)
func isCacheValid(cached *OptimizedProfile, currentFingerprint string) bool {
	// Check fingerprint match
	if cached.HardwareFP != currentFingerprint {
		return false
	}
	// Check age (30 days)
	created, err := time.Parse(time.RFC3339, cached.CreatedAt)
	if err != nil {
		return false
	}
	return time.Since(created) < 30*24*time.Hour
}

func saveProfile(profile *OptimizedProfile, path string) error {
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
