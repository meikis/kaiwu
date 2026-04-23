package engine

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/kaiwu-ai/kaiwu/internal/config"
	"github.com/kaiwu-ai/kaiwu/internal/hardware"
	"github.com/kaiwu-ai/kaiwu/internal/model"
)

// RunningEngine represents a running llama-server instance
type RunningEngine struct {
	PID        int
	Port       int
	ModelID    string
	BinaryPath string
	LogPath    string
	logFile    *os.File // 保持日志文件句柄，进程退出时再关
}

// Start starts llama-server with the given profile
func Start(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe) (*RunningEngine, error) {
	// 运行时探测 llama-server 是否支持 iso3
	if profile.HasIsoQuant && !detectIso3Support(binaryPath) {
		fmt.Println("      llama-server 不支持 iso3，回退到 q8_0/q4_0")
		profile.HasIsoQuant = false
	}
	return StartWithArgs(profile, binaryPath, modelPath, hw, nil)
}

// StartWithArgs starts llama-server, optionally using pre-optimized args from warmup.
// If optimizedArgs is nil, buildArgs() generates defaults.
func StartWithArgs(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, optimizedArgs []string) (*RunningEngine, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	// 检测端口冲突，自动找可用端口
	actualPort := findFreePort(cfg.LlamaPort)
	if actualPort != cfg.LlamaPort {
		fmt.Printf("Port %d in use, using %d instead\n", cfg.LlamaPort, actualPort)
	}

	var args []string
	if len(optimizedArgs) > 0 {
		// Use warmup-optimized args, but patch the port
		args = make([]string, len(optimizedArgs))
		copy(args, optimizedArgs)
		for i, a := range args {
			if a == "--port" && i+1 < len(args) {
				args[i+1] = strconv.Itoa(actualPort)
				break
			}
		}
	} else {
		args = buildArgs(profile, modelPath, actualPort, hw)
	}

	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("llama-server-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	// 不要 defer close — 进程需要持续写入日志

	cmd := exec.Command(binaryPath, args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	setProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// 进程退出后自动关闭日志文件
	go func() {
		cmd.Wait()
		logFile.Close()
	}()

	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0644); err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("failed to write PID file: %w", err)
	}

	eng := &RunningEngine{
		PID:        cmd.Process.Pid,
		Port:       actualPort,
		ModelID:    profile.ModelID,
		BinaryPath: binaryPath,
		LogPath:    logPath,
		logFile:    logFile,
	}

	fmt.Printf("Waiting for llama-server to be ready (port %d)...\n", actualPort)
	// Longer timeout for large models (MoE 30B needs ~60s to load)
	if err := waitForPort("127.0.0.1", actualPort, 90*time.Second); err != nil {
		Stop()
		return nil, fmt.Errorf("llama-server failed to start: %w", err)
	}

	return eng, nil
}

// Stop stops the running llama-server
func Stop() error {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no running model found")
		}
		return fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return fmt.Errorf("invalid PID in file: %w", err)
	}

	if err := killProcess(pid); err != nil {
		return err
	}

	os.Remove(pidPath)
	return nil
}

// Status returns the status of the running engine
func Status() (*RunningEngine, error) {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return nil, fmt.Errorf("invalid PID in file: %w", err)
	}

	if !isProcessAlive(pid) {
		os.Remove(pidPath)
		return nil, nil
	}

	cfg, _ := config.Load()
	return &RunningEngine{
		PID:  pid,
		Port: cfg.LlamaPort,
	}, nil
}

// buildArgs constructs llama-server command-line arguments
func buildArgs(profile *model.DeployProfile, modelPath string, port int, hw *hardware.HardwareProbe) []string {
	gpu := hw.PrimaryGPU()
	vramGB := 0
	if gpu != nil {
		vramGB = gpu.VRAM_MB / 1024
	}

	// 动态计算 ctx：根据实际 VRAM 和模型大小（iso3 压缩后上下文大幅扩展）
	ctxSize := dynamicCtxSize(gpu.VRAM_MB, profile.Size_GB, profile.Layers, profile.CtxOverride, profile.HasIsoQuant)
	threads := hw.CPU.Cores / 2
	if threads < 1 {
		threads = 1
	}

	args := []string{
		"--model", modelPath,
		"--alias", profile.ModelID,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", "999",   // 全部层放 GPU
		"--flash-attn", "on",      // Flash Attention 加速
		"--cont-batching",         // 连续批处理
		"--metrics",               // 暴露 /metrics 端点
		"--no-webui",              // 不启动 Web UI
		"--cache-reuse", "256",    // KV cache 复用窗口
		"--ctx-size", strconv.Itoa(ctxSize), // 上下文大小（动态计算）
		"--threads", strconv.Itoa(threads),  // CPU 线程数（核心数/2）
	}

	// KV cache 量化：IsoQuant 3-bit 压缩
	// iso3: K+V 合计 0.75 bytes/element（vs q8_0+q4_0 的 1.5），压缩 2x
	// 省出的显存用来扩展上下文到 128K+
	// 回退：如果 iso3 不支持当前模型的 head_dim，llama-server 会自动回退到 q8_0/q4_0
	if profile.HasIsoQuant {
		args = append(args, "-ctk", "iso3", "-ctv", "iso3")
	} else {
		args = append(args, "-ctk", "q8_0")
		if vramGB > 16 {
			args = append(args, "-ctv", "f16")
		} else {
			args = append(args, "-ctv", "q4_0")
		}
	}

	// MoE offload 参数
	// 如果是 MoE 模型且需要 offload 到 CPU
	if profile.Mode == "moe_offload" && profile.OTFlags != "" {
		args = append(args, "-ot", profile.OTFlags)      // MoE offload 模板
		args = append(args, "--batch-size", "4096")      // 大 batch 提升 CPU 利用率
		args = append(args, "--ubatch-size", "512")      // 微批次大小
		// MoE offload 需要更多 CPU 线程
		threads = hw.CPU.Cores * 2 / 3
		if threads < 1 {
			threads = 1
		}
		// 更新 threads 参数
		for i, a := range args {
			if a == "--threads" && i+1 < len(args) {
				args[i+1] = strconv.Itoa(threads)
				break
			}
		}
	} else {
		// Full GPU 默认参数
		args = append(args, "--batch-size", "512")   // 标准 batch size
		args = append(args, "--ubatch-size", "128")  // 标准微批次
	}

	return args
}

// detectIso3Support 检测 llama-server 是否支持 iso3 KV cache 量化
// 通过解析 --help 输出中的 allowed values 判断
func detectIso3Support(binaryPath string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, binaryPath, "--help").CombinedOutput()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), "iso3")
}
