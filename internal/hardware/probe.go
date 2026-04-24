package hardware

import (
	"encoding/json"
	"fmt"
)

// HardwareProbe contains detected hardware information
type HardwareProbe struct {
	GPUs []GPUInfo `json:"gpus"`
	CPU  CPUInfo   `json:"cpu"`
	RAM  RAMInfo   `json:"ram"`
	OS   OSInfo    `json:"os"`
}

// GPUInfo holds per-GPU information
type GPUInfo struct {
	Index            int     `json:"index"`
	Name             string  `json:"name"`
	VRAM_MB          int     `json:"vram_mb"`
	VRAMUsed_MB      int     `json:"vram_used_mb"`
	VRAMFree_MB      int     `json:"vram_free_mb"`
	ComputeCap       string  `json:"compute_cap"`        // "8.9" for SM89
	CUDADriver       string  `json:"cuda_driver"`        // "12.8"
	MemBandwidth_GBs float64 `json:"mem_bandwidth_gbs"`  // 1008.0
	IsBlackwell      bool    `json:"is_blackwell"`       // SM120x
}

// CPUInfo holds CPU information
type CPUInfo struct {
	Model       string `json:"model"`
	Cores       int    `json:"cores"`
	Threads     int    `json:"threads"`
	HasAVX2     bool   `json:"has_avx2"`
	HasAVX512   bool   `json:"has_avx512"`
}

// RAMInfo holds RAM information
type RAMInfo struct {
	Total_MB uint64 `json:"total_mb"`
	Used_MB  uint64 `json:"used_mb"`
	Free_MB  uint64 `json:"free_mb"`
	Type     string `json:"type"` // "ddr4", "ddr5", "unknown"
}

// OSInfo holds OS information
type OSInfo struct {
	Platform string `json:"platform"` // "windows", "linux"
	Arch     string `json:"arch"`     // "amd64", "arm64"
	Version  string `json:"version"`  // OS version string
}

// Probe detects hardware and returns a HardwareProbe
func Probe() (*HardwareProbe, error) {
	probe := &HardwareProbe{}

	// Detect GPUs (NVIDIA first, then AMD fallback)
	gpus, err := detectNVIDIA()
	if err == nil && len(gpus) > 0 {
		probe.GPUs = gpus
	} else {
		// Try AMD detection
		gpus, err = detectAMD()
		if err == nil {
			probe.GPUs = gpus
		}
	}

	// Detect CPU
	cpu, err := detectCPU()
	if err != nil {
		return nil, fmt.Errorf("CPU detection failed: %w", err)
	}
	probe.CPU = cpu

	// Detect RAM
	ram, err := detectRAM()
	if err != nil {
		return nil, fmt.Errorf("RAM detection failed: %w", err)
	}
	probe.RAM = ram

	// Detect OS
	probe.OS = detectOS()

	return probe, nil
}

// PrimaryGPU returns the first GPU (highest VRAM)
func (p *HardwareProbe) PrimaryGPU() *GPUInfo {
	if len(p.GPUs) == 0 {
		return nil
	}
	primary := &p.GPUs[0]
	for i := range p.GPUs {
		if p.GPUs[i].VRAM_MB > primary.VRAM_MB {
			primary = &p.GPUs[i]
		}
	}
	return primary
}

// TotalVRAM_MB returns the sum of VRAM across all GPUs
func (p *HardwareProbe) TotalVRAM_MB() int {
	total := 0
	for _, g := range p.GPUs {
		total += g.VRAM_MB
	}
	return total
}

// GPUCount returns the number of GPUs
func (p *HardwareProbe) GPUCount() int {
	return len(p.GPUs)
}

// Fingerprint returns a unique hardware fingerprint for profile caching
// Format: "sm89_24576mb_ddr5"
func (p *HardwareProbe) Fingerprint() string {
	gpu := p.PrimaryGPU()
	if gpu == nil {
		return fmt.Sprintf("cpu_%dmb_%s", p.RAM.Total_MB, p.RAM.Type)
	}

	// Remove dots from compute cap: "8.9" -> "89"
	cc := gpu.ComputeCap
	cc = cc[:len(cc)-2] + cc[len(cc)-1:]

	return fmt.Sprintf("sm%s_%dmb_%s", cc, gpu.VRAM_MB, p.RAM.Type)
}

// JSON returns JSON representation
func (p *HardwareProbe) JSON() (string, error) {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
