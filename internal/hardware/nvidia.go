package hardware

import (
	"encoding/xml"
	"os/exec"
	"strconv"
	"strings"
)

// nvidiaSMILog is the top-level XML structure from nvidia-smi -q -x
type nvidiaSMILog struct {
	DriverVersion string       `xml:"driver_version"`
	CUDAVersion   string       `xml:"cuda_version"`
	GPUs          []nvidiaSMIG `xml:"gpu"`
}

type nvidiaSMIG struct {
	ProductName string `xml:"product_name"`
	FBMemory    struct {
		Total string `xml:"total"`
		Used  string `xml:"used"`
		Free  string `xml:"free"`
	} `xml:"fb_memory_usage"`
}

// detectNVIDIA detects NVIDIA GPUs using nvidia-smi XML output + CSV for compute_cap.
// XML is stable across all driver versions and GPU types (GeForce/Tesla/Quadro).
func detectNVIDIA() ([]GPUInfo, error) {
	// Step 1: XML for name, memory, driver
	xmlOut, err := exec.Command("nvidia-smi", "-q", "-x").Output()
	if err != nil {
		return nil, err
	}

	var smiLog nvidiaSMILog
	if err := xml.Unmarshal(xmlOut, &smiLog); err != nil {
		return nil, err
	}

	if len(smiLog.GPUs) == 0 {
		return nil, nil
	}

	// Step 2: CSV for compute_cap (not available in XML)
	computeCaps := queryComputeCaps(len(smiLog.GPUs))

	gpus := make([]GPUInfo, 0, len(smiLog.GPUs))
	for i, g := range smiLog.GPUs {
		vramTotal := parseMiB(g.FBMemory.Total)
		vramUsed := parseMiB(g.FBMemory.Used)
		vramFree := parseMiB(g.FBMemory.Free)

		cc := ""
		if i < len(computeCaps) {
			cc = computeCaps[i]
		}

		isBlackwell := strings.HasPrefix(cc, "12")

		gpus = append(gpus, GPUInfo{
			Index:            i,
			Name:             strings.TrimSpace(g.ProductName),
			VRAM_MB:          vramTotal,
			VRAMUsed_MB:      vramUsed,
			VRAMFree_MB:      vramFree,
			ComputeCap:       cc,
			CUDADriver:       smiLog.CUDAVersion,
			MemBandwidth_GBs: estimateBandwidth(strings.TrimSpace(g.ProductName)),
			IsBlackwell:      isBlackwell,
		})
	}

	return gpus, nil
}

// queryComputeCaps reads compute capability via CSV (simple, stable format for this one field)
func queryComputeCaps(gpuCount int) []string {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=compute_cap",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return make([]string, gpuCount)
	}

	caps := make([]string, 0, gpuCount)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			caps = append(caps, line)
		}
	}
	return caps
}

// parseMiB extracts integer MiB from strings like "24564 MiB" or "24564"
func parseMiB(s string) int {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, " MiB")
	s = strings.TrimSpace(s)
	v, _ := strconv.Atoi(s)
	return v
}

// detectNVLink checks if NVLink is present between GPUs via nvidia-smi
func detectNVLink() bool {
	out, err := exec.Command("nvidia-smi", "nvlink", "--status").Output()
	if err != nil {
		return false
	}
	return strings.Contains(strings.ToLower(string(out)), "active")
}

// estimateBandwidth estimates memory bandwidth based on GPU name
func estimateBandwidth(name string) float64 {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "5090"):
		return 1792.0
	case strings.Contains(n, "5080"):
		return 960.0
	case strings.Contains(n, "5070 ti"):
		return 896.0
	case strings.Contains(n, "5070"):
		return 672.0
	case strings.Contains(n, "5060"):
		return 448.0
	case strings.Contains(n, "4090"):
		return 1008.0
	case strings.Contains(n, "4080"):
		return 717.0
	case strings.Contains(n, "4070 ti"):
		return 504.0
	case strings.Contains(n, "4070"):
		return 504.0
	case strings.Contains(n, "4060 ti"):
		return 288.0
	case strings.Contains(n, "4060"):
		return 272.0
	case strings.Contains(n, "3090"):
		return 936.0
	case strings.Contains(n, "3080"):
		return 760.0
	case strings.Contains(n, "a100"):
		return 2039.0
	case strings.Contains(n, "h100"):
		return 3350.0
	case strings.Contains(n, "p40"):
		return 346.0
	default:
		return 0.0
	}
}
