package engine

import (
	"fmt"

	"github.com/kaiwu-ai/kaiwu/internal/hardware"
	"github.com/kaiwu-ai/kaiwu/internal/model"
)

// PreflightCheck verifies there's enough VRAM to run the model.
// If a MoE model doesn't fit in full_gpu mode, it auto-switches to offload.
func PreflightCheck(profile *model.DeployProfile, hw *hardware.HardwareProbe) error {
	gpu := hw.PrimaryGPU()
	if gpu == nil {
		// CPU-only: no VRAM check needed
		return nil
	}

	// Use total VRAM minus system reserve (300MB for desktop compositor etc.)
	// Not free VRAM — llama-server will reclaim VRAM from other idle apps
	vramAvailGB := float64(gpu.VRAM_MB-300) / 1024.0
	modelSizeGB := profile.Size_GB
	ctxSize := dynamicCtxSize(gpu.VRAM_MB, profile.Size_GB, profile.Layers, profile.CtxOverride, profile.HasIsoQuant)
	kvEstimateGB := estimateKVCacheGB(ctxSize, profile.Layers, profile.HasIsoQuant)
	// iso3 有 ~0.6GB 固定解码 buffer 开销（实测 4090 + 35B MoE 验证）
	overhead := 0.3
	if profile.HasIsoQuant {
		overhead = 0.9 // 0.3 通用 + 0.6 iso3 解码 buffer
	}
	totalNeeded := modelSizeGB + kvEstimateGB + overhead

	if profile.Mode == "full_gpu" && totalNeeded > vramAvailGB {
		// MoE can fall back to offload
		if profile.OTFlags != "" || profile.Arch == "moe" {
			kvSmaller := estimateKVCacheGB(ctxSize/2, profile.Layers, profile.HasIsoQuant)
			offloadGPU := modelSizeGB*0.1 + kvSmaller + 0.3
			if offloadGPU < vramAvailGB {
				fmt.Println("      ⚠️  显存不足，自动启用 MoE offload 模式")
				profile.Mode = "moe_offload"
				// MoE offload 需要大量 RAM，检查是否足够
				checkRAMForOffload(profile, hw)
				return nil
			}
		}
		return fmt.Errorf("显存不足：需要 %.1f GB，可用 %.1f GB\n"+
			"  建议：选择更小的量化或使用 MoE offload 模型",
			totalNeeded, vramAvailGB)
	}

	return nil
}

// checkRAMForOffload 检查 MoE offload 模式下 RAM 是否充足
// 如果 RAM 紧张，给用户提示建议用更小模型
func checkRAMForOffload(profile *model.DeployProfile, hw *hardware.HardwareProbe) {
	// MoE offload: 90% 模型在 RAM，10% 在 GPU
	ramNeededGB := profile.Size_GB * 0.9
	ramAvailGB := float64(hw.RAM.Total_MB) / 1024.0
	ramFreeGB := float64(hw.RAM.Free_MB) / 1024.0

	// 需要至少 3GB 余量给系统和 KV cache
	if ramNeededGB > ramAvailGB-3 {
		fmt.Printf("      ⚠️  RAM 可能不足：模型需要 %.1f GB，总共 %.1f GB\n", ramNeededGB, ramAvailGB)
		fmt.Println("      建议：使用更小的模型（如 8B）以获得更好的性能")
	} else if ramNeededGB > ramFreeGB {
		fmt.Printf("      ⚠️  RAM 当前可用 %.1f GB，模型需要 %.1f GB\n", ramFreeGB, ramNeededGB)
		fmt.Println("      提示：关闭其他应用可提升性能，或使用更小的模型")
	}
}

// estimateKVCacheGB estimates KV cache size in GB.
// iso3: 3-bit = 0.375 bytes/element (K+V 各 0.375)，总计 0.75 bytes/element
// q8_0+q4_0: K=1 byte + V=0.5 byte = 1.5 bytes/element
// Most modern models use GQA with ~4 kv_heads (not full 32), head_dim ~128
func estimateKVCacheGB(ctxSize, layers int, hasIsoQuant bool) float64 {
	// GQA: kv_heads typically 4 for 8B-32B models
	kvBytesPerElement := 1.5
	if hasIsoQuant {
		kvBytesPerElement = 0.75 // iso3 K + iso3 V = 0.375 + 0.375
	}
	bytes := float64(layers*ctxSize*128*4) * kvBytesPerElement
	return bytes / (1024 * 1024 * 1024)
}

// dynamicCtxSize 计算最优 ctx（避免循环依赖，本地实现）
func dynamicCtxSize(vramMB int, modelSizeGB float64, layers, ctxOverride int, hasIsoQuant bool) int {
	if ctxOverride > 0 {
		return ctxOverride
	}
	if vramMB == 0 {
		return 4096
	}
	// iso3 有 ~600MB 固定解码 buffer 开销（实测验证）
	isoOverheadMB := 0.0
	if hasIsoQuant {
		isoOverheadMB = 600
	}
	vramAvailMB := float64(vramMB) - 300 - modelSizeGB*1024 - isoOverheadMB
	kvBytesPerElement := 1.5
	if hasIsoQuant {
		kvBytesPerElement = 0.75
	}
	kvPerTokenMB := float64(layers) * 128 * 4 * kvBytesPerElement / (1024 * 1024)
	maxTokens := int(vramAvailMB * 0.8 / kvPerTokenMB)
	return roundCtxPow2(maxTokens)
}

func roundCtxPow2(n int) int {
	powers := []int{524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096}
	for _, p := range powers {
		if n >= p {
			return p
		}
	}
	return 4096
}
