package optimizer

import (
	"github.com/kaiwu-ai/kaiwu/internal/hardware"
	"github.com/kaiwu-ai/kaiwu/internal/model"
)

// StartingParams holds initial parameter recommendations
type StartingParams struct {
	Mode       string   // "moe_offload" or "full_gpu"
	OTFlags    string   // MoE offload flags
	BatchSizes []int    // Batch sizes to test
	UBatchSize int      // Micro-batch size
	Threads    int      // CPU threads
	CtxSize    int      // Context size
	KVCacheK   string   // KV cache K quantization
	KVCacheV   string   // KV cache V quantization
}

// DeriveStartingParams applies the three parameter rules from spec
func DeriveStartingParams(hw *hardware.HardwareProbe, profile *model.DeployProfile) StartingParams {
	gpu := hw.PrimaryGPU()
	vramGB := 0
	if gpu != nil {
		vramGB = gpu.VRAM_MB / 1024
	}

	isMoE := profile.Arch == "moe"
	modelFits := profile.Size_GB < float64(vramGB)*0.85

	// 动态计算 ctx
	ctxSize := DynamicCtxSize(hw, profile)

	// IsoQuant: 统一用 iso3
	kvK := "q8_0"
	kvV := kvCacheVByVRAM(vramGB)
	if profile.HasIsoQuant {
		kvK = "iso3"
		kvV = "iso3"
	}

	// Rule 1: MoE + doesn't fit → expert offload to CPU
	if isMoE && !modelFits {
		return StartingParams{
			Mode:       "moe_offload",
			OTFlags:    profile.OTFlags,
			BatchSizes: []int{1024, 2048, 4096},
			UBatchSize: 512,
			Threads:    hw.CPU.Cores * 2 / 3,
			CtxSize:    ctxSize,
			KVCacheK:   kvK,
			KVCacheV:   kvV,
		}
	}

	// Rule 2: Full GPU
	return StartingParams{
		Mode:       "full_gpu",
		BatchSizes: []int{256, 512, 1024},
		UBatchSize: 128,
		Threads:    hw.CPU.Cores / 2,
		CtxSize:    ctxSize,
		KVCacheK:   kvK,
		KVCacheV:   kvV,
	}
}

// DynamicCtxSize 根据实际硬件动态计算最优上下文大小
// 目标：全 VRAM（无 RAM offload），速度 > LM Studio，上下文更长
//
// 参数：
//   - hw: 硬件探测结果（VRAM, RAM）
//   - profile: 模型配置（Size_GB, Layers, HasIsoQuant）
//
// 返回：最优 ctx 大小（2的幂：4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288）
//
// 计算逻辑：
//   1. 可用 VRAM = 总 VRAM - 系统保留(300MB) - 模型大小
//   2. KV cache 单 token 大小 = layers × 128 × 4 × kvBytesPerElement / (1024×1024) MB
//      - 128: head_dim（固定）
//      - 4: kv_heads（GQA 典型值）
//      - kvBytesPerElement: iso3=0.75 (K+V合计), q8_0+q4_0=1.5
//   3. 最大 tokens = 可用 VRAM × 0.8 / KV cache 单 token 大小
//      - 0.8: 留 20% 余量，避免 OOM
//   4. 向下取最近的 2 的幂
func DynamicCtxSize(hw *hardware.HardwareProbe, profile *model.DeployProfile) int {
	// 微调模式：用户手动指定了 ctx 大小
	if profile.CtxOverride > 0 {
		return profile.CtxOverride
	}

	gpu := hw.PrimaryGPU()
	if gpu == nil {
		return 4096 // CPU-only fallback
	}

	// 1. 计算可用 VRAM（MB）
	// iso3 有 ~600MB 固定解码 buffer 开销（实测 4090 + 35B MoE 验证）
	isoOverheadMB := 0.0
	if profile.HasIsoQuant {
		isoOverheadMB = 600
	}
	vramAvailMB := float64(gpu.VRAM_MB) - 300 - profile.Size_GB*1024 - isoOverheadMB

	// 2. KV cache 压缩系数（K+V 合计）
	// iso3: 3-bit = 0.375 bytes/element，K+V = 0.375*2 = 0.75（压缩比 2x vs q8_0+q4_0）
	// q8_0+q4_0: K=1 byte + V=0.5 byte = 1.5 bytes/element
	kvBytesPerElement := 1.5
	if profile.HasIsoQuant {
		kvBytesPerElement = 0.75 // iso3 K(0.375) + V(0.375)
	}

	// 3. 计算单个 token 的 KV cache 大小（MB）
	// Formula: layers × 128 × 4 × kvBytesPerElement / (1024 × 1024)
	kvPerTokenMB := float64(profile.Layers) * 128 * 4 * kvBytesPerElement / (1024 * 1024)

	// 4. 计算最大 tokens（留 20% 余量）
	maxTokens := int(vramAvailMB * 0.8 / kvPerTokenMB)

	// 5. 向下取最近的 2 的幂
	return roundToNearestPowerOf2(maxTokens)
}

// roundToNearestPowerOf2 向下取最近的 2 的幂
// 支持的值：4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288
// iso3 压缩后 MoE 模型可达 512K 上下文
func roundToNearestPowerOf2(n int) int {
	powers := []int{524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096}
	for _, p := range powers {
		if n >= p {
			return p
		}
	}
	return 4096 // 最小 4K
}

// KV cache V quantization by VRAM (K is always q8_0)
func kvCacheVByVRAM(vramGB int) string {
	if vramGB > 16 {
		return "f16" // High VRAM: preserve quality
	}
	return "q4_0" // Low/Mid VRAM: 非对称压缩，省显存给更大ctx
}
