package engine

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/kaiwu-ai/kaiwu/internal/config"
	"github.com/kaiwu-ai/kaiwu/internal/download"
	"github.com/kaiwu-ai/kaiwu/internal/hardware"
)

const releaseTag = "b8864"

// selectBinary returns the local binary name based on hardware
func selectBinary(hw *hardware.HardwareProbe) string {
	gpu := hw.PrimaryGPU()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}

	if gpu == nil {
		return "llama-server" + ext
	}

	if strings.Contains(strings.ToLower(gpu.Name), "nvidia") {
		return "llama-server-cuda" + ext
	}

	return "llama-server-vulkan" + ext
}

// downloadURL returns the correct release asset URL
func downloadURL(hw *hardware.HardwareProbe) string {
	base := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s", releaseTag)
	gpu := hw.PrimaryGPU()

	if runtime.GOOS == "windows" {
		if gpu != nil && strings.Contains(strings.ToLower(gpu.Name), "nvidia") {
			return fmt.Sprintf("%s/llama-%s-bin-win-cuda-12.4-x64.zip", base, releaseTag)
		}
		if gpu != nil {
			return fmt.Sprintf("%s/llama-%s-bin-win-vulkan-x64.zip", base, releaseTag)
		}
		return fmt.Sprintf("%s/llama-%s-bin-win-cpu-x64.zip", base, releaseTag)
	}

	// Linux
	if gpu != nil && strings.Contains(strings.ToLower(gpu.Name), "nvidia") {
		return fmt.Sprintf("%s/llama-%s-bin-ubuntu-x64.tar.gz", base, releaseTag)
	}
	if gpu != nil {
		return fmt.Sprintf("%s/llama-%s-bin-ubuntu-vulkan-x64.tar.gz", base, releaseTag)
	}
	return fmt.Sprintf("%s/llama-%s-bin-ubuntu-x64.tar.gz", base, releaseTag)
}

// EnsureBinary ensures the correct llama-server binary is available
func EnsureBinary(hw *hardware.HardwareProbe) (string, error) {
	binaryName := selectBinary(hw)
	binaryPath := filepath.Join(config.BinDir(), binaryName)

	if _, err := os.Stat(binaryPath); err == nil {
		return binaryPath, nil
	}

	url := downloadURL(hw)
	fmt.Printf("      Downloading: %s\n", filepath.Base(url))

	// Download archive
	archivePath := filepath.Join(config.BinDir(), filepath.Base(url))
	if err := download.DownloadFile(url, archivePath, true); err != nil {
		return "", fmt.Errorf("download failed: %w", err)
	}

	// Extract llama-server from archive
	fmt.Printf("      Extracting llama-server...\n")
	if err := extractLlamaServer(archivePath, binaryPath); err != nil {
		os.Remove(archivePath)
		return "", fmt.Errorf("extraction failed: %w", err)
	}

	// Cleanup archive
	os.Remove(archivePath)

	// Set executable permission on Linux
	if runtime.GOOS == "linux" {
		os.Chmod(binaryPath, 0755)
	}

	return binaryPath, nil
}

// extractLlamaServer extracts llama-server binary from zip or tar.gz
func extractLlamaServer(archivePath, destPath string) error {
	if strings.HasSuffix(archivePath, ".zip") {
		return extractFromZip(archivePath, destPath)
	}
	if strings.HasSuffix(archivePath, ".tar.gz") || strings.HasSuffix(archivePath, ".tgz") {
		return extractFromTarGz(archivePath, destPath)
	}
	return fmt.Errorf("unsupported archive format: %s", archivePath)
}

func extractFromZip(archivePath, destPath string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	target := "llama-server"
	if runtime.GOOS == "windows" {
		target = "llama-server.exe"
	}

	// Also extract CUDA runtime DLLs if present
	dllsToExtract := map[string]bool{
		"cublas64_12.dll":    true,
		"cublasLt64_12.dll":  true,
		"cudart64_12.dll":    true,
		"ggml-cuda.dll":      true,
	}

	binDir := filepath.Dir(destPath)
	found := false

	for _, f := range r.File {
		name := filepath.Base(f.Name)

		if name == target {
			if err := extractZipFile(f, destPath); err != nil {
				return err
			}
			found = true
		} else if dllsToExtract[name] {
			dllPath := filepath.Join(binDir, name)
			if err := extractZipFile(f, dllPath); err != nil {
				fmt.Printf("      Warning: failed to extract %s: %v\n", name, err)
			}
		}
	}

	if !found {
		return fmt.Errorf("llama-server not found in archive")
	}
	return nil
}

func extractZipFile(f *zip.File, destPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, rc)
	return err
}

func extractFromTarGz(archivePath, destPath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	target := "llama-server"

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		name := filepath.Base(hdr.Name)
		if name == target {
			out, err := os.Create(destPath)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
			os.Chmod(destPath, 0755)
			return nil
		}
	}

	return fmt.Errorf("llama-server not found in archive")
}

// ValidateCUDAVersion checks for Blackwell + CUDA 13.x incompatibility
func ValidateCUDAVersion(hw *hardware.HardwareProbe) error {
	gpu := hw.PrimaryGPU()
	if gpu == nil || !gpu.IsBlackwell {
		return nil
	}

	if strings.HasPrefix(gpu.CUDADriver, "13.") {
		fmt.Printf("      Warning: RTX 50 series with CUDA %s detected\n", gpu.CUDADriver)
		fmt.Println("      Kaiwu will use CUDA 12.4 binary for stability.")
	}

	return nil
}
