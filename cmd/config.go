package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// resolveONNXLibraryPath attempts to find the ONNX Runtime shared library
// in the following order:
// 1. ONNX_LIBRARY_PATH environment variable
// 2. "lib" directory next to the running executable (self-contained deployment)
// 3. Standard system locations (Homebrew, /usr/local, etc.)
func resolveONNXLibraryPath() (string, error) {
	// 1. Environment variable
	if envPath := os.Getenv("ONNX_LIBRARY_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath, nil
		}
		// If set but invalid, warn but continue? Or fail? 
		// For now, let's treat it as a suggestion that failed.
	}

	// Determine library name based on OS
	var libName string
	switch runtime.GOOS {
	case "darwin":
		libName = "libonnxruntime.dylib"
	case "linux":
		libName = "libonnxruntime.so"
	case "windows":
		libName = "onnxruntime.dll"
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	// 2. Sidecar "lib" directory (relative to executable)
	exePath, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exePath)
		sidecarPath := filepath.Join(exeDir, "lib", libName)
		if _, err := os.Stat(sidecarPath); err == nil {
			return sidecarPath, nil
		}
		
		// Also check root of executable for Windows style or simpler structure
		rootPath := filepath.Join(exeDir, libName)
		if _, err := os.Stat(rootPath); err == nil {
			return rootPath, nil
		}
	}

	// 3. System paths (Fallback)
	systemPaths := []string{}
	switch runtime.GOOS {
	case "darwin":
		systemPaths = []string{
			"/opt/homebrew/lib/" + libName,
			"/usr/local/lib/" + libName,
		}
	case "linux":
		systemPaths = []string{
			"/usr/lib/" + libName,
			"/usr/local/lib/" + libName,
		}
	}

	for _, p := range systemPaths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	return "", fmt.Errorf("ONNX runtime library (%s) not found. Set ONNX_LIBRARY_PATH or place it in a 'lib' folder next to the executable", libName)
}
