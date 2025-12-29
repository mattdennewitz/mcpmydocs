package paths

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// ResolveONNXLibraryPath attempts to find the ONNX Runtime shared library.
func ResolveONNXLibraryPath(userProvidedPath string) (string, error) {
	// 1. CLI flag
	if userProvidedPath != "" {
		if _, err := os.Stat(userProvidedPath); err == nil {
			return userProvidedPath, nil
		}
		return "", fmt.Errorf("specified ONNX library path not found: %s", userProvidedPath)
	}

	// 2. Environment variable
	if envPath := os.Getenv("ONNX_LIBRARY_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath, nil
		}
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

	// 3. Sidecar "lib" directory (relative to executable)
	exePath, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exePath)
		sidecarPath := filepath.Join(exeDir, "lib", libName)
		if _, err := os.Stat(sidecarPath); err == nil {
			return sidecarPath, nil
		}
		
		// Also check root of executable
		rootPath := filepath.Join(exeDir, libName)
		if _, err := os.Stat(rootPath); err == nil {
			return rootPath, nil
		}
	}

	// 4. System paths (Fallback)
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

// ResolveModelPath attempts to find the embedding model file.
func ResolveModelPath() (string, error) {
	// 1. Environment variable
	if envPath := os.Getenv("MCPMYDOCS_MODEL_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath, nil
		}
	}

	// 2. Executable relative (Deployment)
	exe, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exe)
		
		// Check "assets/models/embed.onnx" next to binary
		path := filepath.Join(exeDir, "assets", "models", "embed.onnx")
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}

		// Check "../assets/models/embed.onnx" (if binary is in bin/)
		path = filepath.Join(exeDir, "..", "assets", "models", "embed.onnx")
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	// 3. CWD (Development fallback)
	cwd, err := os.Getwd()
	if err == nil {
		path := filepath.Join(cwd, "assets", "models", "embed.onnx")
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("model file 'embed.onnx' not found. Set MCPMYDOCS_MODEL_PATH or ensure 'assets' directory is near the executable")
}

// ResolveRerankerModelPath attempts to find the reranker model file.
// Returns empty string if not found (reranker is optional).
func ResolveRerankerModelPath() string {
	// 1. Environment variable
	if envPath := os.Getenv("MCPMYDOCS_RERANKER_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
	}

	// 2. Executable relative (Deployment)
	exe, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exe)

		// Check "assets/models/rerank.onnx" next to binary
		path := filepath.Join(exeDir, "assets", "models", "rerank.onnx")
		if _, err := os.Stat(path); err == nil {
			return path
		}

		// Check "../assets/models/rerank.onnx" (if binary is in bin/)
		path = filepath.Join(exeDir, "..", "assets", "models", "rerank.onnx")
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// 3. CWD (Development fallback)
	cwd, err := os.Getwd()
	if err == nil {
		path := filepath.Join(cwd, "assets", "models", "rerank.onnx")
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// Reranker is optional - return empty string if not found
	return ""
}
