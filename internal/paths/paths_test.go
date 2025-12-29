package paths

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestResolveONNXLibraryPath_UserProvided(t *testing.T) {
	// Create a temp file to simulate the library
	tmpDir := t.TempDir()
	fakePath := filepath.Join(tmpDir, "fake.dylib")
	if err := os.WriteFile(fakePath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Run("valid user provided path", func(t *testing.T) {
		result, err := ResolveONNXLibraryPath(fakePath)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != fakePath {
			t.Errorf("expected %s, got %s", fakePath, result)
		}
	})

	t.Run("invalid user provided path", func(t *testing.T) {
		_, err := ResolveONNXLibraryPath("/nonexistent/path/lib.so")
		if err == nil {
			t.Error("expected error for nonexistent path")
		}
	})
}

func TestResolveONNXLibraryPath_EnvVar(t *testing.T) {
	// Create a temp file
	tmpDir := t.TempDir()
	fakePath := filepath.Join(tmpDir, "onnx.dylib")
	if err := os.WriteFile(fakePath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	// Save and restore env var
	oldEnv := os.Getenv("ONNX_LIBRARY_PATH")
	defer os.Setenv("ONNX_LIBRARY_PATH", oldEnv)

	t.Run("valid env var path", func(t *testing.T) {
		os.Setenv("ONNX_LIBRARY_PATH", fakePath)

		result, err := ResolveONNXLibraryPath("")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != fakePath {
			t.Errorf("expected %s, got %s", fakePath, result)
		}
	})

	t.Run("invalid env var path falls through", func(t *testing.T) {
		os.Setenv("ONNX_LIBRARY_PATH", "/nonexistent/path")

		// Should fall through to system paths (may or may not find it)
		// We just verify it doesn't return the invalid env path
		result, _ := ResolveONNXLibraryPath("")
		if result == "/nonexistent/path" {
			t.Error("should not return invalid env path")
		}
	})
}

func TestResolveONNXLibraryPath_UserProvidedTakesPrecedence(t *testing.T) {
	tmpDir := t.TempDir()

	// Create two fake libraries
	userPath := filepath.Join(tmpDir, "user.dylib")
	envPath := filepath.Join(tmpDir, "env.dylib")
	if err := os.WriteFile(userPath, []byte("user"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(envPath, []byte("env"), 0644); err != nil {
		t.Fatal(err)
	}

	// Set env var
	oldEnv := os.Getenv("ONNX_LIBRARY_PATH")
	defer os.Setenv("ONNX_LIBRARY_PATH", oldEnv)
	os.Setenv("ONNX_LIBRARY_PATH", envPath)

	// User provided should take precedence
	result, err := ResolveONNXLibraryPath(userPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != userPath {
		t.Errorf("expected user path %s, got %s", userPath, result)
	}
}

func TestResolveONNXLibraryPath_LibraryName(t *testing.T) {
	// This test verifies the correct library name is used per OS
	// We can't easily test cross-platform, but we verify the current OS behavior

	var expectedSuffix string
	switch runtime.GOOS {
	case "darwin":
		expectedSuffix = "libonnxruntime.dylib"
	case "linux":
		expectedSuffix = "libonnxruntime.so"
	case "windows":
		expectedSuffix = "onnxruntime.dll"
	default:
		t.Skip("unsupported OS for this test")
	}

	// Clear env to force system path search
	oldEnv := os.Getenv("ONNX_LIBRARY_PATH")
	defer os.Setenv("ONNX_LIBRARY_PATH", oldEnv)
	os.Unsetenv("ONNX_LIBRARY_PATH")

	// The error message should contain the correct library name
	_, err := ResolveONNXLibraryPath("")
	if err != nil {
		// Check error message contains correct library name
		if !strings.Contains(err.Error(), expectedSuffix) {
			t.Errorf("error should mention %s, got: %v", expectedSuffix, err)
		}
	}
	// If no error, it found the library (which is fine)
}

func TestResolveModelPath_EnvVar(t *testing.T) {
	tmpDir := t.TempDir()
	fakePath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(fakePath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	oldEnv := os.Getenv("MCPMYDOCS_MODEL_PATH")
	defer os.Setenv("MCPMYDOCS_MODEL_PATH", oldEnv)

	t.Run("valid env var", func(t *testing.T) {
		os.Setenv("MCPMYDOCS_MODEL_PATH", fakePath)

		result, err := ResolveModelPath()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != fakePath {
			t.Errorf("expected %s, got %s", fakePath, result)
		}
	})

	t.Run("invalid env var falls through", func(t *testing.T) {
		os.Setenv("MCPMYDOCS_MODEL_PATH", "/nonexistent/model.onnx")

		result, _ := ResolveModelPath()
		if result == "/nonexistent/model.onnx" {
			t.Error("should not return invalid env path")
		}
	})
}

func TestResolveModelPath_CWDFallback(t *testing.T) {
	// Save current dir
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	// Clear env var
	oldEnv := os.Getenv("MCPMYDOCS_MODEL_PATH")
	defer os.Setenv("MCPMYDOCS_MODEL_PATH", oldEnv)
	os.Unsetenv("MCPMYDOCS_MODEL_PATH")

	// Create temp dir with assets/models/embed.onnx
	tmpDir := t.TempDir()
	modelDir := filepath.Join(tmpDir, "assets", "models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(modelDir, "embed.onnx")
	if err := os.WriteFile(modelPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	// Change to temp dir
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	result, err := ResolveModelPath()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Result should be the model path (may be absolute or relative)
	if filepath.Base(result) != "embed.onnx" {
		t.Errorf("expected embed.onnx, got %s", result)
	}
}

func TestResolveModelPath_NotFound(t *testing.T) {
	// Save current dir
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	// Clear env var
	oldEnv := os.Getenv("MCPMYDOCS_MODEL_PATH")
	defer os.Setenv("MCPMYDOCS_MODEL_PATH", oldEnv)
	os.Unsetenv("MCPMYDOCS_MODEL_PATH")

	// Change to empty temp dir
	tmpDir := t.TempDir()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	_, err = ResolveModelPath()
	if err == nil {
		t.Error("expected error when model not found")
	}
	if !strings.Contains(err.Error(), "embed.onnx") {
		t.Errorf("error should mention embed.onnx: %v", err)
	}
}

func TestResolveRerankerModelPath_EnvVar(t *testing.T) {
	tmpDir := t.TempDir()
	fakePath := filepath.Join(tmpDir, "rerank.onnx")
	if err := os.WriteFile(fakePath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	oldEnv := os.Getenv("MCPMYDOCS_RERANKER_PATH")
	defer os.Setenv("MCPMYDOCS_RERANKER_PATH", oldEnv)

	t.Run("valid env var", func(t *testing.T) {
		os.Setenv("MCPMYDOCS_RERANKER_PATH", fakePath)

		result := ResolveRerankerModelPath()
		if result != fakePath {
			t.Errorf("expected %s, got %s", fakePath, result)
		}
	})

	t.Run("invalid env var returns empty", func(t *testing.T) {
		os.Setenv("MCPMYDOCS_RERANKER_PATH", "/nonexistent/rerank.onnx")

		result := ResolveRerankerModelPath()
		// Should return empty (or find it elsewhere) - reranker is optional
		if result == "/nonexistent/rerank.onnx" {
			t.Error("should not return invalid env path")
		}
	})
}

func TestResolveRerankerModelPath_CWDFallback(t *testing.T) {
	// Save current dir
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	// Clear env var
	oldEnv := os.Getenv("MCPMYDOCS_RERANKER_PATH")
	defer os.Setenv("MCPMYDOCS_RERANKER_PATH", oldEnv)
	os.Unsetenv("MCPMYDOCS_RERANKER_PATH")

	// Create temp dir with assets/models/rerank.onnx
	tmpDir := t.TempDir()
	modelDir := filepath.Join(tmpDir, "assets", "models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(modelDir, "rerank.onnx")
	if err := os.WriteFile(modelPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	// Change to temp dir
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	result := ResolveRerankerModelPath()
	if filepath.Base(result) != "rerank.onnx" {
		t.Errorf("expected rerank.onnx, got %s", result)
	}
}

func TestResolveRerankerModelPath_NotFound(t *testing.T) {
	// Save current dir
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	// Clear env var
	oldEnv := os.Getenv("MCPMYDOCS_RERANKER_PATH")
	defer os.Setenv("MCPMYDOCS_RERANKER_PATH", oldEnv)
	os.Unsetenv("MCPMYDOCS_RERANKER_PATH")

	// Change to empty temp dir
	tmpDir := t.TempDir()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	result := ResolveRerankerModelPath()
	// Reranker is optional - should return empty string, not error
	if result != "" {
		t.Errorf("expected empty string when reranker not found, got %s", result)
	}
}
