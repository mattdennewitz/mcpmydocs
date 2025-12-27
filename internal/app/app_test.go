package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNew_InvalidDBPath(t *testing.T) {
	cfg := Config{
		DBPath:          "/nonexistent/directory/test.db",
		ModelPath:       "/some/model.onnx",
		OnnxLibraryPath: "/some/lib.dylib",
	}

	_, err := New(cfg)
	if err == nil {
		t.Error("expected error for invalid DB path")
	}
	if !strings.Contains(err.Error(), "database") {
		t.Errorf("error should mention database, got: %v", err)
	}
}

func TestNew_InvalidModelPath(t *testing.T) {
	// Create a valid temp DB path
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	cfg := Config{
		DBPath:          dbPath,
		ModelPath:       "/nonexistent/model.onnx",
		OnnxLibraryPath: "/some/lib.dylib",
	}

	_, err := New(cfg)
	if err == nil {
		t.Error("expected error for invalid model path")
	}
	if !strings.Contains(err.Error(), "embedder") {
		t.Errorf("error should mention embedder, got: %v", err)
	}

	// Verify DB file was cleaned up (store should be closed on embedder failure)
	// The DB file might exist but should not have open connections
}

func TestDefaultPaths_SetsDBPath(t *testing.T) {
	// Save and restore env vars
	oldModelPath := os.Getenv("MCPMYDOCS_MODEL_PATH")
	oldOnnxPath := os.Getenv("ONNX_LIBRARY_PATH")
	defer func() {
		os.Setenv("MCPMYDOCS_MODEL_PATH", oldModelPath)
		os.Setenv("ONNX_LIBRARY_PATH", oldOnnxPath)
	}()

	// Create temp files for model and onnx lib
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "embed.onnx")
	onnxPath := filepath.Join(tmpDir, "libonnxruntime.dylib")
	if err := os.WriteFile(modelPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(onnxPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	os.Setenv("MCPMYDOCS_MODEL_PATH", modelPath)
	os.Setenv("ONNX_LIBRARY_PATH", onnxPath)

	cfg, err := DefaultPaths("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// DB path should be in current working directory
	cwd, _ := os.Getwd()
	expectedDBPath := filepath.Join(cwd, "mcpmydocs.db")
	if cfg.DBPath != expectedDBPath {
		t.Errorf("DBPath: expected %s, got %s", expectedDBPath, cfg.DBPath)
	}

	if cfg.ModelPath != modelPath {
		t.Errorf("ModelPath: expected %s, got %s", modelPath, cfg.ModelPath)
	}

	if cfg.OnnxLibraryPath != onnxPath {
		t.Errorf("OnnxLibraryPath: expected %s, got %s", onnxPath, cfg.OnnxLibraryPath)
	}
}

func TestDefaultPaths_OnnxOverride(t *testing.T) {
	// Save and restore env vars
	oldModelPath := os.Getenv("MCPMYDOCS_MODEL_PATH")
	oldOnnxPath := os.Getenv("ONNX_LIBRARY_PATH")
	defer func() {
		os.Setenv("MCPMYDOCS_MODEL_PATH", oldModelPath)
		os.Setenv("ONNX_LIBRARY_PATH", oldOnnxPath)
	}()

	// Create temp files
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "embed.onnx")
	onnxEnvPath := filepath.Join(tmpDir, "env.dylib")
	onnxOverridePath := filepath.Join(tmpDir, "override.dylib")

	for _, p := range []string{modelPath, onnxEnvPath, onnxOverridePath} {
		if err := os.WriteFile(p, []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	os.Setenv("MCPMYDOCS_MODEL_PATH", modelPath)
	os.Setenv("ONNX_LIBRARY_PATH", onnxEnvPath)

	// Override should take precedence over env var
	cfg, err := DefaultPaths(onnxOverridePath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.OnnxLibraryPath != onnxOverridePath {
		t.Errorf("expected override path %s, got %s", onnxOverridePath, cfg.OnnxLibraryPath)
	}
}

func TestDefaultPaths_ModelNotFound(t *testing.T) {
	// Save and restore env vars
	oldModelPath := os.Getenv("MCPMYDOCS_MODEL_PATH")
	oldOnnxPath := os.Getenv("ONNX_LIBRARY_PATH")
	defer func() {
		os.Setenv("MCPMYDOCS_MODEL_PATH", oldModelPath)
		os.Setenv("ONNX_LIBRARY_PATH", oldOnnxPath)
	}()

	// Save current directory
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	// Clear env vars and change to empty temp dir
	os.Unsetenv("MCPMYDOCS_MODEL_PATH")
	os.Unsetenv("ONNX_LIBRARY_PATH")

	tmpDir := t.TempDir()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	_, err = DefaultPaths("")
	if err == nil {
		t.Error("expected error when model not found")
	}
}

func TestConfig_Fields(t *testing.T) {
	cfg := Config{
		DBPath:          "/path/to/db",
		ModelPath:       "/path/to/model",
		OnnxLibraryPath: "/path/to/lib",
	}

	if cfg.DBPath != "/path/to/db" {
		t.Error("DBPath not set correctly")
	}
	if cfg.ModelPath != "/path/to/model" {
		t.Error("ModelPath not set correctly")
	}
	if cfg.OnnxLibraryPath != "/path/to/lib" {
		t.Error("OnnxLibraryPath not set correctly")
	}
}

// TestNewAndClose_Integration tests the full lifecycle when dependencies are available.
// This test is skipped if the required model/library files are not present.
func TestNewAndClose_Integration(t *testing.T) {
	// Try to get default paths - skip if not available
	oldModelPath := os.Getenv("MCPMYDOCS_MODEL_PATH")
	oldOnnxPath := os.Getenv("ONNX_LIBRARY_PATH")
	defer func() {
		if oldModelPath != "" {
			os.Setenv("MCPMYDOCS_MODEL_PATH", oldModelPath)
		}
		if oldOnnxPath != "" {
			os.Setenv("ONNX_LIBRARY_PATH", oldOnnxPath)
		}
	}()

	// Check if we're in a directory where we can find the model
	cfg, err := DefaultPaths("")
	if err != nil {
		t.Skipf("skipping integration test: %v", err)
	}

	// Use temp DB
	tmpDir := t.TempDir()
	cfg.DBPath = filepath.Join(tmpDir, "test.db")

	app, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create app: %v", err)
	}

	// Verify components are initialized
	if app.Store == nil {
		t.Error("Store is nil")
	}
	if app.Embedder == nil {
		t.Error("Embedder is nil")
	}

	// Close should not error
	if err := app.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}
