package app

import (
	"fmt"
	"os"
	"path/filepath"

	"mcpmydocs/internal/embedder"
	"mcpmydocs/internal/paths"
	"mcpmydocs/internal/store"
)

// App holds the core components of the application.
type App struct {
	Store    *store.Store
	Embedder *embedder.Embedder
}

// Config holds configuration for initializing the App.
type Config struct {
	DBPath          string
	ModelPath       string
	OnnxLibraryPath string
}

// New initializes the application components.
func New(cfg Config) (*App, error) {
	// Initialize store
	st, err := store.New(cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Initialize embedder
	emb, err := embedder.New(cfg.ModelPath, cfg.OnnxLibraryPath)
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	return &App{
		Store:    st,
		Embedder: emb,
	}, nil
}

// Close releases resources.
func (a *App) Close() error {
	var errs []error
	if err := a.Store.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := a.Embedder.Close(); err != nil {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing app: %v", errs)
	}
	return nil
}

// DefaultPaths returns the default paths for the application.
func DefaultPaths(onnxLibOverride string) (Config, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return Config{}, fmt.Errorf("failed to get current working directory: %w", err)
	}

	dbPath := filepath.Join(cwd, "mcpmydocs.db")
	
	modelPath, err := paths.ResolveModelPath()
	if err != nil {
		return Config{}, err
	}

	onnxLibPath, err := paths.ResolveONNXLibraryPath(onnxLibOverride)
	if err != nil {
		return Config{}, err
	}

	return Config{
		DBPath:          dbPath,
		ModelPath:       modelPath,
		OnnxLibraryPath: onnxLibPath,
	}, nil
}
