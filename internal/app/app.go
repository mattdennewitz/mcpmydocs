package app

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/mattdennewitz/mcpmydocs/internal/embedder"
	"github.com/mattdennewitz/mcpmydocs/internal/logger"
	"github.com/mattdennewitz/mcpmydocs/internal/paths"
	"github.com/mattdennewitz/mcpmydocs/internal/reranker"
	"github.com/mattdennewitz/mcpmydocs/internal/store"
)

// App holds the core components of the application.
type App struct {
	Store    *store.Store
	Embedder *embedder.Embedder
	Reranker *reranker.Reranker // nil if reranker model not available
}

// Config holds configuration for initializing the App.
type Config struct {
	DBPath            string
	ModelPath         string
	RerankerModelPath string // optional - empty string means no reranking
	OnnxLibraryPath   string
	ReadOnly          bool // open database in read-only mode to avoid lock conflicts
}

// New initializes the application components.
func New(cfg Config) (*App, error) {
	// Initialize store
	var st *store.Store
	var err error
	if cfg.ReadOnly {
		st, err = store.NewReadOnly(cfg.DBPath)
	} else {
		st, err = store.New(cfg.DBPath)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Initialize embedder
	emb, err := embedder.New(cfg.ModelPath, cfg.OnnxLibraryPath)
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	// Initialize reranker (optional - graceful if missing)
	var rr *reranker.Reranker
	if cfg.RerankerModelPath != "" {
		rr, err = reranker.New(cfg.RerankerModelPath, cfg.OnnxLibraryPath)
		if err != nil {
			logger.Warn("reranker initialization failed, continuing without reranking", "error", err)
			rr = nil
		}
	}

	return &App{
		Store:    st,
		Embedder: emb,
		Reranker: rr,
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
	if a.Reranker != nil {
		if err := a.Reranker.Close(); err != nil {
			errs = append(errs, err)
		}
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

	// Reranker model is optional
	rerankerPath := paths.ResolveRerankerModelPath()

	return Config{
		DBPath:            dbPath,
		ModelPath:         modelPath,
		RerankerModelPath: rerankerPath,
		OnnxLibraryPath:   onnxLibPath,
	}, nil
}
