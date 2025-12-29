package cmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"

	"github.com/mattdennewitz/mcpmydocs/internal/app"
	"github.com/mattdennewitz/mcpmydocs/internal/chunker"
	"github.com/mattdennewitz/mcpmydocs/internal/logger"
	"github.com/mattdennewitz/mcpmydocs/internal/store"
)

// NewIndexCmd creates the index command.
func NewIndexCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "index [directory]",
		Short: "Index a directory of markdown files",
		Args:  cobra.MinimumNArgs(1),
		RunE:  runIndex,
	}
}

func runIndex(cmd *cobra.Command, args []string) error {
	dir := args[0]

	absDir, err := resolveDirectory(dir)
	if err != nil {
		return err
	}

	application, cfg, err := initializeApp()
	if err != nil {
		return err
	}
	defer application.Close()

	logger.Info("starting indexing", "directory", absDir, "database", cfg.DBPath)
	logger.Debug("configuration", "model", cfg.ModelPath, "onnxLib", cfg.OnnxLibraryPath)

	files := collectMarkdownFiles(absDir)
	stats := processFiles(absDir, files, application.Store, application.Embedder, chunker.New())

	fmt.Printf("\r\033[K")
	fmt.Printf("Indexing complete!\n")
	fmt.Printf("  Indexed: %d files\n", stats.indexed.Load())
	fmt.Printf("  Skipped: %d unchanged files\n", stats.skipped.Load())

	return nil
}

func resolveDirectory(dir string) (string, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve directory: %w", err)
	}

	info, err := os.Stat(absDir)
	if err != nil {
		return "", fmt.Errorf("directory not found: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", absDir)
	}

	return absDir, nil
}

func initializeApp() (*app.App, app.Config, error) {
	cfg, err := app.DefaultPaths(OnnxLibraryPath)
	if err != nil {
		return nil, app.Config{}, fmt.Errorf("failed to resolve paths: %w", err)
	}

	application, err := app.New(cfg)
	if err != nil {
		return nil, app.Config{}, fmt.Errorf("failed to initialize application: %w", err)
	}

	return application, cfg, nil
}

func collectMarkdownFiles(absDir string) []string {
	var files []string
	_ = filepath.WalkDir(absDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			files = append(files, path)
		}
		return nil
	})
	return files
}

type indexStats struct {
	processed atomic.Int32
	indexed   atomic.Int32
	skipped   atomic.Int32
}

func processFiles(absDir string, files []string, st *store.Store, emb interface {
	Embed([]string) ([][]float32, error)
}, ch *chunker.Chunker) *indexStats {
	stats := &indexStats{}
	var printMu sync.Mutex
	totalFiles := len(files)

	g, ctx := errgroup.WithContext(context.Background())
	g.SetLimit(runtime.NumCPU())

	for _, path := range files {
		path := path
		g.Go(func() error {
			return processFile(ctx, path, absDir, totalFiles, st, emb, ch, stats, &printMu)
		})
	}

	_ = g.Wait()
	return stats
}

func processFile(ctx context.Context, path, absDir string, totalFiles int, st *store.Store, emb interface {
	Embed([]string) ([][]float32, error)
}, ch *chunker.Chunker, stats *indexStats, printMu *sync.Mutex) error {
	content, err := os.ReadFile(path)
	if err != nil {
		logger.Warn("skipping unreadable file", "path", path, "error", err)
		return nil
	}

	hash := sha256.Sum256(content)
	hashStr := hex.EncodeToString(hash[:])

	if st.FileUnchanged(ctx, path, hashStr) {
		stats.skipped.Add(1)
		return nil
	}

	if err := st.DeleteDocumentByPath(ctx, path); err != nil {
		logger.Warn("failed to delete existing document", "path", path, "error", err)
	}

	docID, err := st.InsertDocument(ctx, path, hashStr, extractTitle(content, path))
	if err != nil {
		return fmt.Errorf("failed to insert document %s: %w", path, err)
	}

	chunks, err := ch.ChunkFile(content)
	if err != nil {
		return fmt.Errorf("failed to chunk %s: %w", path, err)
	}

	if len(chunks) == 0 {
		stats.indexed.Add(1)
		return nil
	}

	embedStart := time.Now()
	if err := embedAndInsertChunks(ctx, docID, chunks, st, emb, path); err != nil {
		return err
	}

	stats.indexed.Add(1)
	newProcessed := stats.processed.Add(1)

	printProgress(printMu, newProcessed, totalFiles, path, absDir, embedStart)
	return nil
}

func embedAndInsertChunks(ctx context.Context, docID int, chunks []chunker.Chunk, st *store.Store, emb interface {
	Embed([]string) ([][]float32, error)
}, path string) error {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}

	embeddings, err := emb.Embed(texts)
	if err != nil {
		return fmt.Errorf("failed to embed chunks for %s: %w", path, err)
	}

	storeChunks := make([]store.Chunk, len(chunks))
	for i, c := range chunks {
		storeChunks[i] = store.Chunk{
			HeadingPath:  c.HeadingPath,
			HeadingLevel: c.HeadingLevel,
			Content:      c.Content,
			StartLine:    c.StartLine,
		}
	}

	if err := st.InsertChunks(ctx, docID, storeChunks, embeddings); err != nil {
		return fmt.Errorf("failed to insert chunks: %w", err)
	}

	return nil
}

func printProgress(printMu *sync.Mutex, processed int32, totalFiles int, path, absDir string, embedStart time.Time) {
	printMu.Lock()
	defer printMu.Unlock()

	relPath, _ := filepath.Rel(absDir, path)
	if relPath == "" {
		relPath = filepath.Base(path)
	}
	displayName := relPath
	if len(displayName) > 50 {
		displayName = "..." + displayName[len(displayName)-47:]
	}
	fmt.Printf("\r\033[K[%d/%d] %s (Embed: %v)", processed, totalFiles, displayName, time.Since(embedStart).Round(time.Millisecond))
}

func extractTitle(content []byte, path string) string {
	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimPrefix(line, "# ")
		}
	}
	return filepath.Base(path)
}
