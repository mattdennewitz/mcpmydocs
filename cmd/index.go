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

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/chunker"
	"mcpmydocs/internal/logger"
	"mcpmydocs/internal/store"
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

	// Resolve to absolute path
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("failed to resolve directory: %w", err)
	}

	// Check directory exists
	info, err := os.Stat(absDir)
	if err != nil {
		return fmt.Errorf("directory not found: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", absDir)
	}

	// Initialize app
	cfg, err := app.DefaultPaths(OnnxLibraryPath)
	if err != nil {
		return fmt.Errorf("failed to resolve paths: %w", err)
	}

	application, err := app.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize application: %w", err)
	}
	defer application.Close()

	st := application.Store
	emb := application.Embedder

	logger.Info("starting indexing", "directory", absDir, "database", cfg.DBPath)
	logger.Debug("configuration", "model", cfg.ModelPath, "onnxLib", cfg.OnnxLibraryPath)

	// Initialize chunker
	ch := chunker.New()

	// Collect all markdown files first
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
	totalFiles := len(files)

	// Process files concurrently
	var (
		processed atomic.Int32
		indexed   atomic.Int32
		skipped   atomic.Int32
		printMu   sync.Mutex
	)

	g, ctx := errgroup.WithContext(context.Background())
	g.SetLimit(runtime.NumCPU())

	for _, path := range files {
		path := path // capture loop variable
		g.Go(func() error {
			// Read file
			content, err := os.ReadFile(path)
			if err != nil {
				logger.Warn("skipping unreadable file", "path", path, "error", err)
				return nil
			}

			// Calculate hash
			hash := sha256.Sum256(content)
			hashStr := hex.EncodeToString(hash[:])

			// Check if unchanged
			if st.FileUnchanged(ctx, path, hashStr) {
				skipped.Add(1)
				return nil
			}

			// Delete existing document
			if err := st.DeleteDocumentByPath(ctx, path); err != nil {
				logger.Warn("failed to delete existing document", "path", path, "error", err)
			}

			// Extract title (first line or filename)
			title := extractTitle(content, path)

			// Insert document
			docID, err := st.InsertDocument(ctx, path, hashStr, title)
			if err != nil {
				return fmt.Errorf("failed to insert document %s: %w", path, err)
			}

			// Chunk the file
			chunks, err := ch.ChunkFile(content)
			if err != nil {
				return fmt.Errorf("failed to chunk %s: %w", path, err)
			}

			if len(chunks) == 0 {
				indexed.Add(1)
				return nil
			}

			// Get embeddings for all chunks
			texts := make([]string, len(chunks))
			for i, c := range chunks {
				texts[i] = c.Content
			}

			embedStart := time.Now()
			embeddings, err := emb.Embed(texts)
			if err != nil {
				return fmt.Errorf("failed to embed chunks for %s: %w", path, err)
			}

			// Convert chunks to store.Chunk
			storeChunks := make([]store.Chunk, len(chunks))
			for i, c := range chunks {
				storeChunks[i] = store.Chunk{
					HeadingPath:  c.HeadingPath,
					HeadingLevel: c.HeadingLevel,
					Content:      c.Content,
					StartLine:    c.StartLine,
				}
			}

			// Insert chunks
			if err := st.InsertChunks(ctx, docID, storeChunks, embeddings); err != nil {
				return fmt.Errorf("failed to insert chunks: %w", err)
			}

			indexed.Add(1)
			newProcessed := processed.Add(1)

			// Update progress bar
			printMu.Lock()
			relPath, _ := filepath.Rel(absDir, path)
			if relPath == "" {
				relPath = filepath.Base(path)
			}
			// Truncate filename if too long
			displayName := relPath
			if len(displayName) > 50 {
				displayName = "..." + displayName[len(displayName)-47:]
			}
			fmt.Printf("\r\033[K[%d/%d] %s (Embed: %v)", newProcessed, totalFiles, displayName, time.Since(embedStart).Round(time.Millisecond))
			printMu.Unlock()

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// Clear progress line
	fmt.Printf("\r\033[K")

	fmt.Printf("Indexing complete!\n")
	fmt.Printf("  Indexed: %d files\n", indexed.Load())
	fmt.Printf("  Skipped: %d unchanged files\n", skipped.Load())

	return nil
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
