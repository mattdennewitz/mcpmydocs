package cmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"mcpmydocs/internal/chunker"
	"mcpmydocs/internal/embedder"
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

	// Setup paths
	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "mcpmydocs.db")
	modelPath := filepath.Join(cwd, "assets/models/embed.onnx")

	// ONNX runtime library path
	onnxLibPath, err := resolveONNXLibraryPath()
	if err != nil {
		return fmt.Errorf("failed to locate ONNX runtime: %w", err)
	}

	logger.Info("starting indexing", "directory", absDir, "database", dbPath)
	logger.Debug("configuration", "model", modelPath, "onnxLib", onnxLibPath)

	// Count markdown files first for progress bar
	var totalFiles int
	_ = filepath.WalkDir(absDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			totalFiles++
		}
		return nil
	})

	// Initialize store
	st, err := store.New(dbPath)
	if err != nil {
		return fmt.Errorf("failed to create store: %w", err)
	}
	defer st.Close()

	// Initialize embedder
	emb, err := embedder.New(modelPath, onnxLibPath)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}

	// Initialize chunker
	ch := chunker.New()

	// Walk directory and index files
	ctx := context.Background()
	var indexed, skipped, processed int

	err = filepath.WalkDir(absDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip non-markdown files
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			return nil
		}

		processed++
		relPath, _ := filepath.Rel(absDir, path)
		if relPath == "" {
			relPath = filepath.Base(path)
		}
		// Truncate filename if too long
		displayName := relPath
		if len(displayName) > 50 {
			displayName = "..." + displayName[len(displayName)-47:]
		}
		fmt.Printf("\r\033[K[%d/%d] %s", processed, totalFiles, displayName)

		// Read file
		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		// Calculate hash
		hash := sha256.Sum256(content)
		hashStr := hex.EncodeToString(hash[:])

		// Check if unchanged
		if st.FileUnchanged(ctx, path, hashStr) {
			logger.Debug("file unchanged, skipping", "path", relPath)
			skipped++
			return nil
		}

		// Delete existing document
		if err := st.DeleteDocumentByPath(ctx, path); err != nil {
			// Continue anyway
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
			indexed++
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
		logger.Debug("chunks embedded", "path", relPath, "chunks", len(texts), "duration", time.Since(embedStart))

		// Insert chunks
		for i, c := range chunks {
			storeChunk := store.Chunk{
				HeadingPath:  c.HeadingPath,
				HeadingLevel: c.HeadingLevel,
				Content:      c.Content,
				StartLine:    c.StartLine,
			}
			if err := st.InsertChunk(ctx, docID, storeChunk, embeddings[i]); err != nil {
				return fmt.Errorf("failed to insert chunk: %w", err)
			}
		}

		logger.Debug("indexed file", "path", relPath, "chunks", len(chunks))
		indexed++
		return nil
	})

	// Clear progress line
	fmt.Printf("\r\033[K")

	if err != nil {
		return err
	}

	fmt.Printf("Indexing complete!\n")
	fmt.Printf("  Indexed: %d files\n", indexed)
	fmt.Printf("  Skipped: %d unchanged files\n", skipped)

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
