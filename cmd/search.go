package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/logger"
)

var searchLimit int

// NewSearchCmd creates the search command.
func NewSearchCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "search [query]",
		Short: "Search indexed documents using semantic similarity",
		Args:  cobra.MinimumNArgs(1),
		RunE:  runSearch,
	}

	cmd.Flags().IntVarP(&searchLimit, "limit", "n", 5, "Maximum number of results to return")

	return cmd
}

func runSearch(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	// Initialize app
	cfg, err := app.DefaultPaths(OnnxLibraryPath)
	if err != nil {
		return fmt.Errorf("failed to resolve paths: %w", err)
	}

	// Check if database exists
	if _, err := os.Stat(cfg.DBPath); os.IsNotExist(err) {
		return fmt.Errorf("database not found at %s. Run 'mcpmydocs index' first", cfg.DBPath)
	}

	// Validate limit
	if searchLimit < 1 {
		return fmt.Errorf("limit must be at least 1")
	}
	if searchLimit > 100 {
		logger.Warn("limit capped at maximum", "requested", searchLimit, "max", 100)
		searchLimit = 100
	}

	logger.Debug("search configuration", "database", cfg.DBPath, "model", cfg.ModelPath, "onnxLib", cfg.OnnxLibraryPath)
	logger.Info("searching", "query", query, "limit", searchLimit)

	application, err := app.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize application: %w", err)
	}
	defer application.Close()

	st := application.Store
	emb := application.Embedder

	// Embed the query
	embedStart := time.Now()
	embeddings, err := emb.Embed([]string{query})
	if err != nil {
		return fmt.Errorf("failed to embed query: %w", err)
	}
	logger.Debug("query embedded", "duration", time.Since(embedStart))

	if len(embeddings) == 0 {
		return fmt.Errorf("no embedding generated for query")
	}

	// Search
	ctx := context.Background()
	searchStart := time.Now()
	results, err := st.Search(ctx, embeddings[0], searchLimit)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}
	logger.Debug("search completed", "results", len(results), "duration", time.Since(searchStart))

	if len(results) == 0 {
		fmt.Println("No results found.")
		return nil
	}

	// Display results
	fmt.Printf("Found %d results for: %q\n\n", len(results), query)

	for i, r := range results {
		similarity := 1.0 - r.Distance // Convert distance to similarity
		fmt.Printf("─────────────────────────────────────────────────────────────\n")
		fmt.Printf("[%d] %s (%.1f%% similar)\n", i+1, r.HeadingPath, similarity*100)
		fmt.Printf("    File: %s:%d\n", r.FilePath, r.StartLine)
		fmt.Printf("\n")

		// Show truncated content
		content := strings.TrimSpace(r.Content)
		lines := strings.Split(content, "\n")
		maxLines := 6
		if len(lines) > maxLines {
			for _, line := range lines[:maxLines] {
				fmt.Printf("    %s\n", line)
			}
			fmt.Printf("    ... (%d more lines)\n", len(lines)-maxLines)
		} else {
			for _, line := range lines {
				fmt.Printf("    %s\n", line)
			}
		}
		fmt.Println()
	}

	return nil
}