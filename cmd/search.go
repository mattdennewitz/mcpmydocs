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
	"mcpmydocs/internal/reranker"
	"mcpmydocs/internal/store"
)

var (
	searchLimit      int
	searchRerank     bool
	searchNoRerank   bool
	searchCandidates int
)

// NewSearchCmd creates the search command.
func NewSearchCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "search [query]",
		Short: "Search indexed documents using semantic similarity",
		Args:  cobra.MinimumNArgs(1),
		RunE:  runSearch,
	}

	cmd.Flags().IntVarP(&searchLimit, "limit", "n", DefaultSearchLimit, "Maximum number of results to return")
	cmd.Flags().BoolVar(&searchRerank, "rerank", false, "Enable cross-encoder reranking (default: auto-detect)")
	cmd.Flags().BoolVar(&searchNoRerank, "no-rerank", false, "Disable cross-encoder reranking")
	cmd.Flags().IntVar(&searchCandidates, "candidates", DefaultCandidates, "Number of candidates to fetch before reranking")

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
	if searchLimit < MinSearchLimit {
		return fmt.Errorf("limit must be at least %d", MinSearchLimit)
	}
	if searchLimit > MaxSearchLimit {
		logger.Warn("limit capped at maximum", "requested", searchLimit, "max", MaxSearchLimit)
		searchLimit = MaxSearchLimit
	}

	// Validate candidates
	if searchCandidates < MinCandidates {
		searchCandidates = MinCandidates
	}
	if searchCandidates > MaxCandidates {
		searchCandidates = MaxCandidates
	}

	application, err := app.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize application: %w", err)
	}
	defer application.Close()

	st := application.Store
	emb := application.Embedder
	rr := application.Reranker

	// Determine rerank mode
	useRerank := rr != nil // Default: use if available
	if searchRerank {
		useRerank = true
		if rr == nil {
			return fmt.Errorf("--rerank specified but reranker model not available")
		}
	}
	if searchNoRerank {
		useRerank = false
	}

	// How many to fetch from vector search
	fetchCount := searchLimit
	if useRerank {
		fetchCount = searchCandidates
	}

	logger.Debug("search configuration",
		"database", cfg.DBPath,
		"rerank", useRerank,
		"candidates", fetchCount,
		"limit", searchLimit)
	logger.Info("searching", "query", query, "rerank", useRerank)

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

	// Vector search
	ctx := context.Background()
	searchStart := time.Now()
	vectorResults, err := st.Search(ctx, embeddings[0], fetchCount)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}
	logger.Debug("vector search completed", "results", len(vectorResults), "duration", time.Since(searchStart))

	if len(vectorResults) == 0 {
		fmt.Println("No results found.")
		return nil
	}

	// Display results
	if useRerank {
		rerankStart := time.Now()
		rerankedResults, err := rr.Rerank(query, vectorResults)
		if err != nil {
			logger.Warn("reranking failed, falling back to vector results", "error", err)
			displayVectorResults(query, vectorResults, searchLimit)
		} else {
			logger.Debug("reranking completed", "results", len(rerankedResults), "duration", time.Since(rerankStart))
			displayRerankedResults(query, rerankedResults, searchLimit)
		}
	} else {
		displayVectorResults(query, vectorResults, searchLimit)
	}

	return nil
}

func displayVectorResults(query string, results []store.SearchResult, limit int) {
	if limit > len(results) {
		limit = len(results)
	}
	results = results[:limit]

	fmt.Printf("Found %d results for: %q\n\n", len(results), query)

	for i, r := range results {
		similarity := 1.0 - r.Distance
		fmt.Printf("─────────────────────────────────────────────────────────────\n")
		fmt.Printf("[%d] %s (%.1f%% similar)\n", i+1, r.HeadingPath, similarity*100)
		fmt.Printf("    File: %s:%d\n", r.FilePath, r.StartLine)
		fmt.Printf("\n")
		printTruncatedContent(r.Content)
		fmt.Println()
	}
}

func displayRerankedResults(query string, results []reranker.ScoredResult, limit int) {
	if limit > len(results) {
		limit = len(results)
	}
	results = results[:limit]

	fmt.Printf("Found %d results for: %q (reranked)\n\n", len(results), query)

	for i, r := range results {
		fmt.Printf("─────────────────────────────────────────────────────────────\n")
		fmt.Printf("[%d] %s (relevance: %.2f)\n", i+1, r.Result.HeadingPath, r.Score)
		fmt.Printf("    File: %s:%d\n", r.Result.FilePath, r.Result.StartLine)
		fmt.Printf("\n")
		printTruncatedContent(r.Result.Content)
		fmt.Println()
	}
}

func printTruncatedContent(content string) {
	content = strings.TrimSpace(content)
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
}