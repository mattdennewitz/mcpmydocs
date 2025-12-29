package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/logger"
	"mcpmydocs/internal/search"
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

	cmd.Flags().IntVarP(&searchLimit, "limit", "n", search.DefaultLimit, "Maximum number of results to return")
	cmd.Flags().BoolVar(&searchRerank, "rerank", false, "Enable cross-encoder reranking (default: auto-detect)")
	cmd.Flags().BoolVar(&searchNoRerank, "no-rerank", false, "Disable cross-encoder reranking")
	cmd.Flags().IntVar(&searchCandidates, "candidates", search.DefaultCandidates, "Number of candidates to fetch before reranking")

	return cmd
}

func runSearch(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	cfg, err := app.DefaultPaths(OnnxLibraryPath)
	if err != nil {
		return fmt.Errorf("failed to resolve paths: %w", err)
	}

	if _, err := os.Stat(cfg.DBPath); os.IsNotExist(err) {
		return fmt.Errorf("database not found at %s. Run 'mcpmydocs index' first", cfg.DBPath)
	}

	application, err := app.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize application: %w", err)
	}
	defer application.Close()

	svc := search.New(application.Store, application.Embedder, application.Reranker)

	// Determine rerank mode
	var rerank *bool
	if searchRerank {
		if !svc.HasReranker() {
			return fmt.Errorf("--rerank specified but reranker model not available")
		}
		t := true
		rerank = &t
	}
	if searchNoRerank {
		f := false
		rerank = &f
	}

	logger.Info("searching", "query", query, "rerank", rerank == nil || (rerank != nil && *rerank))

	result, err := svc.Search(context.Background(), search.Params{
		Query:      query,
		Limit:      searchLimit,
		Candidates: searchCandidates,
		Rerank:     rerank,
	})
	if err != nil {
		return err
	}

	if len(result.Items) == 0 {
		fmt.Println("No results found.")
		return nil
	}

	displayResults(result)
	return nil
}

func displayResults(result *search.Result) {
	suffix := ""
	if result.Reranked {
		suffix = " (reranked)"
	}

	fmt.Printf("Found %d results for: %q%s\n\n", len(result.Items), result.Query, suffix)

	for i, item := range result.Items {
		fmt.Printf("─────────────────────────────────────────────────────────────\n")
		if result.Reranked {
			fmt.Printf("[%d] %s (relevance: %.2f)\n", i+1, item.HeadingPath, item.Score)
		} else {
			fmt.Printf("[%d] %s (%.1f%% similar)\n", i+1, item.HeadingPath, item.Score*100)
		}
		fmt.Printf("    File: %s:%d\n\n", item.FilePath, item.StartLine)
		printTruncatedContent(item.Content)
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
