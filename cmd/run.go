package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/embedder"
	"mcpmydocs/internal/logger"
	"mcpmydocs/internal/reranker"
	"mcpmydocs/internal/store"
)

// Search limit constants
const (
	DefaultSearchLimit    = 5
	MaxSearchLimit        = 20
	MinSearchLimit        = 1
	DefaultCandidates     = 50
	MaxCandidates         = 100
	MinCandidates         = 1
)

// Global instances for the MCP server handlers
var (
	mcpStore    *store.Store
	mcpEmbedder *embedder.Embedder
	mcpReranker *reranker.Reranker // nil if reranker not available
)

// NewRunCmd creates the run command.
func NewRunCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "run",
		Short: "Start the MCP server to connect with AI agents",
		RunE:  runMCPServer,
	}
}

// SearchInput defines the input parameters for the search tool.
type SearchInput struct {
	Query      string `json:"query" jsonschema:"The search query to find relevant documents"`
	Limit      int    `json:"limit,omitempty" jsonschema:"Maximum number of results to return (default: 5, max: 20)"`
	Rerank     *bool  `json:"rerank,omitempty" jsonschema:"Enable cross-encoder reranking for better relevance (default: true when available)"`
	Candidates int    `json:"candidates,omitempty" jsonschema:"Number of candidates to fetch before reranking (default: 50, max: 100)"`
}

// SearchOutput defines the output for the search tool.
type SearchOutput struct {
	Results string `json:"results"`
}

// ListDocumentsInput defines the input for list_documents (empty).
type ListDocumentsInput struct{}

// ListDocumentsOutput defines the output for list_documents.
type ListDocumentsOutput struct {
	Documents string `json:"documents"`
}

func runMCPServer(cmd *cobra.Command, args []string) error {
	// Initialize app
	cfg, err := app.DefaultPaths(OnnxLibraryPath)
	if err != nil {
		return fmt.Errorf("failed to resolve paths: %w", err)
	}

	// Check if database exists
	if _, err := os.Stat(cfg.DBPath); os.IsNotExist(err) {
		return fmt.Errorf("database not found at %s. Run 'mcpmydocs index' first", cfg.DBPath)
	}

	application, err := app.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize application: %w", err)
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("shutdown signal received, cleaning up...")
		cancel()
		application.Close()
	}()

	defer application.Close()

	// Assign globals for handlers
	mcpStore = application.Store
	mcpEmbedder = application.Embedder
	mcpReranker = application.Reranker

	if mcpReranker != nil {
		logger.Info("reranker enabled")
	} else {
		logger.Info("reranker not available, using vector-only search")
	}

	// Create MCP server
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "mcpmydocs",
		Version: "0.1.0",
	}, nil)

	// Register search tool
	searchDesc := "Search indexed markdown documents using semantic similarity. Returns relevant chunks with file paths and similarity scores."
	if mcpReranker != nil {
		searchDesc += " Uses cross-encoder reranking for improved relevance."
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: searchDesc,
	}, handleSearch)

	// Register list_documents tool
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_documents",
		Description: "List all indexed documents with their titles and file paths.",
	}, handleListDocuments)

	// Start server on stdio
	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		// Context cancellation is expected on shutdown
		if ctx.Err() != nil {
			return nil
		}
		return fmt.Errorf("server error: %w", err)
	}

	return nil
}

func handleSearch(ctx context.Context, req *mcp.CallToolRequest, input SearchInput) (*mcp.CallToolResult, SearchOutput, error) {
	// Validate query
	if input.Query == "" {
		return nil, SearchOutput{}, fmt.Errorf("query parameter is required")
	}

	// Check embedder is initialized
	if mcpEmbedder == nil {
		return nil, SearchOutput{}, fmt.Errorf("embedder not initialized")
	}

	// Set default and clamp limit
	limit := input.Limit
	if limit == 0 {
		limit = DefaultSearchLimit
	}
	if limit > MaxSearchLimit {
		limit = MaxSearchLimit
	}
	if limit < MinSearchLimit {
		limit = MinSearchLimit
	}

	// Determine if we should rerank
	// Default: rerank if available
	useRerank := mcpReranker != nil
	if input.Rerank != nil {
		useRerank = *input.Rerank && mcpReranker != nil
	}

	// Set candidates count (only relevant when reranking)
	candidates := input.Candidates
	if candidates == 0 {
		candidates = DefaultCandidates
	}
	if candidates > MaxCandidates {
		candidates = MaxCandidates
	}
	if candidates < MinCandidates {
		candidates = MinCandidates
	}

	// How many to fetch from vector search
	fetchCount := limit
	if useRerank {
		fetchCount = candidates
	}

	// Embed the query
	embedStart := time.Now()
	embeddings, err := mcpEmbedder.Embed([]string{input.Query})
	if err != nil {
		return nil, SearchOutput{}, fmt.Errorf("failed to embed query: %w", err)
	}
	logger.Debug("query embedded", "duration", time.Since(embedStart))

	if len(embeddings) == 0 {
		return nil, SearchOutput{}, fmt.Errorf("no embedding generated for query")
	}

	// Vector search
	searchStart := time.Now()
	vectorResults, err := mcpStore.Search(ctx, embeddings[0], fetchCount)
	if err != nil {
		return nil, SearchOutput{}, fmt.Errorf("search failed: %w", err)
	}
	logger.Debug("vector search completed", "results", len(vectorResults), "duration", time.Since(searchStart))

	if len(vectorResults) == 0 {
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: "No results found."}},
		}, SearchOutput{Results: "No results found."}, nil
	}

	// Format results - either reranked or vector-only
	var output string
	if useRerank && len(vectorResults) > 0 {
		// Rerank results
		rerankStart := time.Now()
		rerankedResults, err := mcpReranker.Rerank(input.Query, vectorResults)
		if err != nil {
			logger.Warn("reranking failed, falling back to vector results", "error", err)
			// Fall back to vector results
			output = formatVectorResults(input.Query, vectorResults, limit)
		} else {
			logger.Debug("reranking completed", "results", len(rerankedResults), "duration", time.Since(rerankStart))
			output = formatRerankedResults(input.Query, rerankedResults, limit)
		}
	} else {
		output = formatVectorResults(input.Query, vectorResults, limit)
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: output}},
	}, SearchOutput{Results: output}, nil
}

// formatVectorResults formats vector search results (no reranking).
func formatVectorResults(query string, results []store.SearchResult, limit int) string {
	if limit > len(results) {
		limit = len(results)
	}
	results = results[:limit]

	var output string
	output += fmt.Sprintf("Found %d results for: %q\n\n", len(results), query)

	for i, r := range results {
		similarity := (1.0 - r.Distance) * 100
		output += fmt.Sprintf("## Result %d (%.1f%% similar)\n", i+1, similarity)
		output += fmt.Sprintf("**File:** %s:%d\n", r.FilePath, r.StartLine)
		output += fmt.Sprintf("**Section:** %s\n\n", r.HeadingPath)
		output += fmt.Sprintf("```\n%s\n```\n\n", r.Content)
	}

	return output
}

// formatRerankedResults formats reranked search results.
func formatRerankedResults(query string, results []reranker.ScoredResult, limit int) string {
	if limit > len(results) {
		limit = len(results)
	}
	results = results[:limit]

	var output string
	output += fmt.Sprintf("Found %d results for: %q (reranked)\n\n", len(results), query)

	for i, r := range results {
		output += fmt.Sprintf("## Result %d (relevance: %.2f)\n", i+1, r.Score)
		output += fmt.Sprintf("**File:** %s:%d\n", r.Result.FilePath, r.Result.StartLine)
		output += fmt.Sprintf("**Section:** %s\n\n", r.Result.HeadingPath)
		output += fmt.Sprintf("```\n%s\n```\n\n", r.Result.Content)
	}

	return output
}

func handleListDocuments(ctx context.Context, req *mcp.CallToolRequest, input ListDocumentsInput) (*mcp.CallToolResult, ListDocumentsOutput, error) {
	docs, err := mcpStore.ListDocuments(ctx)
	if err != nil {
		return nil, ListDocumentsOutput{}, fmt.Errorf("failed to list documents: %w", err)
	}

	if len(docs) == 0 {
		msg := "No documents indexed yet. Run 'mcpmydocs index <directory>' to index documents."
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: msg}},
		}, ListDocumentsOutput{Documents: msg}, nil
	}

	var output string
	output += fmt.Sprintf("Indexed %d documents:\n\n", len(docs))

	for _, d := range docs {
		output += fmt.Sprintf("- **%s**\n  %s\n", d.Title, d.FilePath)
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: output}},
	}, ListDocumentsOutput{Documents: output}, nil
}
