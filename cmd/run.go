package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/logger"
	"mcpmydocs/internal/search"
	"mcpmydocs/internal/store"
)

// Global instances for MCP server handlers
var (
	mcpStore   *store.Store
	mcpSearch  *search.Service
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

	// Initialize search service
	mcpStore = application.Store
	mcpSearch = search.New(application.Store, application.Embedder, application.Reranker)

	if mcpSearch.HasReranker() {
		logger.Info("reranker enabled")
	} else {
		logger.Info("reranker not available, using vector-only search")
	}

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "mcpmydocs",
		Version: "0.1.0",
	}, nil)

	searchDesc := "Search indexed markdown documents using semantic similarity. Returns relevant chunks with file paths and similarity scores."
	if mcpSearch.HasReranker() {
		searchDesc += " Uses cross-encoder reranking for improved relevance."
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: searchDesc,
	}, handleSearch)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_documents",
		Description: "List all indexed documents with their titles and file paths.",
	}, handleListDocuments)

	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		if ctx.Err() != nil {
			return nil
		}
		return fmt.Errorf("server error: %w", err)
	}

	return nil
}

func handleSearch(ctx context.Context, req *mcp.CallToolRequest, input SearchInput) (*mcp.CallToolResult, SearchOutput, error) {
	result, err := mcpSearch.Search(ctx, search.Params{
		Query:      input.Query,
		Limit:      input.Limit,
		Candidates: input.Candidates,
		Rerank:     input.Rerank,
	})
	if err != nil {
		return nil, SearchOutput{}, err
	}

	if len(result.Items) == 0 {
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: "No results found."}},
		}, SearchOutput{Results: "No results found."}, nil
	}

	output := formatResults(result)
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: output}},
	}, SearchOutput{Results: output}, nil
}

func formatResults(result *search.Result) string {
	suffix := ""
	if result.Reranked {
		suffix = " (reranked)"
	}

	output := fmt.Sprintf("Found %d results for: %q%s\n\n", len(result.Items), result.Query, suffix)

	for i, item := range result.Items {
		if result.Reranked {
			output += fmt.Sprintf("## Result %d (relevance: %.2f)\n", i+1, item.Score)
		} else {
			output += fmt.Sprintf("## Result %d (%.1f%% similar)\n", i+1, item.Score*100)
		}
		output += fmt.Sprintf("**File:** %s:%d\n", item.FilePath, item.StartLine)
		output += fmt.Sprintf("**Section:** %s\n\n", item.HeadingPath)
		output += fmt.Sprintf("```\n%s\n```\n\n", item.Content)
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
