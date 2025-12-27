package cmd

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"

	"mcpmydocs/internal/app"
	"mcpmydocs/internal/embedder"
	"mcpmydocs/internal/logger"
	"mcpmydocs/internal/store"
)

// Global instances for the MCP server handlers
var (
	mcpStore    *store.Store
	mcpEmbedder *embedder.Embedder
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
	Query string `json:"query" jsonschema:"The search query to find relevant documents"`
	Limit int    `json:"limit,omitempty" jsonschema:"Maximum number of results to return (default: 5, max: 20)"`
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
	defer application.Close()

	// Assign globals for handlers
	mcpStore = application.Store
	mcpEmbedder = application.Embedder

	// Create MCP server
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "mcpmydocs",
		Version: "0.1.0",
	}, nil)

	// Register search tool
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search indexed markdown documents using semantic similarity. Returns relevant chunks with file paths and similarity scores.",
	}, handleSearch)

	// Register list_documents tool
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_documents",
		Description: "List all indexed documents with their titles and file paths.",
	}, handleListDocuments)

	// Start server on stdio
	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
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
		limit = 5
	}
	if limit > 20 {
		limit = 20
	}
	if limit < 1 {
		limit = 1
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

	// Search
	searchStart := time.Now()
	results, err := mcpStore.Search(ctx, embeddings[0], limit)
	if err != nil {
		return nil, SearchOutput{}, fmt.Errorf("search failed: %w", err)
	}
	logger.Debug("search completed", "results", len(results), "duration", time.Since(searchStart))

	if len(results) == 0 {
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: "No results found."}},
		}, SearchOutput{Results: "No results found."}, nil
	}

	// Format results
	var output string
	output += fmt.Sprintf("Found %d results for: %q\n\n", len(results), input.Query)

	for i, r := range results {
		similarity := (1.0 - r.Distance) * 100
		output += fmt.Sprintf("## Result %d (%.1f%% similar)\n", i+1, similarity)
		output += fmt.Sprintf("**File:** %s:%d\n", r.FilePath, r.StartLine)
		output += fmt.Sprintf("**Section:** %s\n\n", r.HeadingPath)
		output += fmt.Sprintf("```\n%s\n```\n\n", r.Content)
	}

	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: output}},
	}, SearchOutput{Results: output}, nil
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
