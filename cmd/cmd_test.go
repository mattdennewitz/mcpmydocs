package cmd

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"mcpmydocs/internal/embedder"
	"mcpmydocs/internal/search"
	"mcpmydocs/internal/store"
)

func TestExtractTitle(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		path     string
		expected string
	}{
		{
			name:     "h1 heading at start",
			content:  "# My Title\n\nSome content",
			path:     "/path/to/file.md",
			expected: "My Title",
		},
		{
			name:     "h1 heading with leading whitespace",
			content:  "  # Trimmed Title\n\nContent",
			path:     "/path/to/file.md",
			expected: "Trimmed Title",
		},
		{
			name:     "h1 heading after blank lines",
			content:  "\n\n# After Blanks\n\nContent",
			path:     "/path/to/file.md",
			expected: "After Blanks",
		},
		{
			name:     "no h1 heading - use filename",
			content:  "Just some text without heading",
			path:     "/path/to/myfile.md",
			expected: "myfile.md",
		},
		{
			name:     "empty content - use filename",
			content:  "",
			path:     "/docs/readme.md",
			expected: "readme.md",
		},
		{
			name:     "h2 heading only - use filename",
			content:  "## This is H2\n\nNot H1",
			path:     "/test.md",
			expected: "test.md",
		},
		{
			name:     "h1 not at start of line",
			content:  "Some text # Not a heading\n\nMore text",
			path:     "/fallback.md",
			expected: "fallback.md",
		},
		{
			name:     "multiple h1 headings - use first",
			content:  "# First\n\n# Second\n\n# Third",
			path:     "/multi.md",
			expected: "First",
		},
		{
			name:     "h1 with special characters",
			content:  "# API: v2.0 (beta) & more!",
			path:     "/api.md",
			expected: "API: v2.0 (beta) & more!",
		},
		{
			name:     "whitespace only content",
			content:  "   \n\t\n   ",
			path:     "/empty.md",
			expected: "empty.md",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractTitle([]byte(tt.content), tt.path)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestNewIndexCmd(t *testing.T) {
	cmd := NewIndexCmd()
	if cmd == nil {
		t.Fatal("NewIndexCmd returned nil")
	}
	if cmd.Use != "index [directory]" {
		t.Errorf("unexpected Use: %s", cmd.Use)
	}
	if cmd.Short == "" {
		t.Error("Short description is empty")
	}
}

func TestNewSearchCmd(t *testing.T) {
	cmd := NewSearchCmd()
	if cmd == nil {
		t.Fatal("NewSearchCmd returned nil")
	}
	if cmd.Use != "search [query]" {
		t.Errorf("unexpected Use: %s", cmd.Use)
	}
}

func TestNewRunCmd(t *testing.T) {
	cmd := NewRunCmd()
	if cmd == nil {
		t.Fatal("NewRunCmd returned nil")
	}
	if cmd.Use != "run" {
		t.Errorf("unexpected Use: %s", cmd.Use)
	}
}

func TestRunIndex_InvalidDirectory(t *testing.T) {
	cmd := NewIndexCmd()
	cmd.SetArgs([]string{"/nonexistent/path/that/does/not/exist"})

	err := cmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent directory")
	}
}

func TestRunIndex_NotADirectory(t *testing.T) {
	// Create a temp file (not directory)
	tmpFile, err := os.CreateTemp("", "mcpmydocs-test-*.txt")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	cmd := NewIndexCmd()
	cmd.SetArgs([]string{tmpFile.Name()})

	err = cmd.Execute()
	if err == nil {
		t.Error("expected error when path is not a directory")
	}
}

// setupTestMCPEnvironment creates a test environment for MCP handler tests
func setupTestMCPEnvironment(t *testing.T) (func(), string) {
	t.Helper()

	tmpDir, err := os.MkdirTemp("", "mcpmydocs-mcp-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("failed to create store: %v", err)
	}

	// Set global store for handlers
	mcpStore = st
	mcpSearch = nil

	cleanup := func() {
		if mcpStore != nil {
			mcpStore.Close()
		}
		mcpStore = nil
		mcpSearch = nil
		os.RemoveAll(tmpDir)
	}

	return cleanup, tmpDir
}

func TestHandleListDocuments_Empty(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	ctx := context.Background()
	req := &mcp.CallToolRequest{}
	input := ListDocumentsInput{}

	result, output, err := handleListDocuments(ctx, req, input)
	if err != nil {
		t.Fatalf("handleListDocuments failed: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}

	if len(result.Content) == 0 {
		t.Error("expected content in result")
	}

	// Should mention no documents
	if output.Documents == "" {
		t.Error("output.Documents is empty")
	}
}

func TestHandleListDocuments_WithDocuments(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	ctx := context.Background()

	// Insert test documents
	mcpStore.InsertDocument(ctx, "/doc1.md", "hash1", "First Document")
	mcpStore.InsertDocument(ctx, "/doc2.md", "hash2", "Second Document")

	req := &mcp.CallToolRequest{}
	input := ListDocumentsInput{}

	result, output, err := handleListDocuments(ctx, req, input)
	if err != nil {
		t.Fatalf("handleListDocuments failed: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}

	// Should contain document info
	if output.Documents == "" {
		t.Error("output.Documents is empty")
	}

	// Should mention both documents
	if !strings.Contains(output.Documents, "First Document") {
		t.Error("output should contain 'First Document'")
	}
	if !strings.Contains(output.Documents, "Second Document") {
		t.Error("output should contain 'Second Document'")
	}
}

func TestHandleSearch_EmptyQuery(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	// Create a search service with nil embedder
	mcpSearch = search.New(mcpStore, nil, nil)

	ctx := context.Background()
	req := &mcp.CallToolRequest{}
	input := SearchInput{Query: ""}

	_, _, err := handleSearch(ctx, req, input)
	if err == nil {
		t.Error("expected error for empty query")
	}
}

func TestHandleSearch_NoEmbedder(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	// mcpSearch with nil embedder
	mcpSearch = search.New(mcpStore, nil, nil)

	ctx := context.Background()
	req := &mcp.CallToolRequest{}
	input := SearchInput{Query: "test query"}

	_, _, err := handleSearch(ctx, req, input)
	if err == nil {
		t.Error("expected error when embedder is nil")
	}
}

func TestHandleSearch_LimitClamping(t *testing.T) {
	cleanup, tmpDir := setupTestMCPEnvironment(t)
	defer cleanup()

	// Try to set up embedder if model exists
	cwd, _ := os.Getwd()
	modelPath := filepath.Join(cwd, "../assets/models/embed.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(cwd, "assets/models/embed.onnx")
	}

	onnxLibPath := "/opt/homebrew/lib/libonnxruntime.dylib"
	if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
		onnxLibPath = "/usr/local/lib/libonnxruntime.dylib"
	}

	emb, err := embedder.New(modelPath, onnxLibPath)
	if err != nil {
		t.Skip("ONNX model not available, skipping search test")
	}
	mcpSearch = search.New(mcpStore, emb, nil)

	ctx := context.Background()

	// Insert a document with chunk
	docID, _ := mcpStore.InsertDocument(ctx, "/test.md", "hash", "Test")
	embedding := make([]float32, store.EmbeddingDim)
	chunk := store.Chunk{
		HeadingPath:  "# Test",
		HeadingLevel: 1,
		Content:      "Test content for searching",
		StartLine:    1,
	}
	mcpStore.InsertChunk(ctx, docID, chunk, embedding)

	// Test limit clamping - negative limit should become 1
	req := &mcp.CallToolRequest{}
	input := SearchInput{Query: "test", Limit: -5}

	result, _, err := handleSearch(ctx, req, input)
	if err != nil {
		t.Fatalf("handleSearch failed: %v", err)
	}
	if result == nil {
		t.Fatal("result is nil")
	}

	// Test limit clamping - over 20 should become 20
	input = SearchInput{Query: "test", Limit: 100}
	result, _, err = handleSearch(ctx, req, input)
	if err != nil {
		t.Fatalf("handleSearch failed: %v", err)
	}

	_ = tmpDir // silence unused warning
}

func TestHandleSearch_NoResults(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	// Try to set up embedder
	cwd, _ := os.Getwd()
	modelPath := filepath.Join(cwd, "../assets/models/embed.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(cwd, "assets/models/embed.onnx")
	}

	onnxLibPath := "/opt/homebrew/lib/libonnxruntime.dylib"
	if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
		onnxLibPath = "/usr/local/lib/libonnxruntime.dylib"
	}

	emb, err := embedder.New(modelPath, onnxLibPath)
	if err != nil {
		t.Skip("ONNX model not available, skipping search test")
	}
	mcpSearch = search.New(mcpStore, emb, nil)

	ctx := context.Background()
	req := &mcp.CallToolRequest{}
	input := SearchInput{Query: "test query", Limit: 5}

	// Empty database - should return "No results found"
	result, output, err := handleSearch(ctx, req, input)
	if err != nil {
		t.Fatalf("handleSearch failed: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}

	if output.Results != "No results found." {
		t.Errorf("expected 'No results found.', got %q", output.Results)
	}
}
