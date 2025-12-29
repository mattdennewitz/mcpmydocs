package cmd

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/mattdennewitz/mcpmydocs/internal/embedder"
	"github.com/mattdennewitz/mcpmydocs/internal/search"
	"github.com/mattdennewitz/mcpmydocs/internal/store"
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

// setupTestMCPEnvironment creates a test environment for MCP handler tests.
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

	if !strings.Contains(output.Documents, "No documents indexed") {
		t.Error("expected 'No documents indexed' message")
	}
}

func TestHandleListDocuments_WithDocuments(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	ctx := context.Background()

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

	if !strings.Contains(output.Documents, "First Document") {
		t.Error("output should contain 'First Document'")
	}
	if !strings.Contains(output.Documents, "Second Document") {
		t.Error("output should contain 'Second Document'")
	}
}

// TestHandleSearch_Integration tests the MCP handler with real components.
// Unit tests for search logic are in internal/search/search_test.go.
func TestHandleSearch_Integration(t *testing.T) {
	cleanup, _ := setupTestMCPEnvironment(t)
	defer cleanup()

	// Find ONNX components
	modelPath := findModelPath(t)
	onnxLib := findONNXLib(t)
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	mcpSearch = search.New(mcpStore, emb, nil)

	ctx := context.Background()

	// Insert test data
	docID, _ := mcpStore.InsertDocument(ctx, "/test.md", "hash", "Test Doc")
	embeddings, _ := emb.Embed([]string{"test content for searching"})
	chunk := store.Chunk{
		HeadingPath:  "# Test Section",
		HeadingLevel: 1,
		Content:      "test content for searching",
		StartLine:    1,
	}
	mcpStore.InsertChunk(ctx, docID, chunk, embeddings[0])

	req := &mcp.CallToolRequest{}

	t.Run("successful search", func(t *testing.T) {
		input := SearchInput{Query: "test", Limit: 5}
		result, output, err := handleSearch(ctx, req, input)
		if err != nil {
			t.Fatalf("handleSearch failed: %v", err)
		}
		if result == nil {
			t.Fatal("result is nil")
		}
		if !strings.Contains(output.Results, "Test Section") {
			t.Error("expected result to contain 'Test Section'")
		}
	})

	t.Run("no results", func(t *testing.T) {
		// Create new empty store
		tmpDir, _ := os.MkdirTemp("", "empty-test-*")
		defer os.RemoveAll(tmpDir)
		emptySt, _ := store.New(filepath.Join(tmpDir, "empty.db"))
		defer emptySt.Close()

		mcpSearch = search.New(emptySt, emb, nil)

		input := SearchInput{Query: "nonexistent query xyz", Limit: 5}
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
	})
}

func TestFormatResults(t *testing.T) {
	t.Run("vector results", func(t *testing.T) {
		result := &search.Result{
			Query: "test query",
			Items: []search.Item{
				{
					FilePath:    "/path/to/file.md",
					HeadingPath: "# Test",
					Content:     "test content",
					StartLine:   10,
					Score:       0.85,
				},
			},
			Reranked: false,
		}

		output := formatResults(result)

		if !strings.Contains(output, "test query") {
			t.Error("output should contain query")
		}
		if !strings.Contains(output, "85.0% similar") {
			t.Error("output should contain similarity percentage")
		}
		if !strings.Contains(output, "/path/to/file.md:10") {
			t.Error("output should contain file path and line")
		}
		if strings.Contains(output, "reranked") {
			t.Error("output should not mention reranked")
		}
	})

	t.Run("reranked results", func(t *testing.T) {
		result := &search.Result{
			Query: "test query",
			Items: []search.Item{
				{
					FilePath:    "/path/to/file.md",
					HeadingPath: "# Test",
					Content:     "test content",
					StartLine:   10,
					Score:       1.5,
				},
			},
			Reranked: true,
		}

		output := formatResults(result)

		if !strings.Contains(output, "reranked") {
			t.Error("output should mention reranked")
		}
		if !strings.Contains(output, "relevance: 1.50") {
			t.Error("output should contain relevance score")
		}
	})
}

// Helper functions for tests

func findONNXLib(t *testing.T) string {
	t.Helper()
	paths := []string{
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/lib/libonnxruntime.so",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func findModelPath(t *testing.T) string {
	t.Helper()
	cwd, _ := os.Getwd()
	paths := []string{
		filepath.Join(cwd, "../assets/models/embed.onnx"),
		filepath.Join(cwd, "assets/models/embed.onnx"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
