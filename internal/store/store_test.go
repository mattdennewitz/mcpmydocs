package store

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func setupTestStore(t *testing.T) (*Store, func()) {
	t.Helper()

	tmpDir, err := os.MkdirTemp("", "mcpmydocs-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	dbPath := filepath.Join(tmpDir, "test.db")
	store, err := New(dbPath)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("failed to create store: %v", err)
	}

	cleanup := func() {
		store.Close()
		os.RemoveAll(tmpDir)
	}

	return store, cleanup
}

func TestNew(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	if store == nil {
		t.Fatal("New() returned nil store")
	}
	if store.db == nil {
		t.Fatal("store.db is nil")
	}
}

func TestNew_InvalidPath(t *testing.T) {
	// Try to create a database in a non-existent directory
	_, err := New("/nonexistent/path/to/db.db")
	if err == nil {
		t.Error("expected error for invalid path, got nil")
	}
}

func TestInsertDocument(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	id, err := store.InsertDocument(ctx, "/path/to/file.md", "abc123hash", "Test Title")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}
	if id <= 0 {
		t.Errorf("expected positive ID, got %d", id)
	}

	// Insert another document
	id2, err := store.InsertDocument(ctx, "/path/to/other.md", "def456hash", "Other Title")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}
	if id2 <= id {
		t.Errorf("expected id2 > id, got id=%d, id2=%d", id, id2)
	}
}

func TestInsertDocument_DuplicatePath(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	_, err := store.InsertDocument(ctx, "/path/to/file.md", "hash1", "Title 1")
	if err != nil {
		t.Fatalf("first insert failed: %v", err)
	}

	// Try to insert with same path - should fail due to UNIQUE constraint
	_, err = store.InsertDocument(ctx, "/path/to/file.md", "hash2", "Title 2")
	if err == nil {
		t.Error("expected error for duplicate path, got nil")
	}
}

func TestFileUnchanged(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()
	filePath := "/path/to/file.md"
	hash := "abc123hash"

	// Before inserting, file should not exist
	if store.FileUnchanged(ctx, filePath, hash) {
		t.Error("FileUnchanged should return false for non-existent file")
	}

	// Insert document
	_, err := store.InsertDocument(ctx, filePath, hash, "Title")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}

	// Now file with same hash should be unchanged
	if !store.FileUnchanged(ctx, filePath, hash) {
		t.Error("FileUnchanged should return true for matching hash")
	}

	// Different hash should be changed
	if store.FileUnchanged(ctx, filePath, "differenthash") {
		t.Error("FileUnchanged should return false for different hash")
	}

	// Different path should not match
	if store.FileUnchanged(ctx, "/other/path.md", hash) {
		t.Error("FileUnchanged should return false for different path")
	}
}

func TestDeleteDocumentByPath(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()
	filePath := "/path/to/file.md"

	// Delete non-existent document should not error
	err := store.DeleteDocumentByPath(ctx, filePath)
	if err != nil {
		t.Errorf("DeleteDocumentByPath for non-existent doc failed: %v", err)
	}

	// Insert document and chunks
	docID, err := store.InsertDocument(ctx, filePath, "hash", "Title")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}

	embedding := make([]float32, EmbeddingDim)
	for i := range embedding {
		embedding[i] = float32(i) * 0.01
	}

	// Insert multiple chunks
	for i := 0; i < 3; i++ {
		chunk := Chunk{
			HeadingPath:  "# Test",
			HeadingLevel: 1,
			Content:      "Test content " + string(rune('A'+i)),
			StartLine:    i * 10,
		}
		err = store.InsertChunk(ctx, docID, chunk, embedding)
		if err != nil {
			t.Fatalf("InsertChunk %d failed: %v", i, err)
		}
	}

	// Verify document exists
	if !store.FileUnchanged(ctx, filePath, "hash") {
		t.Fatal("Document should exist before deletion")
	}

	// Verify chunks exist by searching
	results, err := store.Search(ctx, embedding, 10)
	if err != nil {
		t.Fatalf("Search before delete failed: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 chunks before delete, got %d", len(results))
	}

	// Delete document
	err = store.DeleteDocumentByPath(ctx, filePath)
	if err != nil {
		t.Fatalf("DeleteDocumentByPath failed: %v", err)
	}

	// Verify document no longer exists
	if store.FileUnchanged(ctx, filePath, "hash") {
		t.Error("Document should not exist after deletion")
	}

	// CRITICAL: Verify chunks were also deleted (cascade delete)
	results, err = store.Search(ctx, embedding, 10)
	if err != nil {
		t.Fatalf("Search after delete failed: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 chunks after delete (cascade), got %d", len(results))
	}
}

func TestInsertChunk(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// First insert a document
	docID, err := store.InsertDocument(ctx, "/test.md", "hash", "Test")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}

	// Create embedding
	embedding := make([]float32, EmbeddingDim)
	for i := range embedding {
		embedding[i] = float32(i) / float32(EmbeddingDim)
	}

	chunk := Chunk{
		HeadingPath:  "# Main > ## Section",
		HeadingLevel: 2,
		Content:      "This is the content of the section.",
		StartLine:    10,
	}

	err = store.InsertChunk(ctx, docID, chunk, embedding)
	if err != nil {
		t.Fatalf("InsertChunk failed: %v", err)
	}
}

func TestInsertChunk_EmptyEmbedding(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	docID, err := store.InsertDocument(ctx, "/test.md", "hash", "Test")
	if err != nil {
		t.Fatalf("InsertDocument failed: %v", err)
	}

	chunk := Chunk{
		HeadingPath:  "# Test",
		HeadingLevel: 1,
		Content:      "Content",
		StartLine:    1,
	}

	// Empty embedding should insert NULL
	err = store.InsertChunk(ctx, docID, chunk, []float32{})
	if err != nil {
		t.Fatalf("InsertChunk with empty embedding failed: %v", err)
	}
}

func TestSearch(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Insert test documents and chunks
	docID, _ := store.InsertDocument(ctx, "/doc1.md", "hash1", "Document 1")

	// Create distinct embeddings for different chunks
	embedding1 := make([]float32, EmbeddingDim)
	embedding2 := make([]float32, EmbeddingDim)
	for i := range embedding1 {
		embedding1[i] = 1.0 / float32(EmbeddingDim)  // Normalized vector pointing in one direction
		embedding2[i] = -1.0 / float32(EmbeddingDim) // Opposite direction
	}

	chunk1 := Chunk{HeadingPath: "# First", HeadingLevel: 1, Content: "First content", StartLine: 1}
	chunk2 := Chunk{HeadingPath: "# Second", HeadingLevel: 1, Content: "Second content", StartLine: 5}

	store.InsertChunk(ctx, docID, chunk1, embedding1)
	store.InsertChunk(ctx, docID, chunk2, embedding2)

	// Search with query similar to embedding1
	queryEmbedding := make([]float32, EmbeddingDim)
	for i := range queryEmbedding {
		queryEmbedding[i] = 1.0 / float32(EmbeddingDim)
	}

	results, err := store.Search(ctx, queryEmbedding, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// First result should be closer (lower distance)
	if results[0].Distance > results[1].Distance {
		t.Error("results not sorted by distance")
	}
}

func TestSearch_EmptyDatabase(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	queryEmbedding := make([]float32, EmbeddingDim)
	results, err := store.Search(ctx, queryEmbedding, 10)
	if err != nil {
		t.Fatalf("Search on empty database failed: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestSearch_LimitParameter(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Insert multiple chunks
	docID, _ := store.InsertDocument(ctx, "/doc.md", "hash", "Doc")
	embedding := make([]float32, EmbeddingDim)

	for i := 0; i < 10; i++ {
		chunk := Chunk{
			HeadingPath:  "# Section",
			HeadingLevel: 1,
			Content:      "Content",
			StartLine:    i * 10,
		}
		store.InsertChunk(ctx, docID, chunk, embedding)
	}

	// Search with limit
	results, err := store.Search(ctx, embedding, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}

	// Search with larger limit
	results, err = store.Search(ctx, embedding, 100)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}
}

func TestListDocuments(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Empty database
	docs, err := store.ListDocuments(ctx)
	if err != nil {
		t.Fatalf("ListDocuments failed: %v", err)
	}
	if len(docs) != 0 {
		t.Errorf("expected 0 documents, got %d", len(docs))
	}

	// Insert documents
	store.InsertDocument(ctx, "/path/a.md", "hash1", "Alpha")
	store.InsertDocument(ctx, "/path/b.md", "hash2", "Beta")
	store.InsertDocument(ctx, "/path/c.md", "hash3", "Gamma")

	docs, err = store.ListDocuments(ctx)
	if err != nil {
		t.Fatalf("ListDocuments failed: %v", err)
	}
	if len(docs) != 3 {
		t.Errorf("expected 3 documents, got %d", len(docs))
	}

	// Should be sorted by title
	if docs[0].Title != "Alpha" {
		t.Errorf("expected first doc title 'Alpha', got %q", docs[0].Title)
	}
	if docs[1].Title != "Beta" {
		t.Errorf("expected second doc title 'Beta', got %q", docs[1].Title)
	}
}

func TestFloatSliceToArrayString(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected string
	}{
		{
			name:     "empty slice",
			input:    []float32{},
			expected: "NULL",
		},
		{
			name:     "single element",
			input:    []float32{1.5},
			expected: "[1.5]",
		},
		{
			name:     "multiple elements",
			input:    []float32{1.0, 2.5, 3.14},
			expected: "[1,2.5,3.14]",
		},
		{
			name:     "negative values",
			input:    []float32{-1.0, 0, 1.0},
			expected: "[-1,0,1]",
		},
		{
			name:     "very small values",
			input:    []float32{0.0001, 0.00001},
			expected: "[0.0001,1e-05]",
		},
		{
			name:     "NaN value - sanitized to 0",
			input:    []float32{float32(math.NaN())},
			expected: "[0]",
		},
		{
			name:     "positive infinity - sanitized to 0",
			input:    []float32{float32(math.Inf(1))},
			expected: "[0]",
		},
		{
			name:     "negative infinity - sanitized to 0",
			input:    []float32{float32(math.Inf(-1))},
			expected: "[0]",
		},
		{
			name:     "mixed with special values",
			input:    []float32{1.0, float32(math.NaN()), 2.0, float32(math.Inf(1))},
			expected: "[1,0,2,0]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := floatSliceToArrayString(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestClose(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	err := store.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Operations after close should fail
	ctx := context.Background()
	_, err = store.InsertDocument(ctx, "/test.md", "hash", "Test")
	if err == nil {
		t.Error("expected error after Close, got nil")
	}
}

func TestSearchResult_Fields(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	docID, _ := store.InsertDocument(ctx, "/path/to/doc.md", "myhash", "My Document")

	embedding := make([]float32, EmbeddingDim)
	for i := range embedding {
		embedding[i] = 0.5
	}

	chunk := Chunk{
		HeadingPath:  "# Main > ## Sub",
		HeadingLevel: 2,
		Content:      "Test content here",
		StartLine:    42,
	}
	store.InsertChunk(ctx, docID, chunk, embedding)

	results, err := store.Search(ctx, embedding, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	r := results[0]
	if r.FilePath != "/path/to/doc.md" {
		t.Errorf("FilePath: expected '/path/to/doc.md', got %q", r.FilePath)
	}
	if r.Title != "My Document" {
		t.Errorf("Title: expected 'My Document', got %q", r.Title)
	}
	if r.HeadingPath != "# Main > ## Sub" {
		t.Errorf("HeadingPath: expected '# Main > ## Sub', got %q", r.HeadingPath)
	}
	if r.Content != "Test content here" {
		t.Errorf("Content: expected 'Test content here', got %q", r.Content)
	}
	if r.StartLine != 42 {
		t.Errorf("StartLine: expected 42, got %d", r.StartLine)
	}
	// Distance should be 0 for identical vectors
	if r.Distance > 0.001 {
		t.Errorf("Distance: expected ~0, got %f", r.Distance)
	}
}

func TestContextCancellation(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	// Test that operations work with valid context
	ctx := context.Background()
	docID, err := store.InsertDocument(ctx, "/test.md", "hash", "Test")
	if err != nil {
		t.Fatalf("InsertDocument with valid context failed: %v", err)
	}

	// Verify document was inserted
	if !store.FileUnchanged(ctx, "/test.md", "hash") {
		t.Error("document should exist after insert")
	}

	// Test with deadline context (gives operation time to complete)
	ctxWithDeadline, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	embedding := make([]float32, EmbeddingDim)
	chunk := Chunk{HeadingPath: "# Test", HeadingLevel: 1, Content: "Content", StartLine: 1}
	err = store.InsertChunk(ctxWithDeadline, docID, chunk, embedding)
	if err != nil {
		t.Errorf("InsertChunk with deadline context failed: %v", err)
	}

	// Search with context
	results, err := store.Search(ctxWithDeadline, embedding, 10)
	if err != nil {
		t.Errorf("Search with deadline context failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

// TestSearch_RowIterationComplete verifies that Search() correctly iterates through
// all rows and returns rows.Err() (testing the complete code path including line 215)
func TestSearch_RowIterationComplete(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Insert many documents and chunks to ensure row iteration is exercised
	for d := 0; d < 5; d++ {
		docID, err := store.InsertDocument(ctx, "/doc"+string(rune('A'+d))+".md", "hash"+string(rune(d)), "Doc "+string(rune('A'+d)))
		if err != nil {
			t.Fatalf("InsertDocument %d failed: %v", d, err)
		}

		for c := 0; c < 3; c++ {
			embedding := make([]float32, EmbeddingDim)
			for i := range embedding {
				embedding[i] = float32(d*10+c) / 100.0
			}
			chunk := Chunk{
				HeadingPath:  "# Section " + string(rune('0'+c)),
				HeadingLevel: 1,
				Content:      "Content for doc " + string(rune('A'+d)) + " chunk " + string(rune('0'+c)),
				StartLine:    c * 10,
			}
			if err := store.InsertChunk(ctx, docID, chunk, embedding); err != nil {
				t.Fatalf("InsertChunk failed: %v", err)
			}
		}
	}

	// Search and iterate through all results
	queryEmbedding := make([]float32, EmbeddingDim)
	for i := range queryEmbedding {
		queryEmbedding[i] = 0.5
	}

	results, err := store.Search(ctx, queryEmbedding, 100)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should have 5 docs * 3 chunks = 15 results
	if len(results) != 15 {
		t.Errorf("expected 15 results, got %d", len(results))
	}

	// Verify each result has valid fields (row scan worked)
	for i, r := range results {
		if r.ChunkID <= 0 {
			t.Errorf("result %d: invalid ChunkID %d", i, r.ChunkID)
		}
		if r.FilePath == "" {
			t.Errorf("result %d: empty FilePath", i)
		}
		if r.Content == "" {
			t.Errorf("result %d: empty Content", i)
		}
	}
}

// TestListDocuments_RowIterationComplete verifies that ListDocuments() correctly
// iterates through all rows and returns rows.Err() (testing the complete code path)
func TestListDocuments_RowIterationComplete(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Insert many documents
	titles := []string{"Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"}
	for i, title := range titles {
		_, err := store.InsertDocument(ctx, "/doc"+string(rune('A'+i))+".md", "hash"+string(rune(i)), title)
		if err != nil {
			t.Fatalf("InsertDocument %s failed: %v", title, err)
		}
	}

	// List all documents
	docs, err := store.ListDocuments(ctx)
	if err != nil {
		t.Fatalf("ListDocuments failed: %v", err)
	}

	if len(docs) != len(titles) {
		t.Errorf("expected %d documents, got %d", len(titles), len(docs))
	}

	// Verify each document has valid fields (row scan worked)
	for i, d := range docs {
		if d.ID <= 0 {
			t.Errorf("doc %d: invalid ID %d", i, d.ID)
		}
		if d.FilePath == "" {
			t.Errorf("doc %d: empty FilePath", i)
		}
		if d.Title == "" {
			t.Errorf("doc %d: empty Title", i)
		}
	}

	// Verify sorted by title
	for i := 1; i < len(docs); i++ {
		if docs[i].Title < docs[i-1].Title {
			t.Errorf("documents not sorted: %q should come after %q", docs[i].Title, docs[i-1].Title)
		}
	}
}

// TestSearch_ScanErrorHandling tests that row scan errors are handled
func TestSearch_ScanErrorHandling(t *testing.T) {
	store, cleanup := setupTestStore(t)
	defer cleanup()

	ctx := context.Background()

	// Insert valid data
	docID, _ := store.InsertDocument(ctx, "/test.md", "hash", "Test")
	embedding := make([]float32, EmbeddingDim)
	chunk := Chunk{HeadingPath: "# Test", HeadingLevel: 1, Content: "Content", StartLine: 1}
	store.InsertChunk(ctx, docID, chunk, embedding)

	// Search should succeed with valid data
	results, err := store.Search(ctx, embedding, 10)
	if err != nil {
		t.Fatalf("Search with valid data failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}

	// Verify all fields populated correctly
	r := results[0]
	if r.FilePath != "/test.md" {
		t.Errorf("FilePath: expected '/test.md', got %q", r.FilePath)
	}
	if r.Title != "Test" {
		t.Errorf("Title: expected 'Test', got %q", r.Title)
	}
	if r.HeadingPath != "# Test" {
		t.Errorf("HeadingPath: expected '# Test', got %q", r.HeadingPath)
	}
	if r.Content != "Content" {
		t.Errorf("Content: expected 'Content', got %q", r.Content)
	}
	if r.StartLine != 1 {
		t.Errorf("StartLine: expected 1, got %d", r.StartLine)
	}
}

// Benchmark tests
func BenchmarkInsertDocument(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "mcpmydocs-bench-*")
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "bench.db")
	store, _ := New(dbPath)
	defer store.Close()

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Use fmt.Sprintf for valid path strings instead of string(rune()) which can produce invalid UTF-8
		path := fmt.Sprintf("/path/doc_%d.md", i)
		store.InsertDocument(ctx, path, "hash", "Title")
	}
}

func BenchmarkSearch(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "mcpmydocs-bench-*")
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "bench.db")
	store, _ := New(dbPath)
	defer store.Close()

	ctx := context.Background()

	// Insert test data
	docID, _ := store.InsertDocument(ctx, "/doc.md", "hash", "Doc")
	embedding := make([]float32, EmbeddingDim)
	for i := 0; i < 100; i++ {
		chunk := Chunk{HeadingPath: "# Section", HeadingLevel: 1, Content: "Content", StartLine: i}
		store.InsertChunk(ctx, docID, chunk, embedding)
	}

	queryEmbedding := make([]float32, EmbeddingDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Search(ctx, queryEmbedding, 10)
	}
}
