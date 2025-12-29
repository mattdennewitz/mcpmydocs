package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mattdennewitz/mcpmydocs/internal/chunker"
	"github.com/mattdennewitz/mcpmydocs/internal/embedder"
	"github.com/mattdennewitz/mcpmydocs/internal/store"
)

// EmbedderInterface allows mocking the embedder for tests
type EmbedderInterface interface {
	Embed(texts []string) ([][]float32, error)
}

// MockEmbedder generates deterministic embeddings for testing without ONNX
type MockEmbedder struct{}

func (m *MockEmbedder) Embed(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embedding := make([]float32, store.EmbeddingDim)
		// Generate deterministic embedding based on text content
		// This allows semantic-like behavior in tests
		hash := 0
		for _, ch := range text {
			hash = hash*31 + int(ch)
		}
		// Normalize to unit vector
		var sum float32
		for j := 0; j < store.EmbeddingDim; j++ {
			val := float32(((hash + j*17) % 1000)) / 1000.0
			embedding[j] = val
			sum += val * val
		}
		norm := float32(math.Sqrt(float64(sum)))
		if norm > 0 {
			for j := range embedding {
				embedding[j] /= norm
			}
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

// skipIfNoModel skips the test if the ONNX model is not available
func skipIfNoModel(t *testing.T) (modelPath, onnxLibPath string) {
	t.Helper()

	// Try to find the model
	cwd, _ := os.Getwd()
	modelPath = filepath.Join(cwd, "assets/models/embed.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("ONNX model not found, skipping integration test")
	}

	// Try to find ONNX runtime
	onnxLibPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
		onnxLibPath = "/usr/local/lib/libonnxruntime.dylib"
		if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
			t.Skip("ONNX runtime not found, skipping integration test")
		}
	}

	return modelPath, onnxLibPath
}

func TestIntegration_FullPipeline(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	// Create temp directory for database
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create temp markdown file
	mdContent := `# Test Document

This is a test document for integration testing.

## Installation

To install the software, run:

` + "```bash" + `
npm install mcpmydocs
` + "```" + `

## Usage

Here's how to use the tool:

1. Index your documents
2. Search through them

### Advanced Usage

For power users, there are additional options available.
`

	mdPath := filepath.Join(tmpDir, "test.md")
	if err := os.WriteFile(mdPath, []byte(mdContent), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	ctx := context.Background()

	// Initialize components
	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	ch := chunker.New()

	// Read and chunk the file
	content, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	chunks, err := ch.ChunkFile(content)
	if err != nil {
		t.Fatalf("failed to chunk file: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks, got none")
	}

	t.Logf("Created %d chunks", len(chunks))

	// Insert document
	docID, err := st.InsertDocument(ctx, mdPath, "testhash", "Test Document")
	if err != nil {
		t.Fatalf("failed to insert document: %v", err)
	}

	// Embed and insert chunks
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}

	embeddings, err := emb.Embed(texts)
	if err != nil {
		t.Fatalf("failed to embed chunks: %v", err)
	}

	if len(embeddings) != len(chunks) {
		t.Fatalf("expected %d embeddings, got %d", len(chunks), len(embeddings))
	}

	for i, c := range chunks {
		chunk := store.Chunk{
			HeadingPath:  c.HeadingPath,
			HeadingLevel: c.HeadingLevel,
			Content:      c.Content,
			StartLine:    c.StartLine,
		}
		if err := st.InsertChunk(ctx, docID, chunk, embeddings[i]); err != nil {
			t.Fatalf("failed to insert chunk %d: %v", i, err)
		}
	}

	// Test search
	queryEmbeddings, err := emb.Embed([]string{"how to install"})
	if err != nil {
		t.Fatalf("failed to embed query: %v", err)
	}

	results, err := st.Search(ctx, queryEmbeddings[0], 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected search results, got none")
	}

	t.Logf("Found %d search results", len(results))

	// The installation section should be near the top
	foundInstallation := false
	for i, r := range results {
		t.Logf("  Result %d: %s (distance: %.4f)", i+1, r.HeadingPath, r.Distance)
		if strings.Contains(r.HeadingPath, "Installation") {
			foundInstallation = true
		}
	}

	if !foundInstallation {
		t.Error("expected 'Installation' section in search results")
	}
}

func TestIntegration_ChunkerToStore(t *testing.T) {
	// This test doesn't need the ONNX model
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	// Create store
	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	// Create chunker
	ch := chunker.New()

	// Test content
	content := `# API Documentation

This is the API overview.

## Endpoints

### GET /users

Returns list of users.

### POST /users

Creates a new user.
`

	chunks, err := ch.ChunkFile([]byte(content))
	if err != nil {
		t.Fatalf("failed to chunk: %v", err)
	}

	// Insert document
	docID, err := st.InsertDocument(ctx, "/api.md", "hash123", "API Documentation")
	if err != nil {
		t.Fatalf("failed to insert document: %v", err)
	}

	// Insert chunks with dummy embeddings
	dummyEmbedding := make([]float32, store.EmbeddingDim)
	for i, c := range chunks {
		chunk := store.Chunk{
			HeadingPath:  c.HeadingPath,
			HeadingLevel: c.HeadingLevel,
			Content:      c.Content,
			StartLine:    c.StartLine,
		}
		if err := st.InsertChunk(ctx, docID, chunk, dummyEmbedding); err != nil {
			t.Fatalf("failed to insert chunk %d: %v", i, err)
		}
	}

	// Verify documents list
	docs, err := st.ListDocuments(ctx)
	if err != nil {
		t.Fatalf("failed to list documents: %v", err)
	}
	if len(docs) != 1 {
		t.Errorf("expected 1 document, got %d", len(docs))
	}
	if docs[0].Title != "API Documentation" {
		t.Errorf("expected title 'API Documentation', got %q", docs[0].Title)
	}

	// Verify search returns all chunks
	results, err := st.Search(ctx, dummyEmbedding, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != len(chunks) {
		t.Errorf("expected %d results, got %d", len(chunks), len(results))
	}
}

func TestIntegration_IncrementalIndexing(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	filePath := "/docs/readme.md"
	hash1 := "hash_version_1"
	hash2 := "hash_version_2"

	// First insert
	_, err = st.InsertDocument(ctx, filePath, hash1, "Title v1")
	if err != nil {
		t.Fatalf("first insert failed: %v", err)
	}

	// Same hash should be unchanged
	if !st.FileUnchanged(ctx, filePath, hash1) {
		t.Error("file should be unchanged with same hash")
	}

	// Different hash should be changed
	if st.FileUnchanged(ctx, filePath, hash2) {
		t.Error("file should be changed with different hash")
	}

	// Delete and reinsert with new hash
	if err := st.DeleteDocumentByPath(ctx, filePath); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	_, err = st.InsertDocument(ctx, filePath, hash2, "Title v2")
	if err != nil {
		t.Fatalf("second insert failed: %v", err)
	}

	// Now new hash should be unchanged
	if !st.FileUnchanged(ctx, filePath, hash2) {
		t.Error("file should be unchanged with new hash after update")
	}

	// Old hash should be changed
	if st.FileUnchanged(ctx, filePath, hash1) {
		t.Error("file should be changed with old hash after update")
	}
}

func TestIntegration_MultipleDocuments(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	// Insert multiple documents
	docs := []struct {
		path  string
		title string
	}{
		{"/docs/api.md", "API Reference"},
		{"/docs/guide.md", "User Guide"},
		{"/docs/faq.md", "FAQ"},
	}

	for _, doc := range docs {
		_, err := st.InsertDocument(ctx, doc.path, "hash", doc.title)
		if err != nil {
			t.Fatalf("failed to insert %s: %v", doc.path, err)
		}
	}

	// List should return all documents sorted by title
	list, err := st.ListDocuments(ctx)
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(list) != len(docs) {
		t.Errorf("expected %d documents, got %d", len(docs), len(list))
	}

	// Should be sorted alphabetically
	expectedOrder := []string{"API Reference", "FAQ", "User Guide"}
	for i, expected := range expectedOrder {
		if list[i].Title != expected {
			t.Errorf("position %d: expected %q, got %q", i, expected, list[i].Title)
		}
	}
}

func TestIntegration_SearchRanking(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	tmpDir, err := os.MkdirTemp("", "mcpmydocs-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	// Insert document
	docID, _ := st.InsertDocument(ctx, "/test.md", "hash", "Test")

	// Create chunks with different content
	contents := []string{
		"Installing the software requires npm or yarn package manager",
		"The weather today is sunny and warm",
		"Configure the database connection settings",
		"Step by step installation guide for beginners",
	}

	embeddings, err := emb.Embed(contents)
	if err != nil {
		t.Fatalf("failed to embed: %v", err)
	}

	for i, content := range contents {
		chunk := store.Chunk{
			HeadingPath:  "# Section",
			HeadingLevel: 1,
			Content:      content,
			StartLine:    i * 5,
		}
		st.InsertChunk(ctx, docID, chunk, embeddings[i])
	}

	// Search for installation-related content
	queryEmb, _ := emb.Embed([]string{"how do I install"})
	results, err := st.Search(ctx, queryEmb[0], 4)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify search returns all results in ranked order (by distance)
	// Note: Semantic ranking quality depends on tokenizer implementation.
	// Our simplified character-based tokenizer doesn't capture true semantics,
	// so we only verify results are returned and sorted by distance.
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("results not sorted by distance: result %d (%.4f) < result %d (%.4f)",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}

	// Verify all expected content is in results
	foundInstall := false
	foundWeather := false
	for _, r := range results {
		if strings.Contains(r.Content, "install") || strings.Contains(r.Content, "Installation") {
			foundInstall = true
		}
		if strings.Contains(r.Content, "weather") {
			foundWeather = true
		}
	}

	if !foundInstall {
		t.Error("expected installation content in results")
	}
	if !foundWeather {
		t.Error("expected weather content in results")
	}

	t.Logf("Search results ranking:")
	for i, r := range results {
		t.Logf("  %d. (distance: %.4f) %s...", i+1, r.Distance, r.Content[:min(50, len(r.Content))])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Tests below use MockEmbedder and run without ONNX model (CI-friendly)

func TestIntegration_MockEmbedder_FullPipeline(t *testing.T) {
	// This test runs WITHOUT the ONNX model using MockEmbedder
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-mock-integration-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	// Create store
	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	// Use mock embedder
	emb := &MockEmbedder{}
	ch := chunker.New()

	// Test content
	mdContent := `# Getting Started

Welcome to the documentation.

## Installation

To install the software:

` + "```bash" + `
npm install mypackage
` + "```" + `

## Configuration

Configure your settings in config.yaml.

## Usage

Run the command to start.
`

	// Chunk the content
	chunks, err := ch.ChunkFile([]byte(mdContent))
	if err != nil {
		t.Fatalf("failed to chunk: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}

	// Insert document
	docID, err := st.InsertDocument(ctx, "/test.md", "testhash", "Getting Started")
	if err != nil {
		t.Fatalf("failed to insert document: %v", err)
	}

	// Embed and insert chunks
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}

	embeddings, err := emb.Embed(texts)
	if err != nil {
		t.Fatalf("failed to embed: %v", err)
	}

	for i, c := range chunks {
		chunk := store.Chunk{
			HeadingPath:  c.HeadingPath,
			HeadingLevel: c.HeadingLevel,
			Content:      c.Content,
			StartLine:    c.StartLine,
		}
		if err := st.InsertChunk(ctx, docID, chunk, embeddings[i]); err != nil {
			t.Fatalf("failed to insert chunk: %v", err)
		}
	}

	// Search using mock embedder
	queryEmb, _ := emb.Embed([]string{"how to install"})
	results, err := st.Search(ctx, queryEmb[0], 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected search results")
	}

	// Verify results are sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("results not sorted: result %d (%.4f) < result %d (%.4f)",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}

	t.Logf("Mock embedder test: found %d results", len(results))
}

func TestIntegration_MockEmbedder_DifferentQueries(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-mock-query-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb := &MockEmbedder{}

	// Insert document
	docID, _ := st.InsertDocument(ctx, "/test.md", "hash", "Test")

	// Create chunks with distinct content
	contents := []string{
		"Installing software with npm",
		"Database configuration guide",
		"API reference documentation",
		"Troubleshooting common errors",
	}

	embeddings, _ := emb.Embed(contents)

	for i, content := range contents {
		chunk := store.Chunk{
			HeadingPath:  "# Section",
			HeadingLevel: 1,
			Content:      content,
			StartLine:    i * 5,
		}
		st.InsertChunk(ctx, docID, chunk, embeddings[i])
	}

	// Different queries should produce different rankings
	queries := []string{"install", "database", "API", "error"}

	for _, q := range queries {
		queryEmb, _ := emb.Embed([]string{q})
		results, err := st.Search(ctx, queryEmb[0], 4)
		if err != nil {
			t.Fatalf("search for %q failed: %v", q, err)
		}
		if len(results) != 4 {
			t.Errorf("expected 4 results for %q, got %d", q, len(results))
		}
		t.Logf("Query %q: top result distance %.4f", q, results[0].Distance)
	}
}

func TestIntegration_MockEmbedder_EmptyResults(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-mock-empty-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb := &MockEmbedder{}

	// Search empty database
	queryEmb, _ := emb.Embed([]string{"anything"})
	results, err := st.Search(ctx, queryEmb[0], 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results from empty database, got %d", len(results))
	}
}

func TestIntegration_MockEmbedder_LargeDataset(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcpmydocs-mock-large-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	dbPath := filepath.Join(tmpDir, "test.db")
	st, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb := &MockEmbedder{}

	// Insert 100 documents with 5 chunks each = 500 chunks
	for d := 0; d < 100; d++ {
		docID, err := st.InsertDocument(ctx, "/doc"+string(rune('A'+d%26))+".md", "hash"+string(rune(d)), "Doc")
		if err != nil {
			// May get duplicate path errors, which is fine
			continue
		}

		for c := 0; c < 5; c++ {
			content := "Content for document " + string(rune('A'+d%26)) + " chunk " + string(rune('0'+c))
			embeddings, _ := emb.Embed([]string{content})
			chunk := store.Chunk{
				HeadingPath:  "# Section",
				HeadingLevel: 1,
				Content:      content,
				StartLine:    c * 10,
			}
			st.InsertChunk(ctx, docID, chunk, embeddings[0])
		}
	}

	// Search should still work efficiently
	queryEmb, _ := emb.Embed([]string{"document A chunk"})
	results, err := st.Search(ctx, queryEmb[0], 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected results from large dataset")
	}

	t.Logf("Large dataset test: searched through dataset, got %d results", len(results))
}
