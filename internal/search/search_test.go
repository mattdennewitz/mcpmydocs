package search

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"mcpmydocs/internal/embedder"
	"mcpmydocs/internal/reranker"
	"mcpmydocs/internal/store"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		name       string
		val        int
		min        int
		max        int
		defaultVal int
		expected   int
	}{
		{"zero returns default", 0, 1, 20, 5, 5},
		{"value in range", 10, 1, 20, 5, 10},
		{"value below min", -5, 1, 20, 5, 1},
		{"value above max", 100, 1, 20, 5, 20},
		{"value equals min", 1, 1, 20, 5, 1},
		{"value equals max", 20, 1, 20, 5, 20},
		{"negative min", -10, -20, 0, -5, -10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := clamp(tt.val, tt.min, tt.max, tt.defaultVal)
			if result != tt.expected {
				t.Errorf("clamp(%d, %d, %d, %d) = %d; want %d",
					tt.val, tt.min, tt.max, tt.defaultVal, result, tt.expected)
			}
		})
	}
}

func TestNew(t *testing.T) {
	svc := New(nil, nil, nil)
	if svc == nil {
		t.Fatal("New returned nil")
	}
}

func TestHasReranker(t *testing.T) {
	t.Run("without reranker", func(t *testing.T) {
		svc := New(nil, nil, nil)
		if svc.HasReranker() {
			t.Error("HasReranker should return false when reranker is nil")
		}
	})

	// Note: Testing with actual reranker requires ONNX model, tested in integration tests
}

func TestSearch_EmptyQuery(t *testing.T) {
	svc := New(nil, nil, nil)

	_, err := svc.Search(context.Background(), Params{Query: ""})
	if err == nil {
		t.Error("expected error for empty query")
	}
	if err.Error() != "query is required" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSearch_NilEmbedder(t *testing.T) {
	svc := New(nil, nil, nil)

	_, err := svc.Search(context.Background(), Params{Query: "test"})
	if err == nil {
		t.Error("expected error for nil embedder")
	}
	if err.Error() != "embedder not initialized" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestConstants(t *testing.T) {
	// Verify constants are sensible
	if MinLimit < 1 {
		t.Error("MinLimit should be at least 1")
	}
	if MaxLimit < MinLimit {
		t.Error("MaxLimit should be >= MinLimit")
	}
	if DefaultLimit < MinLimit || DefaultLimit > MaxLimit {
		t.Error("DefaultLimit should be within range")
	}

	if MinCandidates < 1 {
		t.Error("MinCandidates should be at least 1")
	}
	if MaxCandidates < MinCandidates {
		t.Error("MaxCandidates should be >= MinCandidates")
	}
	if DefaultCandidates < MinCandidates || DefaultCandidates > MaxCandidates {
		t.Error("DefaultCandidates should be within range")
	}
}

// Helper to find ONNX library path
func findONNXLib() string {
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

// Helper to find model path
func findModelPath() string {
	cwd, _ := os.Getwd()
	paths := []string{
		filepath.Join(cwd, "../../assets/models/embed.onnx"),
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

// Helper to find reranker model path
func findRerankerPath() string {
	cwd, _ := os.Getwd()
	paths := []string{
		filepath.Join(cwd, "../../assets/models/rerank.onnx"),
		filepath.Join(cwd, "../assets/models/rerank.onnx"),
		filepath.Join(cwd, "assets/models/rerank.onnx"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// Integration tests - require ONNX runtime and models

func TestSearch_NoResults(t *testing.T) {
	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	svc := New(st, emb, nil)

	result, err := svc.Search(context.Background(), Params{Query: "test query"})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}
	if len(result.Items) != 0 {
		t.Errorf("expected 0 items, got %d", len(result.Items))
	}
	if result.Reranked {
		t.Error("should not be reranked without reranker")
	}
}

func TestSearch_WithResults(t *testing.T) {
	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()

	// Insert test data
	docID, err := st.InsertDocument(ctx, "/test.md", "hash123", "Test Doc")
	if err != nil {
		t.Fatalf("failed to insert document: %v", err)
	}

	content := "How to install the application on your computer"
	embeddings, err := emb.Embed([]string{content})
	if err != nil {
		t.Fatalf("failed to embed content: %v", err)
	}

	chunk := store.Chunk{
		HeadingPath:  "# Installation",
		HeadingLevel: 1,
		Content:      content,
		StartLine:    1,
	}
	err = st.InsertChunk(ctx, docID, chunk, embeddings[0])
	if err != nil {
		t.Fatalf("failed to insert chunk: %v", err)
	}

	svc := New(st, emb, nil)

	result, err := svc.Search(ctx, Params{Query: "how to install", Limit: 5})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if result == nil {
		t.Fatal("result is nil")
	}
	if len(result.Items) == 0 {
		t.Error("expected at least 1 result")
	}
	if result.Items[0].Content != content {
		t.Errorf("unexpected content: %s", result.Items[0].Content)
	}
	if result.Items[0].HeadingPath != "# Installation" {
		t.Errorf("unexpected heading: %s", result.Items[0].HeadingPath)
	}
	if result.Reranked {
		t.Error("should not be reranked without reranker")
	}
}

func TestSearch_LimitClamping(t *testing.T) {
	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()

	// Insert multiple chunks
	docID, _ := st.InsertDocument(ctx, "/test.md", "hash", "Test")
	for i := 0; i < 25; i++ {
		embedding := make([]float32, store.EmbeddingDim)
		chunk := store.Chunk{
			HeadingPath:  "# Test",
			HeadingLevel: 1,
			Content:      "Test content",
			StartLine:    i,
		}
		st.InsertChunk(ctx, docID, chunk, embedding)
	}

	svc := New(st, emb, nil)

	tests := []struct {
		name     string
		limit    int
		expected int // expected max results (may be less if fewer in DB)
	}{
		{"zero uses default", 0, DefaultLimit},
		{"negative clamped to min", -5, MinLimit},
		{"over max clamped", 100, MaxLimit},
		{"valid limit", 10, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := svc.Search(ctx, Params{Query: "test", Limit: tt.limit})
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			if len(result.Items) > tt.expected {
				t.Errorf("got %d results, expected at most %d", len(result.Items), tt.expected)
			}
		})
	}
}

func TestSearch_RerankFlag(t *testing.T) {
	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()

	// Insert a chunk
	docID, _ := st.InsertDocument(ctx, "/test.md", "hash", "Test")
	embeddings, _ := emb.Embed([]string{"test content"})
	chunk := store.Chunk{
		HeadingPath:  "# Test",
		HeadingLevel: 1,
		Content:      "test content",
		StartLine:    1,
	}
	st.InsertChunk(ctx, docID, chunk, embeddings[0])

	// Test without reranker
	svc := New(st, emb, nil)

	t.Run("rerank=true without reranker", func(t *testing.T) {
		rerank := true
		result, err := svc.Search(ctx, Params{Query: "test", Rerank: &rerank})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		// Should not be reranked since reranker is nil
		if result.Reranked {
			t.Error("should not be reranked without reranker")
		}
	})

	t.Run("rerank=false", func(t *testing.T) {
		rerank := false
		result, err := svc.Search(ctx, Params{Query: "test", Rerank: &rerank})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if result.Reranked {
			t.Error("should not be reranked when disabled")
		}
	})
}

// searchTestEnv holds test resources for search tests
type searchTestEnv struct {
	tmpDir string
	st     *store.Store
	emb    *embedder.Embedder
	rr     *reranker.Reranker
}

func setupSearchTestEnv(t *testing.T, withReranker bool) *searchTestEnv {
	t.Helper()

	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	var rerankerPath string
	if withReranker {
		rerankerPath = findRerankerPath()
		if rerankerPath == "" {
			t.Skip("Reranker model not available")
		}
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("failed to create store: %v", err)
	}

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		st.Close()
		os.RemoveAll(tmpDir)
		t.Skipf("failed to create embedder: %v", err)
	}

	env := &searchTestEnv{tmpDir: tmpDir, st: st, emb: emb}

	if withReranker {
		rr, err := reranker.New(rerankerPath, onnxLib)
		if err != nil {
			env.cleanup()
			t.Skipf("failed to create reranker: %v", err)
		}
		env.rr = rr
	}

	return env
}

func (e *searchTestEnv) cleanup() {
	if e.rr != nil {
		e.rr.Close()
	}
	if e.emb != nil {
		e.emb.Close()
	}
	if e.st != nil {
		e.st.Close()
	}
	if e.tmpDir != "" {
		os.RemoveAll(e.tmpDir)
	}
}

func (e *searchTestEnv) insertTestChunks(ctx context.Context, t *testing.T, contents []string) {
	t.Helper()

	docID, _ := e.st.InsertDocument(ctx, "/test.md", "hash", "Test")
	for i, content := range contents {
		embeddings, _ := e.emb.Embed([]string{content})
		chunk := store.Chunk{
			HeadingPath:  "# Section",
			HeadingLevel: 1,
			Content:      content,
			StartLine:    i * 10,
		}
		e.st.InsertChunk(ctx, docID, chunk, embeddings[0])
	}
}

func TestSearch_WithReranker(t *testing.T) {
	env := setupSearchTestEnv(t, true)
	defer env.cleanup()

	ctx := context.Background()
	contents := []string{
		"How to install the software on Windows",
		"Configuration options for advanced users",
		"Troubleshooting installation problems",
	}
	env.insertTestChunks(ctx, t, contents)

	svc := New(env.st, env.emb, env.rr)

	if !svc.HasReranker() {
		t.Error("HasReranker should return true")
	}

	t.Run("auto rerank (default)", func(t *testing.T) {
		result, err := svc.Search(ctx, Params{Query: "install", Limit: 3})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if !result.Reranked {
			t.Error("should be reranked by default when reranker available")
		}
		if len(result.Items) == 0 {
			t.Error("expected results")
		}
	})

	t.Run("explicit rerank=true", func(t *testing.T) {
		rerank := true
		result, err := svc.Search(ctx, Params{Query: "install", Rerank: &rerank})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if !result.Reranked {
			t.Error("should be reranked")
		}
	})

	t.Run("explicit rerank=false", func(t *testing.T) {
		rerank := false
		result, err := svc.Search(ctx, Params{Query: "install", Rerank: &rerank})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if result.Reranked {
			t.Error("should not be reranked when disabled")
		}
	})
}

func TestResult_QueryPreserved(t *testing.T) {
	modelPath := findModelPath()
	onnxLib := findONNXLib()
	if modelPath == "" || onnxLib == "" {
		t.Skip("ONNX model or library not available")
	}

	tmpDir, err := os.MkdirTemp("", "search-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.New(filepath.Join(tmpDir, "test.db"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer st.Close()

	emb, err := embedder.New(modelPath, onnxLib)
	if err != nil {
		t.Skipf("failed to create embedder: %v", err)
	}
	defer emb.Close()

	svc := New(st, emb, nil)

	query := "my specific query"
	result, err := svc.Search(context.Background(), Params{Query: query})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if result.Query != query {
		t.Errorf("query not preserved: got %q, want %q", result.Query, query)
	}
}
