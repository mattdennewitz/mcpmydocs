package reranker

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mattdennewitz/mcpmydocs/internal/store"
)

func TestSplitWords(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "simple words",
			input:    "hello world",
			expected: []string{"hello", "world"},
		},
		{
			name:     "with punctuation",
			input:    "hello, world!",
			expected: []string{"hello", ",", "world", "!"},
		},
		{
			name:     "multiple spaces",
			input:    "hello   world",
			expected: []string{"hello", "world"},
		},
		{
			name:     "empty string",
			input:    "",
			expected: nil,
		},
		{
			name:     "only spaces",
			input:    "   ",
			expected: nil,
		},
		{
			name:     "hyphenated",
			input:    "cross-encoder",
			expected: []string{"cross", "-", "encoder"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitWords(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("expected %d words, got %d: %v", len(tt.expected), len(result), result)
				return
			}
			for i, word := range result {
				if word != tt.expected[i] {
					t.Errorf("word %d: expected %q, got %q", i, tt.expected[i], word)
				}
			}
		})
	}
}

func TestConstants(t *testing.T) {
	if MaxSeqLen != 512 {
		t.Errorf("MaxSeqLen should be 512, got %d", MaxSeqLen)
	}
}

// skipIfNoModel skips the test if the ONNX model is not available
func skipIfNoModel(t *testing.T) (modelPath, onnxLibPath string) {
	t.Helper()

	cwd, _ := os.Getwd()
	modelPath = filepath.Join(cwd, "../../assets/models/rerank.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(cwd, "../assets/models/rerank.onnx")
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			modelPath = filepath.Join(cwd, "assets/models/rerank.onnx")
			if _, err := os.Stat(modelPath); os.IsNotExist(err) {
				t.Skip("rerank.onnx model not found")
			}
		}
	}

	onnxLibPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
		onnxLibPath = "/usr/local/lib/libonnxruntime.dylib"
		if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
			t.Skip("ONNX runtime not found")
		}
	}

	return modelPath, onnxLibPath
}

func TestNew_WithModel(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if r == nil {
		t.Fatal("New() returned nil reranker")
	}
	defer r.Close()

	if r.modelPath != modelPath {
		t.Errorf("modelPath: expected %q, got %q", modelPath, r.modelPath)
	}
	if len(r.vocab) == 0 {
		t.Error("vocab is empty")
	}
}

func TestRerank_EmptyResults(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	results, err := r.Rerank("test query", nil)
	if err != nil {
		t.Fatalf("Rerank() with nil results failed: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil, got %v", results)
	}

	results, err = r.Rerank("test query", []store.SearchResult{})
	if err != nil {
		t.Fatalf("Rerank() with empty results failed: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil, got %v", results)
	}
}

func TestRerank_SingleResult(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	input := []store.SearchResult{
		{
			ChunkID:     1,
			FilePath:    "/test.md",
			Title:       "Test",
			HeadingPath: "# Test",
			Content:     "This is test content about installation.",
			StartLine:   1,
			Distance:    0.1,
		},
	}

	results, err := r.Rerank("how to install", input)
	if err != nil {
		t.Fatalf("Rerank() failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Score should be a valid float
	if results[0].Score != results[0].Score { // NaN check
		t.Error("score is NaN")
	}
}

func TestRerank_MultipleResults(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	input := []store.SearchResult{
		{
			ChunkID:     1,
			Content:     "The weather today is sunny and warm.",
			HeadingPath: "# Weather",
		},
		{
			ChunkID:     2,
			Content:     "To install the software, run npm install.",
			HeadingPath: "# Installation",
		},
		{
			ChunkID:     3,
			Content:     "Installation guide: first download the package.",
			HeadingPath: "# Guide",
		},
	}

	results, err := r.Rerank("how to install", input)
	if err != nil {
		t.Fatalf("Rerank() failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Results should be sorted by score (descending)
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: result %d (%.4f) > result %d (%.4f)",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Log scores for debugging
	t.Logf("Reranking results for 'how to install':")
	for i, r := range results {
		t.Logf("  %d. (score: %.4f) %s", i+1, r.Score, r.Result.HeadingPath)
	}
}

func TestRerank_LongPassage(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoModel(t)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	// Create very long content that exceeds MaxSeqLen
	longContent := ""
	for i := 0; i < MaxSeqLen*2; i++ {
		longContent += "word "
	}

	input := []store.SearchResult{
		{
			ChunkID: 1,
			Content: longContent,
		},
	}

	// Should not panic and should produce valid results
	results, err := r.Rerank("test query", input)
	if err != nil {
		t.Fatalf("Rerank() with long content failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
}

// Benchmark tests

func BenchmarkRerank_10Results(b *testing.B) {
	modelPath, onnxLibPath := skipIfNoModelBench(b)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	input := make([]store.SearchResult, 10)
	for i := range input {
		input[i] = store.SearchResult{
			ChunkID: i,
			Content: "This is sample content for benchmarking the reranker performance.",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Rerank("benchmark query", input)
	}
}

func BenchmarkRerank_50Results(b *testing.B) {
	modelPath, onnxLibPath := skipIfNoModelBench(b)

	r, err := New(modelPath, onnxLibPath)
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer r.Close()

	input := make([]store.SearchResult, 50)
	for i := range input {
		input[i] = store.SearchResult{
			ChunkID: i,
			Content: "This is sample content for benchmarking the reranker performance.",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Rerank("benchmark query", input)
	}
}

func skipIfNoModelBench(b *testing.B) (modelPath, onnxLibPath string) {
	b.Helper()

	cwd, _ := os.Getwd()
	modelPath = filepath.Join(cwd, "../../assets/models/rerank.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skip("rerank.onnx model not found")
	}

	onnxLibPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
		onnxLibPath = "/usr/local/lib/libonnxruntime.dylib"
		if _, err := os.Stat(onnxLibPath); os.IsNotExist(err) {
			b.Skip("ONNX runtime not found")
		}
	}

	return modelPath, onnxLibPath
}
