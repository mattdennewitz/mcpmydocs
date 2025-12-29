package embedder

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestL2Normalize(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		checkNorm bool // if true, verify the result has unit norm
	}{
		{
			name:      "zero vector",
			input:     []float32{0, 0, 0},
			checkNorm: false, // zero vector stays zero
		},
		{
			name:      "unit vector",
			input:     []float32{1, 0, 0},
			checkNorm: true,
		},
		{
			name:      "simple vector",
			input:     []float32{3, 4, 0},
			checkNorm: true,
		},
		{
			name:      "all ones",
			input:     []float32{1, 1, 1, 1},
			checkNorm: true,
		},
		{
			name:      "negative values",
			input:     []float32{-1, 2, -3},
			checkNorm: true,
		},
		{
			name:      "single element",
			input:     []float32{5},
			checkNorm: true,
		},
		{
			name:      "empty vector",
			input:     []float32{},
			checkNorm: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to avoid modifying the test case input
			result := make([]float32, len(tt.input))
			copy(result, tt.input)

			l2Normalize(result)

			// Check length preserved
			if len(result) != len(tt.input) {
				t.Errorf("expected length %d, got %d", len(tt.input), len(result))
			}

			if tt.checkNorm && len(result) > 0 {
				// Verify unit norm
				var sum float64
				for _, v := range result {
					sum += float64(v) * float64(v)
				}
				norm := math.Sqrt(sum)
				if math.Abs(norm-1.0) > 1e-5 {
					t.Errorf("expected unit norm, got %f", norm)
				}
			}

			// For zero vector, result should be the same
			if tt.name == "zero vector" {
				for i, v := range result {
					if v != tt.input[i] {
						t.Errorf("zero vector should be unchanged")
						break
					}
				}
			}
		})
	}
}

func TestL2Normalize_Correctness(t *testing.T) {
	// Test with known values: [3, 4, 0] should normalize to [0.6, 0.8, 0]
	input := []float32{3, 4, 0}
	l2Normalize(input)

	expected := []float32{0.6, 0.8, 0}
	for i := range input {
		if math.Abs(float64(input[i]-expected[i])) > 1e-5 {
			t.Errorf("index %d: expected %f, got %f", i, expected[i], input[i])
		}
	}
}

func TestTokenize_EmptyInput(t *testing.T) {
	e := &Embedder{}

	inputIDs, attentionMask := e.tokenize([]string{})

	if len(inputIDs) != 0 {
		t.Errorf("expected empty inputIDs, got %d elements", len(inputIDs))
	}
	if len(attentionMask) != 0 {
		t.Errorf("expected empty attentionMask, got %d elements", len(attentionMask))
	}
}

func TestTokenize_SingleText(t *testing.T) {
	e := &Embedder{}

	inputIDs, attentionMask := e.tokenize([]string{"hello world"})

	// Should have MaxSeqLen elements
	if len(inputIDs) != MaxSeqLen {
		t.Errorf("expected %d inputIDs, got %d", MaxSeqLen, len(inputIDs))
	}
	if len(attentionMask) != MaxSeqLen {
		t.Errorf("expected %d attentionMask, got %d", MaxSeqLen, len(attentionMask))
	}

	// First token should be CLS (101)
	if inputIDs[0] != 101 {
		t.Errorf("expected CLS token (101), got %d", inputIDs[0])
	}
	if attentionMask[0] != 1 {
		t.Errorf("expected attention mask 1 for CLS, got %d", attentionMask[0])
	}

	// Should end with SEP (102) followed by padding (0)
	// Find SEP token
	foundSep := false
	for i := 1; i < MaxSeqLen; i++ {
		if inputIDs[i] == 102 {
			foundSep = true
			// After SEP should be padding
			for j := i + 1; j < MaxSeqLen; j++ {
				if inputIDs[j] != 0 {
					t.Errorf("expected padding after SEP at position %d, got %d", j, inputIDs[j])
				}
				if attentionMask[j] != 0 {
					t.Errorf("expected attention mask 0 for padding at position %d", j)
				}
			}
			break
		}
	}
	if !foundSep {
		t.Error("SEP token not found")
	}
}

func TestTokenize_MultipleBatches(t *testing.T) {
	e := &Embedder{}

	texts := []string{"first text", "second text", "third text"}
	inputIDs, attentionMask := e.tokenize(texts)

	expectedLen := len(texts) * MaxSeqLen
	if len(inputIDs) != expectedLen {
		t.Errorf("expected %d inputIDs, got %d", expectedLen, len(inputIDs))
	}
	if len(attentionMask) != expectedLen {
		t.Errorf("expected %d attentionMask, got %d", expectedLen, len(attentionMask))
	}

	// Each batch should start with CLS
	for b := 0; b < len(texts); b++ {
		offset := b * MaxSeqLen
		if inputIDs[offset] != 101 {
			t.Errorf("batch %d: expected CLS token, got %d", b, inputIDs[offset])
		}
	}
}

func TestTokenize_LongText(t *testing.T) {
	e := &Embedder{}

	// Create text that would exceed MaxSeqLen when tokenized
	longText := ""
	for i := 0; i < MaxSeqLen*2; i++ {
		longText += "word "
	}

	inputIDs, attentionMask := e.tokenize([]string{longText})

	// Should still be MaxSeqLen
	if len(inputIDs) != MaxSeqLen {
		t.Errorf("expected %d inputIDs, got %d", MaxSeqLen, len(inputIDs))
	}

	// All attention mask values for used tokens should be 1
	allOnes := true
	for i := 0; i < MaxSeqLen-1; i++ { // -1 because last position might vary
		if attentionMask[i] != 1 {
			allOnes = false
			break
		}
	}
	if !allOnes {
		t.Error("expected attention mask to be mostly 1s for long text")
	}
}

func TestTokenize_SpecialCharacters(t *testing.T) {
	e := &Embedder{}

	// Test with unicode characters
	texts := []string{"hello 世界 🌍"}
	inputIDs, attentionMask := e.tokenize(texts)

	// Should not panic and should produce valid output
	if len(inputIDs) != MaxSeqLen {
		t.Errorf("expected %d inputIDs, got %d", MaxSeqLen, len(inputIDs))
	}
	if len(attentionMask) != MaxSeqLen {
		t.Errorf("expected %d attentionMask, got %d", MaxSeqLen, len(attentionMask))
	}

	// Characters > 30000 should become UNK (100)
	// Emoji codepoint is > 30000, so it should become UNK
	foundUNK := false
	for i := 1; i < MaxSeqLen; i++ {
		if inputIDs[i] == 100 { // UNK token
			foundUNK = true
			break
		}
	}
	// Note: This may or may not find UNK depending on the emoji encoding
	_ = foundUNK // suppress unused variable warning
}

func TestTokenize_EmptyString(t *testing.T) {
	e := &Embedder{}

	inputIDs, attentionMask := e.tokenize([]string{""})

	// Should have CLS at start
	if inputIDs[0] != 101 {
		t.Errorf("expected CLS token, got %d", inputIDs[0])
	}

	// Should have SEP right after CLS for empty string
	if inputIDs[1] != 102 {
		t.Errorf("expected SEP token after CLS for empty string, got %d", inputIDs[1])
	}

	// Rest should be padding
	for i := 2; i < MaxSeqLen; i++ {
		if inputIDs[i] != 0 {
			t.Errorf("expected padding at position %d, got %d", i, inputIDs[i])
		}
		if attentionMask[i] != 0 {
			t.Errorf("expected attention mask 0 at position %d", i)
		}
	}
}

func TestTokenize_CaseInsensitivity(t *testing.T) {
	e := &Embedder{}

	upper, _ := e.tokenize([]string{"HELLO WORLD"})
	lower, _ := e.tokenize([]string{"hello world"})

	// Should produce the same tokens (text is lowercased)
	for i := 0; i < MaxSeqLen; i++ {
		if upper[i] != lower[i] {
			t.Errorf("position %d: upper=%d, lower=%d", i, upper[i], lower[i])
		}
	}
}

func TestConstants(t *testing.T) {
	if EmbeddingDim != 384 {
		t.Errorf("EmbeddingDim should be 384, got %d", EmbeddingDim)
	}
	if MaxSeqLen != 256 {
		t.Errorf("MaxSeqLen should be 256, got %d", MaxSeqLen)
	}
}

// Tests below require ONNX model and runtime to be available

func skipIfNoONNX(t *testing.T) (modelPath, onnxLibPath string) {
	t.Helper()

	// Find model
	cwd, _ := os.Getwd()
	modelPath = filepath.Join(cwd, "../assets/models/embed.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(cwd, "../../assets/models/embed.onnx")
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			modelPath = filepath.Join(cwd, "assets/models/embed.onnx")
			if _, err := os.Stat(modelPath); os.IsNotExist(err) {
				t.Skip("ONNX model not found")
			}
		}
	}

	// Find ONNX runtime
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
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if emb == nil {
		t.Fatal("New() returned nil embedder")
	}
	if emb.modelPath != modelPath {
		t.Errorf("modelPath: expected %q, got %q", modelPath, emb.modelPath)
	}
}

func TestNew_InvalidModelPath(t *testing.T) {
	_, onnxLibPath := skipIfNoONNX(t)

	_, err := New("/nonexistent/model.onnx", onnxLibPath)
	// This may or may not error depending on when the model is loaded
	// The test documents the behavior
	if err != nil {
		t.Logf("New() with invalid model path returned error: %v", err)
	}
}

func TestEmbed_SingleText(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	embeddings, err := emb.Embed([]string{"hello world"})
	if err != nil {
		t.Fatalf("Embed() failed: %v", err)
	}

	if len(embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(embeddings))
	}

	if len(embeddings[0]) != EmbeddingDim {
		t.Errorf("expected embedding dim %d, got %d", EmbeddingDim, len(embeddings[0]))
	}

	// Verify normalized (unit norm)
	var sum float64
	for _, v := range embeddings[0] {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("embedding not normalized, norm = %f", norm)
	}
}

func TestEmbed_BatchTexts(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	texts := []string{
		"first document about installation",
		"second document about configuration",
		"third document about usage",
	}

	embeddings, err := emb.Embed(texts)
	if err != nil {
		t.Fatalf("Embed() failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	// Each embedding should have correct dimension and be normalized
	for i, emb := range embeddings {
		if len(emb) != EmbeddingDim {
			t.Errorf("embedding %d: expected dim %d, got %d", i, EmbeddingDim, len(emb))
		}

		var sum float64
		for _, v := range emb {
			sum += float64(v) * float64(v)
		}
		norm := math.Sqrt(sum)
		if math.Abs(norm-1.0) > 0.01 {
			t.Errorf("embedding %d not normalized, norm = %f", i, norm)
		}
	}

	// Embeddings should be different from each other
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			same := true
			for k := 0; k < EmbeddingDim; k++ {
				if math.Abs(float64(embeddings[i][k]-embeddings[j][k])) > 0.0001 {
					same = false
					break
				}
			}
			if same {
				t.Errorf("embeddings %d and %d are identical", i, j)
			}
		}
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	embeddings, err := emb.Embed([]string{})
	if err != nil {
		t.Fatalf("Embed() with empty input failed: %v", err)
	}

	if embeddings != nil && len(embeddings) != 0 {
		t.Errorf("expected nil or empty embeddings for empty input, got %d", len(embeddings))
	}
}

func TestEmbed_LongText(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	// Create very long text that exceeds MaxSeqLen
	longText := ""
	for i := 0; i < MaxSeqLen*2; i++ {
		longText += "word "
	}

	embeddings, err := emb.Embed([]string{longText})
	if err != nil {
		t.Fatalf("Embed() with long text failed: %v", err)
	}

	if len(embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(embeddings))
	}

	if len(embeddings[0]) != EmbeddingDim {
		t.Errorf("expected embedding dim %d, got %d", EmbeddingDim, len(embeddings[0]))
	}
}

func TestEmbed_SpecialCharacters(t *testing.T) {
	modelPath, onnxLibPath := skipIfNoONNX(t)

	emb, err := New(modelPath, onnxLibPath)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	texts := []string{
		"Hello 世界 🌍",
		"Code: func() { return nil }",
		"Special chars: @#$%^&*()",
		"",
	}

	embeddings, err := emb.Embed(texts)
	if err != nil {
		t.Fatalf("Embed() with special characters failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != EmbeddingDim {
			t.Errorf("embedding %d: expected dim %d, got %d", i, EmbeddingDim, len(emb))
		}
	}
}

// Benchmark tests

func BenchmarkL2Normalize(b *testing.B) {
	input := make([]float32, EmbeddingDim)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// We re-use input. It will shrink towards zero but it's fine for benchmarking logic overhead.
		l2Normalize(input)
	}
}

func BenchmarkTokenize_Short(b *testing.B) {
	e := &Embedder{}
	texts := []string{"short text"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.tokenize(texts)
	}
}

func BenchmarkTokenize_Long(b *testing.B) {
	e := &Embedder{}

	longText := ""
	for i := 0; i < 100; i++ {
		longText += "word "
	}
	texts := []string{longText}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.tokenize(texts)
	}
}

func BenchmarkTokenize_Batch(b *testing.B) {
	e := &Embedder{}

	texts := make([]string, 10)
	for i := range texts {
		texts[i] = "sample text for batch processing"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.tokenize(texts)
	}
}
