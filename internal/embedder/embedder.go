package embedder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	EmbeddingDim = 384 // all-MiniLM-L6-v2 output dimension
	MaxSeqLen    = 256 // Maximum sequence length
)

var (
	ortOnce    sync.Once
	ortInitErr error
)

// Embedder generates embeddings using ONNX runtime.
type Embedder struct {
	modelPath string
	vocab     map[string]int64
	session   *ort.DynamicAdvancedSession
}

// New creates a new Embedder.
func New(modelPath, onnxLibPath string) (*Embedder, error) {
	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(onnxLibPath)
		ortInitErr = ort.InitializeEnvironment()
	})

	if ortInitErr != nil {
		return nil, fmt.Errorf("failed to init onnx environment: %w", ortInitErr)
	}

	// Load tokenizer vocabulary
	tokenizerPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	vocab, err := loadVocab(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Create persistent dynamic session
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Embedder{
		modelPath: modelPath,
		vocab:     vocab,
		session:   session,
	}, nil
}

// Close destroys the ONNX session.
func (e *Embedder) Close() error {
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}

// loadVocab loads the vocabulary from tokenizer.json
func loadVocab(path string) (map[string]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tokenizer struct {
		Model struct {
			Vocab map[string]int64 `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &tokenizer); err != nil {
		return nil, err
	}

	if len(tokenizer.Model.Vocab) == 0 {
		return nil, fmt.Errorf("vocabulary is empty; check tokenizer.json structure (expected model.vocab)")
	}

	return tokenizer.Model.Vocab, nil
}

// Embed generates embeddings for a batch of texts.
func (e *Embedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	batchSize := int64(len(texts))

	// Simple tokenization: convert to token IDs
	inputIDs, attentionMask := e.tokenize(texts)
	seqLen := int64(MaxSeqLen)

	// Create input tensors
	inputShape := ort.NewShape(batchSize, seqLen)

	inputIDsTensor, err := ort.NewTensor(inputShape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	// Token type IDs (all zeros for single sequence)
	tokenTypeIDs := make([]int64, batchSize*seqLen)
	tokenTypeIDsTensor, err := ort.NewTensor(inputShape, tokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDsTensor.Destroy()

	// Create output tensor
	// The model outputs [batch_size, seq_len, hidden_size]
	outputShape := ort.NewShape(batchSize, seqLen, int64(EmbeddingDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = e.session.Run(
		[]ort.ArbitraryTensor{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Mean pooling over sequence dimension
	outputData := outputTensor.GetData()
	embeddings := make([][]float32, batchSize)

	for b := int64(0); b < batchSize; b++ {
		embedding := make([]float32, EmbeddingDim)
		validTokens := float32(0)

		for s := int64(0); s < seqLen; s++ {
			// Only pool over non-padded tokens
			if attentionMask[b*seqLen+s] == 1 {
				validTokens++
				for d := 0; d < EmbeddingDim; d++ {
					idx := b*seqLen*int64(EmbeddingDim) + s*int64(EmbeddingDim) + int64(d)
					embedding[d] += outputData[idx]
				}
			}
		}

		// Average
		if validTokens > 0 {
			for d := 0; d < EmbeddingDim; d++ {
				embedding[d] /= validTokens
			}
		}

		// L2 normalize (in-place)
		l2Normalize(embedding)
		embeddings[b] = embedding
	}

	return embeddings, nil
}

// tokenize performs WordPiece tokenization using the loaded vocabulary.
func (e *Embedder) tokenize(texts []string) ([]int64, []int64) {
	batchSize := len(texts)
	inputIDs := make([]int64, batchSize*MaxSeqLen)
	attentionMask := make([]int64, batchSize*MaxSeqLen)

	const (
		clsToken = 101
		sepToken = 102
		padToken = 0
		unkToken = 100
	)

	for b, text := range texts {
		offset := b * MaxSeqLen

		// Add CLS token
		inputIDs[offset] = clsToken
		attentionMask[offset] = 1
		pos := 1

		// Normalize and tokenize
		text = strings.ToLower(text)
		words := tokenizeText(text)

		for _, word := range words {
			if pos >= MaxSeqLen-1 {
				break
			}

			// WordPiece tokenization for this word
			subTokens := e.wordPieceTokenize(word)
			for _, tokenID := range subTokens {
				if pos >= MaxSeqLen-1 {
					break
				}
				inputIDs[offset+pos] = tokenID
				attentionMask[offset+pos] = 1
				pos++
			}
		}

		// Add SEP token
		if pos < MaxSeqLen {
			inputIDs[offset+pos] = sepToken
			attentionMask[offset+pos] = 1
			pos++
		}

		// Pad remaining
		for ; pos < MaxSeqLen; pos++ {
			inputIDs[offset+pos] = padToken
			attentionMask[offset+pos] = 0
		}
	}

	return inputIDs, attentionMask
}

// tokenizeText splits text into words, handling punctuation and CJK characters.
func tokenizeText(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		// Skip control characters
		if unicode.IsControl(r) {
			continue
		}

		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) || isCJK(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			words = append(words, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		(r >= 0x3040 && r <= 0x30ff) || // Hiragana and Katakana
		(r >= 0xac00 && r <= 0xd7af)    // Hangul
}

// wordPieceTokenize applies WordPiece algorithm to a single word.
func (e *Embedder) wordPieceTokenize(word string) []int64 {
	const (
		unkToken   = 100
		maxWordLen = 100 // Limit word length to prevent DoS (O(N^2) complexity)
	)

	// Check if whole word is in vocab
	if id, ok := e.vocab[word]; ok {
		return []int64{id}
	}

	// If word is too long, treat as UNK to avoid expensive processing
	if len(word) > maxWordLen {
		return []int64{unkToken}
	}

	var tokens []int64
	start := 0
	wordRunes := []rune(word)

	for start < len(wordRunes) {
		end := len(wordRunes)
		found := false

		for end > start {
			substr := string(wordRunes[start:end])
			if start > 0 {
				substr = "##" + substr
			}

			if id, ok := e.vocab[substr]; ok {
				tokens = append(tokens, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			// Character not in vocab, use UNK
			tokens = append(tokens, unkToken)
			start++
		}
	}

	return tokens
}

func l2Normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	norm := float32(1.0 / math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= norm
	}
}
