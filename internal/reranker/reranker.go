package reranker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"

	ort "github.com/yalue/onnxruntime_go"

	"mcpmydocs/internal/store"
)

const (
	MaxSeqLen = 512 // Cross-encoder max sequence length
)

// Reranker scores query-document pairs using a cross-encoder model.
type Reranker struct {
	modelPath string
	vocab     map[string]int64
	session   *ort.DynamicAdvancedSession
}

// ScoredResult wraps a search result with its cross-encoder relevance score.
type ScoredResult struct {
	Result store.SearchResult
	Score  float32 // higher = more relevant
}

var (
	ortOnce sync.Once
)

// New creates a new Reranker.
func New(modelPath, onnxLibPath string) (*Reranker, error) {
	// Initialize ONNX environment if not already done (embedder may have initialized it)
	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(onnxLibPath)
		// Ignore error - embedder may have already initialized
		_ = ort.InitializeEnvironment()
	})

	// Load tokenizer vocabulary (same as embedder)
	tokenizerPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	vocab, err := loadVocab(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Create session for cross-encoder (BGE reranker uses only input_ids and attention_mask)
	// Output is typically "logits" or similar - try common names
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create reranker session: %w", err)
	}

	return &Reranker{
		modelPath: modelPath,
		vocab:     vocab,
		session:   session,
	}, nil
}

// Close destroys the ONNX session.
func (r *Reranker) Close() error {
	if r.session != nil {
		return r.session.Destroy()
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
		return nil, fmt.Errorf("vocabulary is empty; check tokenizer.json structure")
	}

	return tokenizer.Model.Vocab, nil
}

// Rerank scores and reorders search results based on relevance to the query.
// Returns results sorted by cross-encoder score (highest first).
func (r *Reranker) Rerank(query string, results []store.SearchResult) ([]ScoredResult, error) {
	if len(results) == 0 {
		return nil, nil
	}

	batchSize := int64(len(results))
	seqLen := int64(MaxSeqLen)

	// Tokenize all query-passage pairs
	inputIDs, attentionMask := r.tokenizePairs(query, results)

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

	// Create output tensor - cross-encoder outputs [batch_size, 1] logits
	outputShape := ort.NewShape(batchSize, 1)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = r.session.Run(
		[]ort.ArbitraryTensor{inputIDsTensor, attentionMaskTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to run reranker inference: %w", err)
	}

	// Extract scores
	outputData := outputTensor.GetData()
	scoredResults := make([]ScoredResult, len(results))

	for i, result := range results {
		scoredResults[i] = ScoredResult{
			Result: result,
			Score:  outputData[i],
		}
	}

	// Sort by score descending (higher = more relevant)
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].Score > scoredResults[j].Score
	})

	return scoredResults, nil
}

// tokenizePairs tokenizes query-passage pairs for cross-encoder.
// Format: [CLS] query [SEP] passage [SEP]
func (r *Reranker) tokenizePairs(query string, results []store.SearchResult) ([]int64, []int64) {
	batchSize := len(results)
	inputIDs := make([]int64, batchSize*MaxSeqLen)
	attentionMask := make([]int64, batchSize*MaxSeqLen)

	const (
		clsToken = 101
		sepToken = 102
		padToken = 0
	)

	// Tokenize query once
	queryTokens := r.tokenizeText(strings.ToLower(query))

	// Reserve space for query: [CLS] query [SEP] = len(queryTokens) + 2
	// Reserve at least 64 tokens for passage
	maxQueryTokens := MaxSeqLen - 64 - 2 // leave room for passage
	if len(queryTokens) > maxQueryTokens {
		queryTokens = queryTokens[:maxQueryTokens]
	}

	for b, result := range results {
		offset := b * MaxSeqLen
		pos := 0

		// [CLS]
		inputIDs[offset+pos] = clsToken
		attentionMask[offset+pos] = 1
		pos++

		// Query tokens
		for _, tokenID := range queryTokens {
			if pos >= MaxSeqLen-2 {
				break
			}
			inputIDs[offset+pos] = tokenID
			attentionMask[offset+pos] = 1
			pos++
		}

		// [SEP] after query
		if pos < MaxSeqLen-1 {
			inputIDs[offset+pos] = sepToken
			attentionMask[offset+pos] = 1
			pos++
		}

		// Passage tokens
		passageTokens := r.tokenizeText(strings.ToLower(result.Content))
		for _, tokenID := range passageTokens {
			if pos >= MaxSeqLen-1 {
				break
			}
			inputIDs[offset+pos] = tokenID
			attentionMask[offset+pos] = 1
			pos++
		}

		// [SEP] after passage
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

// tokenizeText converts text to token IDs using WordPiece tokenization.
func (r *Reranker) tokenizeText(text string) []int64 {
	words := splitWords(text)
	var tokens []int64

	for _, word := range words {
		subTokens := r.wordPieceTokenize(word)
		tokens = append(tokens, subTokens...)
	}

	return tokens
}

// splitWords splits text into words, handling punctuation.
func splitWords(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsControl(r) {
			continue
		}

		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) {
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

// wordPieceTokenize applies WordPiece algorithm to a single word.
func (r *Reranker) wordPieceTokenize(word string) []int64 {
	const (
		unkToken   = 100
		maxWordLen = 100
	)

	// Check if whole word is in vocab
	if id, ok := r.vocab[word]; ok {
		return []int64{id}
	}

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

			if id, ok := r.vocab[substr]; ok {
				tokens = append(tokens, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			tokens = append(tokens, unkToken)
			start++
		}
	}

	return tokens
}
