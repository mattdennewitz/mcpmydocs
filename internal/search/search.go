package search

import (
	"context"
	"fmt"
	"time"

	"github.com/mattdennewitz/mcpmydocs/internal/embedder"
	"github.com/mattdennewitz/mcpmydocs/internal/logger"
	"github.com/mattdennewitz/mcpmydocs/internal/reranker"
	"github.com/mattdennewitz/mcpmydocs/internal/store"
)

// Limits for search parameters.
const (
	DefaultLimit      = 5
	MaxLimit          = 20
	MinLimit          = 1
	DefaultCandidates = 50
	MaxCandidates     = 100
	MinCandidates     = 1
)

// Service handles semantic search with optional reranking.
type Service struct {
	store    *store.Store
	embedder *embedder.Embedder
	reranker *reranker.Reranker // nil if not available
}

// New creates a search service.
func New(st *store.Store, emb *embedder.Embedder, rr *reranker.Reranker) *Service {
	return &Service{
		store:    st,
		embedder: emb,
		reranker: rr,
	}
}

// HasReranker returns true if reranking is available.
func (s *Service) HasReranker() bool {
	return s.reranker != nil
}

// Params configures a search request.
type Params struct {
	Query      string
	Limit      int   // Final results to return (default: 5, max: 20)
	Candidates int   // Candidates for reranking (default: 50, max: 100)
	Rerank     *bool // nil=auto, true=force, false=skip
}

// Result holds search results.
type Result struct {
	Query    string
	Items    []Item
	Reranked bool
}

// Item represents a single search result.
type Item struct {
	FilePath    string
	Title       string
	HeadingPath string
	Content     string
	StartLine   int
	Score       float32 // Similarity (0-1) or rerank score
}

// Search executes a semantic search with optional reranking.
func (s *Service) Search(ctx context.Context, p Params) (*Result, error) {
	if p.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder not initialized")
	}

	// Apply defaults and clamp
	limit := clamp(p.Limit, MinLimit, MaxLimit, DefaultLimit)
	candidates := clamp(p.Candidates, MinCandidates, MaxCandidates, DefaultCandidates)

	// Determine rerank mode
	useRerank := s.reranker != nil
	if p.Rerank != nil {
		useRerank = *p.Rerank && s.reranker != nil
	}

	// Fetch more if reranking
	fetchCount := limit
	if useRerank {
		fetchCount = candidates
	}

	// Embed query
	embedStart := time.Now()
	embeddings, err := s.embedder.Embed([]string{p.Query})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding generated")
	}
	logger.Debug("query embedded", "duration", time.Since(embedStart))

	// Vector search
	searchStart := time.Now()
	vectorResults, err := s.store.Search(ctx, embeddings[0], fetchCount)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	logger.Debug("vector search", "results", len(vectorResults), "duration", time.Since(searchStart))

	if len(vectorResults) == 0 {
		return &Result{Query: p.Query}, nil
	}

	// Rerank or convert directly
	if useRerank {
		rerankStart := time.Now()
		reranked, err := s.reranker.Rerank(p.Query, vectorResults)
		if err != nil {
			logger.Warn("reranking failed, using vector results", "error", err)
			return s.vectorResult(p.Query, vectorResults, limit, false), nil
		}
		logger.Debug("reranked", "results", len(reranked), "duration", time.Since(rerankStart))
		return s.rerankedResult(p.Query, reranked, limit), nil
	}

	return s.vectorResult(p.Query, vectorResults, limit, false), nil
}

func (s *Service) vectorResult(query string, results []store.SearchResult, limit int, reranked bool) *Result {
	if limit > len(results) {
		limit = len(results)
	}

	items := make([]Item, limit)
	for i := 0; i < limit; i++ {
		r := results[i]
		items[i] = Item{
			FilePath:    r.FilePath,
			Title:       r.Title,
			HeadingPath: r.HeadingPath,
			Content:     r.Content,
			StartLine:   r.StartLine,
			Score:       float32(1.0 - r.Distance), // Convert distance to similarity
		}
	}

	return &Result{
		Query:    query,
		Items:    items,
		Reranked: reranked,
	}
}

func (s *Service) rerankedResult(query string, results []reranker.ScoredResult, limit int) *Result {
	if limit > len(results) {
		limit = len(results)
	}

	items := make([]Item, limit)
	for i := 0; i < limit; i++ {
		r := results[i]
		items[i] = Item{
			FilePath:    r.Result.FilePath,
			Title:       r.Result.Title,
			HeadingPath: r.Result.HeadingPath,
			Content:     r.Result.Content,
			StartLine:   r.Result.StartLine,
			Score:       r.Score,
		}
	}

	return &Result{
		Query:    query,
		Items:    items,
		Reranked: true,
	}
}

func clamp(val, min, max, defaultVal int) int {
	if val == 0 {
		return defaultVal
	}
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}
