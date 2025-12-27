package store

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"strings"

	_ "github.com/marcboeker/go-duckdb"
)

const EmbeddingDim = 384

type Store struct {
	db *sql.DB
}

// New creates a new Store and initializes the database schema.
func New(dbPath string) (*Store, error) {
	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open duckdb: %w", err)
	}

	store := &Store{db: db}
	if err := store.initialize(); err != nil {
		db.Close()
		return nil, err
	}

	return store, nil
}

func (s *Store) initialize() error {
	ctx := context.Background()

	queries := []string{
		// Install and load VSS extension
		"INSTALL vss",
		"LOAD vss",
		"SET hnsw_enable_experimental_persistence = true",

		// Create sequences for auto-increment
		`CREATE SEQUENCE IF NOT EXISTS documents_id_seq`,
		`CREATE SEQUENCE IF NOT EXISTS chunks_id_seq`,

		// Documents table
		`CREATE TABLE IF NOT EXISTS documents (
			id INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
			file_path VARCHAR NOT NULL UNIQUE,
			file_hash VARCHAR NOT NULL,
			title VARCHAR,
			indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		// Chunks table with embeddings
		`CREATE TABLE IF NOT EXISTS chunks (
			id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
			document_id INTEGER NOT NULL,
			heading_path VARCHAR NOT NULL,
			heading_level INTEGER NOT NULL,
			content VARCHAR NOT NULL,
			start_line INTEGER,
			embedding FLOAT[384],
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		// Index for document lookups
		`CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id)`,
	}

	for _, q := range queries {
		if _, err := s.db.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("failed to execute %q: %w", q, err)
		}
	}

	// Create HNSW index (ignore error if exists)
	s.db.ExecContext(ctx, `CREATE INDEX chunks_embedding_idx ON chunks USING HNSW (embedding) WITH (metric = 'cosine')`)

	return nil
}

// FileUnchanged checks if a file has already been indexed with the same hash.
func (s *Store) FileUnchanged(ctx context.Context, filePath, hash string) bool {
	var count int
	err := s.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM documents WHERE file_path = ? AND file_hash = ?",
		filePath, hash,
	).Scan(&count)
	return err == nil && count > 0
}

// DeleteDocumentByPath deletes a document and its chunks by file path.
func (s *Store) DeleteDocumentByPath(ctx context.Context, filePath string) error {
	// Get document ID first
	var docID int
	err := s.db.QueryRowContext(ctx,
		"SELECT id FROM documents WHERE file_path = ?", filePath,
	).Scan(&docID)
	if err == sql.ErrNoRows {
		return nil
	}
	if err != nil {
		return err
	}

	// Delete chunks then document
	if _, err := s.db.ExecContext(ctx, "DELETE FROM chunks WHERE document_id = ?", docID); err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, "DELETE FROM documents WHERE id = ?", docID)
	return err
}

// InsertDocument inserts a new document and returns its ID.
func (s *Store) InsertDocument(ctx context.Context, filePath, hash, title string) (int, error) {
	var id int
	err := s.db.QueryRowContext(ctx,
		`INSERT INTO documents (file_path, file_hash, title) VALUES (?, ?, ?) RETURNING id`,
		filePath, hash, title,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	return id, nil
}

// Chunk represents a section of a markdown document.
type Chunk struct {
	HeadingPath  string
	HeadingLevel int
	Content      string
	StartLine    int
}

// InsertChunk inserts a chunk with its embedding.
func (s *Store) InsertChunk(ctx context.Context, docID int, chunk Chunk, embedding []float32) error {
	arrayStr := floatSliceToArrayString(embedding)

	query := fmt.Sprintf(`
		INSERT INTO chunks (document_id, heading_path, heading_level, content, start_line, embedding)
		VALUES (?, ?, ?, ?, ?, %s::FLOAT[384])
	`, arrayStr)

	_, err := s.db.ExecContext(ctx, query,
		docID, chunk.HeadingPath, chunk.HeadingLevel, chunk.Content, chunk.StartLine,
	)
	return err
}

// floatSliceToArrayString converts []float32 to DuckDB array literal.
// NaN and Inf values are sanitized to 0 to prevent SQL issues.
func floatSliceToArrayString(v []float32) string {
	if len(v) == 0 {
		return "NULL"
	}

	var buf strings.Builder
	buf.WriteString("[")
	for i, f := range v {
		if i > 0 {
			buf.WriteString(",")
		}
		// Sanitize special float values that could cause SQL issues
		f64 := float64(f)
		if math.IsNaN(f64) || math.IsInf(f64, 0) {
			buf.WriteString("0")
		} else {
			buf.WriteString(fmt.Sprintf("%g", f))
		}
	}
	buf.WriteString("]")
	return buf.String()
}

// SearchResult represents a search result with similarity score.
type SearchResult struct {
	ChunkID      int
	FilePath     string
	Title        string
	HeadingPath  string
	Content      string
	StartLine    int
	Distance     float64
}

// Search finds chunks similar to the query embedding.
func (s *Store) Search(ctx context.Context, queryEmbedding []float32, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("limit must be positive")
	}
	if limit > 1000 {
		limit = 1000 // Hard cap at store level
	}

	arrayStr := floatSliceToArrayString(queryEmbedding)

	query := fmt.Sprintf(`
		SELECT
			c.id,
			d.file_path,
			d.title,
			c.heading_path,
			c.content,
			c.start_line,
			array_cosine_distance(c.embedding, %s::FLOAT[384]) as distance
		FROM chunks c
		JOIN documents d ON c.document_id = d.id
		ORDER BY distance ASC
		LIMIT ?
	`, arrayStr)

	rows, err := s.db.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("search query failed: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(&r.ChunkID, &r.FilePath, &r.Title, &r.HeadingPath, &r.Content, &r.StartLine, &r.Distance); err != nil {
			return nil, fmt.Errorf("failed to scan result: %w", err)
		}
		results = append(results, r)
	}

	return results, rows.Err()
}

// Document represents an indexed document.
type Document struct {
	ID       int
	FilePath string
	Title    string
}

// ListDocuments returns all indexed documents.
func (s *Store) ListDocuments(ctx context.Context) ([]Document, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, file_path, title FROM documents ORDER BY title
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.FilePath, &d.Title); err != nil {
			return nil, err
		}
		docs = append(docs, d)
	}

	return docs, rows.Err()
}

// Close closes the database connection.
func (s *Store) Close() error {
	return s.db.Close()
}
