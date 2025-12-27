package chunker

import (
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	c := New()
	if c == nil {
		t.Fatal("New() returned nil")
	}
	if c.md == nil {
		t.Fatal("Chunker.md is nil")
	}
}

func TestChunkFile_EmptyInput(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"empty string", ""},
		{"whitespace only", "   \n\t\n   "},
		{"newlines only", "\n\n\n"},
	}

	c := New()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := c.ChunkFile([]byte(tt.input))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if chunks != nil {
				t.Errorf("expected nil chunks for empty input, got %d chunks", len(chunks))
			}
		})
	}
}

func TestChunkFile_NoHeadings(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		content string
	}{
		{
			name:    "plain text",
			input:   "Just some plain text without any headings.",
			content: "Just some plain text without any headings.",
		},
		{
			name:    "multiple paragraphs",
			input:   "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
			content: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
		},
		{
			name:    "with code block",
			input:   "```go\nfunc main() {}\n```",
			content: "```go\nfunc main() {}\n```",
		},
		{
			name:    "with list",
			input:   "- Item 1\n- Item 2\n- Item 3",
			content: "- Item 1\n- Item 2\n- Item 3",
		},
	}

	c := New()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := c.ChunkFile([]byte(tt.input))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(chunks) != 1 {
				t.Fatalf("expected 1 chunk, got %d", len(chunks))
			}
			if chunks[0].HeadingPath != "(root)" {
				t.Errorf("expected HeadingPath '(root)', got %q", chunks[0].HeadingPath)
			}
			if chunks[0].HeadingLevel != 0 {
				t.Errorf("expected HeadingLevel 0, got %d", chunks[0].HeadingLevel)
			}
			if chunks[0].StartLine != 1 {
				t.Errorf("expected StartLine 1, got %d", chunks[0].StartLine)
			}
		})
	}
}

func TestChunkFile_SingleHeading(t *testing.T) {
	input := `# Hello World

This is content under the heading.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}

	chunk := chunks[0]
	if chunk.HeadingPath != "# Hello World" {
		t.Errorf("expected HeadingPath '# Hello World', got %q", chunk.HeadingPath)
	}
	if chunk.HeadingLevel != 1 {
		t.Errorf("expected HeadingLevel 1, got %d", chunk.HeadingLevel)
	}
	if chunk.StartLine != 1 {
		t.Errorf("expected StartLine 1, got %d", chunk.StartLine)
	}
	if !strings.Contains(chunk.Content, "Hello World") {
		t.Errorf("content should contain heading text")
	}
	if !strings.Contains(chunk.Content, "content under the heading") {
		t.Errorf("content should contain paragraph text")
	}
}

func TestChunkFile_MultipleSameLevelHeadings(t *testing.T) {
	input := `# First

Content one.

# Second

Content two.

# Third

Content three.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	expectedPaths := []string{"# First", "# Second", "# Third"}
	for i, expected := range expectedPaths {
		if chunks[i].HeadingPath != expected {
			t.Errorf("chunk %d: expected path %q, got %q", i, expected, chunks[i].HeadingPath)
		}
		if chunks[i].HeadingLevel != 1 {
			t.Errorf("chunk %d: expected level 1, got %d", i, chunks[i].HeadingLevel)
		}
	}
}

func TestChunkFile_NestedHeadings(t *testing.T) {
	input := `# Parent

Parent content.

## Child

Child content.

### Grandchild

Grandchild content.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	expectedPaths := []string{
		"# Parent",
		"# Parent > ## Child",
		"# Parent > ## Child > ### Grandchild",
	}
	expectedLevels := []int{1, 2, 3}

	for i := range chunks {
		if chunks[i].HeadingPath != expectedPaths[i] {
			t.Errorf("chunk %d: expected path %q, got %q", i, expectedPaths[i], chunks[i].HeadingPath)
		}
		if chunks[i].HeadingLevel != expectedLevels[i] {
			t.Errorf("chunk %d: expected level %d, got %d", i, expectedLevels[i], chunks[i].HeadingLevel)
		}
	}
}

func TestChunkFile_HeadingHierarchyReset(t *testing.T) {
	input := `# First

Content.

## Nested

More content.

# Second

Back to level 1.

## Another Nested

Under second.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d", len(chunks))
	}

	expectedPaths := []string{
		"# First",
		"# First > ## Nested",
		"# Second",
		"# Second > ## Another Nested",
	}

	for i, expected := range expectedPaths {
		if chunks[i].HeadingPath != expected {
			t.Errorf("chunk %d: expected path %q, got %q", i, expected, chunks[i].HeadingPath)
		}
	}
}

func TestChunkFile_DeepHierarchy(t *testing.T) {
	input := `# H1

## H2

### H3

#### H4

##### H5

###### H6

Content at level 6.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 6 {
		t.Fatalf("expected 6 chunks, got %d", len(chunks))
	}

	// Check the deepest level has full path
	lastChunk := chunks[5]
	if lastChunk.HeadingLevel != 6 {
		t.Errorf("expected level 6, got %d", lastChunk.HeadingLevel)
	}
	expectedPath := "# H1 > ## H2 > ### H3 > #### H4 > ##### H5 > ###### H6"
	if lastChunk.HeadingPath != expectedPath {
		t.Errorf("expected path %q, got %q", expectedPath, lastChunk.HeadingPath)
	}
}

func TestChunkFile_StartLineAccuracy(t *testing.T) {
	input := `# First

Content on lines 3-4.

# Second

Content on lines 7-8.

# Third

Content on lines 11-12.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	expectedLines := []int{1, 5, 9}
	for i, expected := range expectedLines {
		if chunks[i].StartLine != expected {
			t.Errorf("chunk %d: expected StartLine %d, got %d", i, expected, chunks[i].StartLine)
		}
	}
}

func TestChunkFile_ContentBoundaries(t *testing.T) {
	input := `# First

First content paragraph.

More first content.

# Second

Second content only.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	// First chunk should contain both paragraphs
	if !strings.Contains(chunks[0].Content, "First content paragraph") {
		t.Error("first chunk missing first paragraph")
	}
	if !strings.Contains(chunks[0].Content, "More first content") {
		t.Error("first chunk missing second paragraph")
	}

	// First chunk should NOT contain second heading content
	if strings.Contains(chunks[0].Content, "Second content only") {
		t.Error("first chunk incorrectly contains second chunk content")
	}

	// Second chunk should only have its content
	if !strings.Contains(chunks[1].Content, "Second content only") {
		t.Error("second chunk missing its content")
	}
	if strings.Contains(chunks[1].Content, "More first content") {
		t.Error("second chunk incorrectly contains first chunk content")
	}
}

func TestChunkFile_CodeBlocksPreserved(t *testing.T) {
	input := "# Installation\n\nRun this:\n\n```bash\nnpm install foo\necho \"done\"\n```\n\nThat's it."

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}

	content := chunks[0].Content
	if !strings.Contains(content, "```bash") {
		t.Error("code block opening not preserved")
	}
	if !strings.Contains(content, "npm install foo") {
		t.Error("code block content not preserved")
	}
	if !strings.Contains(content, "```") {
		t.Error("code block closing not preserved")
	}
}

func TestChunkFile_StyledHeadings(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		expectedText string
	}{
		{
			name:         "link in heading",
			input:        "# Check out [this link](https://example.com)\n\nContent.",
			expectedText: "Check out this link",
		},
		{
			name:         "bold in heading",
			input:        "# The **important** section\n\nContent.",
			expectedText: "The important section",
		},
		{
			name:         "italic in heading",
			input:        "# An *emphasized* word\n\nContent.",
			expectedText: "An emphasized word",
		},
		{
			name:         "code span in heading",
			input:        "# Using `foo()` method\n\nContent.",
			expectedText: "foo()",
		},
		{
			name:         "mixed styling",
			input:        "# The **bold** and [linked](url) heading\n\nContent.",
			expectedText: "bold",
		},
	}

	c := New()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := c.ChunkFile([]byte(tt.input))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(chunks) != 1 {
				t.Fatalf("expected 1 chunk, got %d", len(chunks))
			}
			if !strings.Contains(chunks[0].HeadingPath, tt.expectedText) {
				t.Errorf("expected heading path to contain %q, got %q", tt.expectedText, chunks[0].HeadingPath)
			}
		})
	}
}

func TestChunkFile_SpecialCharactersInHeadings(t *testing.T) {
	input := `# API: v2.0 (beta)

Content.

## GET /users/:id

More content.

### Query Parameters & Options

Even more content.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	if !strings.Contains(chunks[0].HeadingPath, "API: v2.0 (beta)") {
		t.Errorf("special characters not preserved in heading: %q", chunks[0].HeadingPath)
	}
	if !strings.Contains(chunks[1].HeadingPath, "GET /users/:id") {
		t.Errorf("path characters not preserved in heading: %q", chunks[1].HeadingPath)
	}
	if !strings.Contains(chunks[2].HeadingPath, "Query Parameters & Options") {
		t.Errorf("ampersand not preserved in heading: %q", chunks[2].HeadingPath)
	}
}

func TestChunkFile_HeadingAtEndOfFile(t *testing.T) {
	input := `# Main Content

Some text here.

## Empty Section`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	// Last chunk should still exist even with no content after heading
	lastChunk := chunks[1]
	if lastChunk.HeadingPath != "# Main Content > ## Empty Section" {
		t.Errorf("unexpected heading path: %q", lastChunk.HeadingPath)
	}
	// Content should just be the heading itself (trimmed)
	if !strings.Contains(lastChunk.Content, "Empty Section") {
		t.Errorf("heading text not in content: %q", lastChunk.Content)
	}
}

func TestChunkFile_SkippedHeadingLevels(t *testing.T) {
	// Jump from H1 to H3 (skipping H2)
	input := `# Top Level

Content.

### Skipped to H3

More content.`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	// The path building logic should handle skipped levels
	// Based on the implementation, it pops until len(stack) >= level
	// So H3 after H1 should work, but path might be unusual
	if chunks[1].HeadingLevel != 3 {
		t.Errorf("expected level 3, got %d", chunks[1].HeadingLevel)
	}
}

func TestChunkFile_EmptyHeadings(t *testing.T) {
	// This tests the edge case where empty headings (just "#" with no text)
	// could cause invalid byte positions leading to slice bounds errors
	input := `# Requirements

Some content here.

# Decision

We decided something.

-----

#

#

#

# Questions

1. First question
`

	c := New()
	chunks, err := c.ChunkFile([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should not panic and should produce chunks
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}

	// Verify we got the meaningful headings
	foundRequirements := false
	foundDecision := false
	foundQuestions := false
	for _, chunk := range chunks {
		if strings.Contains(chunk.HeadingPath, "Requirements") {
			foundRequirements = true
		}
		if strings.Contains(chunk.HeadingPath, "Decision") {
			foundDecision = true
		}
		if strings.Contains(chunk.HeadingPath, "Questions") {
			foundQuestions = true
		}
	}

	if !foundRequirements {
		t.Error("expected 'Requirements' heading")
	}
	if !foundDecision {
		t.Error("expected 'Decision' heading")
	}
	if !foundQuestions {
		t.Error("expected 'Questions' heading")
	}
}

func TestChunkFile_LargeFile(t *testing.T) {
	// Generate a large markdown file
	var builder strings.Builder
	for i := 1; i <= 100; i++ {
		builder.WriteString("# Section ")
		builder.WriteString(strings.Repeat("A", 10))
		builder.WriteString("\n\n")
		builder.WriteString(strings.Repeat("Content paragraph. ", 50))
		builder.WriteString("\n\n")
	}

	c := New()
	chunks, err := c.ChunkFile([]byte(builder.String()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 100 {
		t.Errorf("expected 100 chunks, got %d", len(chunks))
	}
}

func TestBuildHeadingPath(t *testing.T) {
	tests := []struct {
		name     string
		stack    []stackItem
		expected string
	}{
		{
			name:     "single item",
			stack:    []stackItem{{level: 1, text: "Hello"}},
			expected: "# Hello",
		},
		{
			name:     "two items",
			stack:    []stackItem{{level: 1, text: "Parent"}, {level: 2, text: "Child"}},
			expected: "# Parent > ## Child",
		},
		{
			name:     "three items",
			stack:    []stackItem{{level: 1, text: "A"}, {level: 2, text: "B"}, {level: 3, text: "C"}},
			expected: "# A > ## B > ### C",
		},
		{
			name:     "skipped level",
			stack:    []stackItem{{level: 1, text: "H1"}, {level: 3, text: "H3"}},
			expected: "# H1 > ### H3",
		},
		{
			name:     "empty stack",
			stack:    []stackItem{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildHeadingPath(tt.stack)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestCountLines(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int
	}{
		{"empty", "", 0},
		{"no newlines", "hello", 0},
		{"one newline", "hello\n", 1},
		{"multiple newlines", "a\nb\nc\n", 3},
		{"only newlines", "\n\n\n", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := countLines([]byte(tt.input))
			if result != tt.expected {
				t.Errorf("expected %d, got %d", tt.expected, result)
			}
		})
	}
}

// Benchmark tests
func BenchmarkChunkFile_Small(b *testing.B) {
	input := []byte(`# Title

Content here.

## Section 1

More content.

## Section 2

Even more content.`)

	c := New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.ChunkFile(input)
	}
}

func BenchmarkChunkFile_Large(b *testing.B) {
	var builder strings.Builder
	for i := 0; i < 100; i++ {
		builder.WriteString("# Section\n\n")
		builder.WriteString(strings.Repeat("Content paragraph. ", 100))
		builder.WriteString("\n\n")
	}
	input := []byte(builder.String())

	c := New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.ChunkFile(input)
	}
}
