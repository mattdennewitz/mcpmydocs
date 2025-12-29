package chunker

import (
	"bytes"
	"strings"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/text"
)

// Chunk represents a section of a markdown document.
type Chunk struct {
	HeadingPath  string
	HeadingLevel int
	Content      string
	StartLine    int
}

// Chunker parses markdown and splits by heading sections.
type Chunker struct {
	md goldmark.Markdown
}

// New creates a new Chunker.
func New() *Chunker {
	return &Chunker{
		md: goldmark.New(),
	}
}

type stackItem struct {
	level int
	text  string
}

// ChunkFile parses markdown and splits by heading sections.
func (c *Chunker) ChunkFile(source []byte) ([]Chunk, error) {
	reader := text.NewReader(source)
	doc := c.md.Parser().Parse(reader)

	// Collect all headings with their positions
	type headingInfo struct {
		level     int
		text      string
		startByte int
		startLine int
	}

	var headings []headingInfo
	lastEnd := 0

	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		// Only process on entry for capturing start positions
		if entering {
			if heading, ok := n.(*ast.Heading); ok {
				// Extract heading text
				var headingText bytes.Buffer
				extractText(heading, source, &headingText)

				// Calculate StartLine based on the heading text position
				// This ensures the metadata points to the heading itself,
				// not the preceding whitespace/gap included in the chunk content.
				textStart := lastEnd
				if heading.Lines().Len() > 0 {
					textStart = heading.Lines().At(0).Start
				}
				startLine := countLines(source[:textStart]) + 1

				// Start byte is the end of the previous content (including whitespace/markers)
				startByte := lastEnd

				headings = append(headings, headingInfo{
					level:     heading.Level,
					text:      headingText.String(),
					startByte: startByte,
					startLine: startLine,
				})
			}
		}

		// Update lastEnd based on the node's lines (if any)
		// This tracks the progression of the document
		// We do this for both entering and exiting to be safe,
		// though blocks usually define lines on entry.
		if n.Type() == ast.TypeBlock {
			lines := n.Lines()
			if lines != nil {
				for i := 0; i < lines.Len(); i++ {
					seg := lines.At(i)
					if seg.Stop > lastEnd {
						lastEnd = seg.Stop
					}
				}
			}
		}

		return ast.WalkContinue, nil
	})

	// If no headings, treat whole file as one chunk
	if len(headings) == 0 {
		content := strings.TrimSpace(string(source))
		if content == "" {
			return nil, nil
		}
		return []Chunk{{
				HeadingPath:  "(root)",
				HeadingLevel: 0,
				Content:      content,
				StartLine:    1,
			}},
			nil
	}

	var chunks []Chunk

	// 1. Preamble
	if headings[0].startByte > 0 {
		preambleContent := strings.TrimSpace(string(source[:headings[0].startByte]))
		if preambleContent != "" {
			chunks = append(chunks, Chunk{
				HeadingPath:  "(root)",
				HeadingLevel: 0,
				Content:      preambleContent,
				StartLine:    1,
			})
		}
	}

	// 2. Build chunks
	var headingStack []stackItem

	for i, h := range headings {
		// Update heading stack
		for len(headingStack) > 0 && headingStack[len(headingStack)-1].level >= h.level {
			headingStack = headingStack[:len(headingStack)-1]
		}
		headingStack = append(headingStack, stackItem{h.level, h.text})

		headingPath := buildHeadingPath(headingStack)

		startByte := h.startByte
		var endByte int
		if i+1 < len(headings) {
			endByte = headings[i+1].startByte
		} else {
			endByte = len(source)
		}

		// Ensure we don't go out of bounds (though lastEnd logic makes this robust)
		if startByte < 0 {
			startByte = 0
		}
		if endByte < startByte {
			endByte = startByte
		}
		if endByte > len(source) {
			endByte = len(source)
		}

		content := strings.TrimSpace(string(source[startByte:endByte]))

		chunks = append(chunks, Chunk{
			HeadingPath:  headingPath,
			HeadingLevel: h.level,
			Content:      content,
			StartLine:    h.startLine,
		})
	}

	return chunks, nil
}

// extractText recursively extracts all text content from an AST node.
func extractText(n ast.Node, source []byte, buf *bytes.Buffer) {
	for child := n.FirstChild(); child != nil; child = child.NextSibling() {
		switch t := child.(type) {
		case *ast.Text:
			buf.Write(t.Value(source))
			if t.SoftLineBreak() || t.HardLineBreak() {
				buf.WriteByte(' ')
			}
		case *ast.CodeSpan:
			buf.Write(t.Text(source))
		default:
			extractText(child, source, buf)
		}
	}
}

func buildHeadingPath(stack []stackItem) string {
	var parts []string
	for _, h := range stack {
		prefix := strings.Repeat("#", h.level)
		parts = append(parts, prefix+" "+h.text)
	}
	return strings.Join(parts, " > ")
}

func countLines(data []byte) int {
	return bytes.Count(data, []byte("\n"))
}
