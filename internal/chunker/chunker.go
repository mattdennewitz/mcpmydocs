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

type headingInfo struct {
	level     int
	text      string
	startByte int
	startLine int
}

// ChunkFile parses markdown and splits by heading sections.
func (c *Chunker) ChunkFile(source []byte) ([]Chunk, error) {
	reader := text.NewReader(source)
	doc := c.md.Parser().Parse(reader)

	headings := collectHeadings(doc, source)

	if len(headings) == 0 {
		return createSingleChunk(source), nil
	}

	return buildChunks(headings, source), nil
}

func collectHeadings(doc ast.Node, source []byte) []headingInfo {
	var headings []headingInfo
	lastEnd := 0

	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if entering {
			if heading, ok := n.(*ast.Heading); ok {
				headings = append(headings, extractHeadingInfo(heading, source, lastEnd))
			}
		}
		lastEnd = updateLastEnd(n, lastEnd)
		return ast.WalkContinue, nil
	})

	return headings
}

func extractHeadingInfo(heading *ast.Heading, source []byte, lastEnd int) headingInfo {
	var headingText bytes.Buffer
	extractText(heading, source, &headingText)

	textStart := lastEnd
	if heading.Lines().Len() > 0 {
		textStart = heading.Lines().At(0).Start
	}

	return headingInfo{
		level:     heading.Level,
		text:      headingText.String(),
		startByte: lastEnd,
		startLine: countLines(source[:textStart]) + 1,
	}
}

func updateLastEnd(n ast.Node, lastEnd int) int {
	if n.Type() != ast.TypeBlock {
		return lastEnd
	}

	lines := n.Lines()
	if lines == nil {
		return lastEnd
	}

	for i := 0; i < lines.Len(); i++ {
		if seg := lines.At(i); seg.Stop > lastEnd {
			lastEnd = seg.Stop
		}
	}
	return lastEnd
}

func createSingleChunk(source []byte) []Chunk {
	content := strings.TrimSpace(string(source))
	if content == "" {
		return nil
	}
	return []Chunk{{
		HeadingPath:  "(root)",
		HeadingLevel: 0,
		Content:      content,
		StartLine:    1,
	}}
}

func buildChunks(headings []headingInfo, source []byte) []Chunk {
	var chunks []Chunk

	if preamble := createPreamble(headings, source); preamble != nil {
		chunks = append(chunks, *preamble)
	}

	var headingStack []stackItem
	for i, h := range headings {
		headingStack = updateHeadingStack(headingStack, h)
		chunk := createChunkFromHeading(headings, i, source, headingStack)
		chunks = append(chunks, chunk)
	}

	return chunks
}

func createPreamble(headings []headingInfo, source []byte) *Chunk {
	if headings[0].startByte <= 0 {
		return nil
	}
	content := strings.TrimSpace(string(source[:headings[0].startByte]))
	if content == "" {
		return nil
	}
	return &Chunk{
		HeadingPath:  "(root)",
		HeadingLevel: 0,
		Content:      content,
		StartLine:    1,
	}
}

func updateHeadingStack(stack []stackItem, h headingInfo) []stackItem {
	for len(stack) > 0 && stack[len(stack)-1].level >= h.level {
		stack = stack[:len(stack)-1]
	}
	return append(stack, stackItem{h.level, h.text})
}

func createChunkFromHeading(headings []headingInfo, idx int, source []byte, stack []stackItem) Chunk {
	h := headings[idx]
	startByte, endByte := getChunkBounds(headings, idx, len(source))
	content := strings.TrimSpace(string(source[startByte:endByte]))

	return Chunk{
		HeadingPath:  buildHeadingPath(stack),
		HeadingLevel: h.level,
		Content:      content,
		StartLine:    h.startLine,
	}
}

func getChunkBounds(headings []headingInfo, idx, sourceLen int) (int, int) {
	startByte := headings[idx].startByte
	endByte := sourceLen

	if idx+1 < len(headings) {
		endByte = headings[idx+1].startByte
	}

	if startByte < 0 {
		startByte = 0
	}
	if endByte < startByte {
		endByte = startByte
	}
	if endByte > sourceLen {
		endByte = sourceLen
	}

	return startByte, endByte
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
