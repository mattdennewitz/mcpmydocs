# mcpmydocs

A local-first semantic search engine for Markdown documentation. Indexes your docs, generates embeddings using ONNX models, stores them in DuckDB, and exposes search via CLI or MCP server for AI agents like Claude Code.

## Features

- **Semantic search** - Find documents by meaning, not just keywords
- **Cross-encoder reranking** - Two-stage retrieval for improved relevance
- **Local-first** - All processing happens on your machine, no API calls
- **Fast** - DuckDB with HNSW vector indexing for millisecond queries
- **MCP server** - Integrate with Claude Code or other MCP-compatible AI tools
- **Incremental indexing** - Only re-indexes changed files

## Requirements

- macOS Apple Silicon, Linux x64, or Linux ARM64 (pre-built binaries)
- macOS Intel (build from source)
- ONNX Runtime library
- Embedding model files

## Installation

### Quick install (macOS Apple Silicon / Linux)

```bash
curl -sSL https://raw.githubusercontent.com/mattdennewitz/mcpmydocs/main/install.sh | bash
```

This will:
- Download the latest release for your platform
- Install the binary to `~/.local/bin`
- Download the embedding and reranker models to `~/.local/share/mcpmydocs`

**Note:** You still need ONNX Runtime installed (see below).

For macOS Intel, see [Building from source](#building-from-source).

### Manual installation

If you have a pre-built binary, you need to install the dependencies manually:

#### 1. Install ONNX Runtime

**macOS (Homebrew):**
```bash
brew install onnxruntime
```

**Linux (Ubuntu/Debian):**
```bash
# Download from https://github.com/microsoft/onnxruntime/releases
# Extract and copy libonnxruntime.so to /usr/local/lib/
sudo ldconfig
```

Alternatively, place the library in a `lib/` directory next to the binary, or set the `ONNX_LIBRARY_PATH` environment variable.

#### 2. Download the models

Download the model files and place them in an `assets/models/` directory next to the binary:

```bash
mkdir -p assets/models

# Download embedding model (~90MB)
curl -L -o assets/models/embed.onnx \
  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

# Download tokenizer
curl -L -o assets/models/tokenizer.json \
  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

# Download reranker model (~90MB, optional but recommended)
curl -L -o assets/models/rerank.onnx \
  "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx"
```

Alternatively, set environment variables to point to the model files:
- `MCPMYDOCS_MODEL_PATH` - path to embed.onnx
- `MCPMYDOCS_RERANKER_PATH` - path to rerank.onnx (optional)

#### 3. Verify installation

```bash
./mcpmydocs --help
```

### Building from source

Requires Go 1.24+ and Homebrew (macOS).

#### 1. Clone and build

```bash
git clone https://github.com/yourusername/mcpmydocs.git
cd mcpmydocs
make all
```

This will:
- Install system dependencies (DuckDB, ONNX Runtime) via Homebrew
- Download Go modules
- Download the embedding model, reranker model, and tokenizer
- Build the binary to `dist/mcpmydocs`

#### 2. Verify installation

```bash
./dist/mcpmydocs --help
```

## Usage

### Index a directory

Index all Markdown files in a directory:

```bash
./dist/mcpmydocs index ~/Documents/wiki
```

Example output:
```
Indexing directory: /Users/matt/Documents/wiki
Database: /Users/matt/src/mcpmydocs/mcpmydocs.db
[247/247] guides/advanced-configuration.md
Indexing complete!
  Indexed: 247 files
  Skipped: 0 unchanged files
```

Re-running the command only processes changed files:
```
Indexing complete!
  Indexed: 2 files
  Skipped: 245 unchanged files
```

### Search from CLI

```bash
./dist/mcpmydocs search "how to configure authentication"
```

Example output (with reranking enabled by default):
```
Found 5 results for: "how to configure authentication" (reranked)

─────────────────────────────────────────────────────────────
[1] # Authentication > ## OAuth Setup (relevance: 2.15)
    File: /Users/matt/Documents/wiki/auth/oauth.md:15

    OAuth Setup

    To configure OAuth authentication, you'll need to:
    1. Register your application with the OAuth provider
    2. Set the callback URL to https://yourapp.com/auth/callback
    ...

─────────────────────────────────────────────────────────────
[2] # Security > ## API Keys (relevance: 1.82)
    File: /Users/matt/Documents/wiki/security/api-keys.md:8

    API Keys

    API keys provide a simple authentication mechanism...
```

Options:
```bash
# Return more results
./dist/mcpmydocs search "database migrations" -n 10

# Disable reranking (faster, uses vector similarity only)
./dist/mcpmydocs search "quick lookup" --no-rerank

# Adjust candidate pool for reranking (default: 50)
./dist/mcpmydocs search "detailed query" --candidates 100
```

### Run as MCP server

Start the MCP server for integration with AI tools:

```bash
./dist/mcpmydocs run
```

The server communicates via stdio using JSON-RPC 2.0 (the MCP protocol).

## Integrating with Claude Code

### Option 1: Wrapper script (recommended)

Create a wrapper script that sets the working directory:

```bash
# Create the script
cat > ~/bin/mcpmydocs-server << 'EOF'
#!/bin/bash
cd /Users/matt/src/mcpmydocs && ./dist/mcpmydocs run
EOF

chmod +x ~/bin/mcpmydocs-server

# Add to Claude Code
claude mcp add --transport stdio mcpmydocs -- ~/bin/mcpmydocs-server
```

### Option 2: Edit settings.json directly

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "mcpmydocs": {
      "command": "/Users/matt/src/mcpmydocs/dist/mcpmydocs",
      "args": ["run"],
      "cwd": "/Users/matt/src/mcpmydocs"
    }
  }
}
```

### Available MCP tools

Once connected, Claude Code has access to:

| Tool | Description |
|------|-------------|
| `search` | Semantic search with cross-encoder reranking. Parameters: `query` (required), `limit` (default 5, max 20), `rerank` (default true), `candidates` (default 50, max 100) |
| `list_documents` | List all indexed documents with titles and paths |

Example usage in Claude Code:
```
> Search my docs for information about database migrations

Claude will use the search tool and return relevant documentation snippets.
```

## How it works

1. **Chunking** - Markdown files are split into chunks by heading structure
2. **Embedding** - Each chunk is converted to a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
3. **Storage** - Vectors are stored in DuckDB with HNSW indexing via the [vss extension](https://duckdb.org/docs/extensions/vss.html)
4. **Search** - Two-stage retrieval:
   - **Stage 1 (Retrieval)**: Query is embedded and top-N candidates are fetched using cosine similarity
   - **Stage 2 (Reranking)**: Candidates are rescored using [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) cross-encoder for improved relevance

## Project structure

```
mcpmydocs/
├── cmd/
│   ├── index.go      # Index command
│   ├── search.go     # Search command
│   └── run.go        # MCP server command
├── internal/
│   ├── chunker/      # Markdown chunking logic
│   ├── embedder/     # ONNX embedding generation
│   ├── reranker/     # Cross-encoder reranking
│   ├── search/       # Unified search service
│   └── store/        # DuckDB storage layer
├── assets/
│   └── models/       # Downloaded ONNX models
├── Makefile
├── Brewfile
└── main.go
```

## Files

| File | Description |
|------|-------------|
| `mcpmydocs.db` | DuckDB database with indexed documents and embeddings |
| `assets/models/embed.onnx` | Embedding model (~90MB) |
| `assets/models/rerank.onnx` | Reranker model (~90MB) |
| `assets/models/tokenizer.json` | WordPiece tokenizer vocabulary |

## Development

```bash
# Install dependencies only
make deps

# Build only
make build

# Clean build artifacts and models
make clean

# Run tests
go test ./...
```

## Troubleshooting

### "database not found" error

Run the index command first:
```bash
./dist/mcpmydocs index /path/to/your/docs
```

### Poor search results

The search uses semantic similarity with cross-encoder reranking. Try:
- More descriptive queries ("how to set up OAuth" vs "oauth")
- Check that documents were actually indexed (use `list_documents` via MCP)
- Ensure the reranker model is available (check for "reranker enabled" in logs)

### Reranker not loading

If search results show similarity percentages instead of relevance scores, the reranker isn't loaded:
- Ensure `rerank.onnx` exists in `assets/models/` or set `MCPMYDOCS_RERANKER_PATH`
- Check logs for "reranker initialization failed" messages

### MCP server not connecting

1. Ensure the `cwd` is set correctly (the server needs access to `mcpmydocs.db` and `assets/models/`)
2. Check Claude Code logs: `~/.claude/logs/`
3. Test the server directly: `echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | ./dist/mcpmydocs run`

## License

MIT
