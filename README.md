# mcpmydocs

A local-first semantic search engine for Markdown documentation. Indexes your docs, generates embeddings using ONNX models, stores them in DuckDB, and exposes search via CLI or MCP server for AI agents like Claude Code.

## Features

- **Semantic search** - Find documents by meaning, not just keywords
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
- Download the embedding model to `~/.local/share/mcpmydocs`

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

#### 2. Download the embedding model

Download the model files and place them in an `assets/models/` directory next to the binary:

```bash
mkdir -p assets/models

# Download embedding model (~90MB)
curl -L -o assets/models/embed.onnx \
  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

# Download tokenizer
curl -L -o assets/models/tokenizer.json \
  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
```

Alternatively, set the `MCPMYDOCS_MODEL_PATH` environment variable to point to the model file.

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
- Download the embedding model (~90MB) and tokenizer
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

Example output:
```
Found 5 results for: "how to configure authentication"

─────────────────────────────────────────────────────────────
[1] # Authentication > ## OAuth Setup (67.3% similar)
    File: /Users/matt/Documents/wiki/auth/oauth.md:15

    OAuth Setup

    To configure OAuth authentication, you'll need to:
    1. Register your application with the OAuth provider
    2. Set the callback URL to https://yourapp.com/auth/callback
    ...

─────────────────────────────────────────────────────────────
[2] # Security > ## API Keys (54.2% similar)
    File: /Users/matt/Documents/wiki/security/api-keys.md:8

    API Keys

    API keys provide a simple authentication mechanism...
```

Options:
```bash
# Return more results
./dist/mcpmydocs search "database migrations" -n 10

# Quoted phrases work naturally
./dist/mcpmydocs search "NLA reporting fixes"
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
| `search` | Semantic search over indexed documents. Parameters: `query` (string), `limit` (int, optional, default 5, max 20) |
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
4. **Search** - Query text is embedded and compared against stored vectors using cosine similarity

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
| `assets/models/rerank.onnx` | Reranker model (~1.1GB, for future use) |
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

The search uses semantic similarity, not keyword matching. Try:
- More descriptive queries ("how to set up OAuth" vs "oauth")
- Check that documents were actually indexed (use `list_documents` via MCP)

### MCP server not connecting

1. Ensure the `cwd` is set correctly (the server needs access to `mcpmydocs.db` and `assets/models/`)
2. Check Claude Code logs: `~/.claude/logs/`
3. Test the server directly: `echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | ./dist/mcpmydocs run`

## License

MIT
