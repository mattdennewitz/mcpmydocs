# mcpmydocs

[![CI](https://github.com/mattdennewitz/mcpmydocs/actions/workflows/ci.yml/badge.svg)](https://github.com/mattdennewitz/mcpmydocs/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/mattdennewitz/mcpmydocs?v=0.3.0)](https://goreportcard.com/report/github.com/mattdennewitz/mcpmydocs)
[![GitHub Release](https://img.shields.io/github/v/release/mattdennewitz/mcpmydocs)](https://github.com/mattdennewitz/mcpmydocs/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local-first semantic search engine for Markdown documentation. Indexes your docs, generates embeddings using ONNX models, stores them in DuckDB, and exposes search via CLI or MCP server for AI agents like Claude Code.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Integrating with Claude Code](#integrating-with-claude-code)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quick Start

```bash
# Install mcpmydocs and models
curl -sSL https://raw.githubusercontent.com/mattdennewitz/mcpmydocs/main/install.sh | bash

# Install ONNX Runtime
brew install onnxruntime  # macOS
# Linux: see Installation section for instructions

# Index your documentation
mcpmydocs index ~/Documents/wiki

# Search
mcpmydocs search "how to configure authentication"
```

> **Note:** A `Brewfile` is included for macOS users building from source—run `brew bundle` to install all dependencies.

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
- Download the embedding and reranker models to `~/.local/share/mcpmydocs/models/`

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

The application automatically searches for models in:
1. `~/.local/share/mcpmydocs/models/` (install script location)
2. `assets/models/` relative to the binary
3. `assets/models/` in the current working directory

#### 3. Verify installation

```bash
./mcpmydocs --help
```

### Building from source

Requires Go 1.24+ and Homebrew (macOS).

#### 1. Clone and build

```bash
git clone https://github.com/mattdennewitz/mcpmydocs.git
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
mcpmydocs --help
```

## Configuration

### Database location

The database file `mcpmydocs.db` is created in the current working directory by default. Use the `--db` flag to specify a custom path:

```bash
mcpmydocs index ~/Documents/wiki --db ~/data/mcpmydocs.db
mcpmydocs search "query" --db ~/data/mcpmydocs.db
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `ONNX_LIBRARY_PATH` | Path to the ONNX Runtime library (if not in standard locations) |
| `MCPMYDOCS_MODEL_PATH` | Path to `embed.onnx` embedding model |
| `MCPMYDOCS_RERANKER_PATH` | Path to `rerank.onnx` reranker model |

## Usage

> **Note:** Examples below assume `mcpmydocs` is in your PATH. If you built from source, use `mcpmydocs` instead.

### Index a directory

Index all Markdown files in a directory:

```bash
mcpmydocs index ~/Documents/wiki
```

Example output:
```
Indexing directory: /home/user/docs
Database: /home/user/docs/mcpmydocs.db
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
mcpmydocs search "how to configure authentication"
```

Example output (with reranking enabled by default):
```
Found 5 results for: "how to configure authentication" (reranked)

─────────────────────────────────────────────────────────────
[1] # Authentication > ## OAuth Setup (relevance: 2.15)
    File: /home/user/docs/auth/oauth.md:15

    OAuth Setup

    To configure OAuth authentication, you'll need to:
    1. Register your application with the OAuth provider
    2. Set the callback URL to https://yourapp.com/auth/callback
    ...

─────────────────────────────────────────────────────────────
[2] # Security > ## API Keys (relevance: 1.82)
    File: /home/user/docs/security/api-keys.md:8

    API Keys

    API keys provide a simple authentication mechanism...
```

Options:
```bash
# Return more results
mcpmydocs search "database migrations" -n 10

# Disable reranking (faster, uses vector similarity only)
mcpmydocs search "quick lookup" --no-rerank

# Adjust candidate pool for reranking (default: 50)
mcpmydocs search "detailed query" --candidates 100
```

### Run as MCP server

Start the MCP server for integration with AI tools:

```bash
mcpmydocs run
```

The server communicates via stdio using JSON-RPC 2.0 (the MCP protocol).

## Integrating with Claude Code

Create a wrapper script that sets the working directory, then register it with Claude Code:

```bash
# Create the script (adjust MCPMYDOCS_DIR to your install location)
MCPMYDOCS_DIR="$HOME/.local/share/mcpmydocs"

mkdir -p ~/bin
cat > ~/bin/mcpmydocs-server << EOF
#!/bin/bash
cd $MCPMYDOCS_DIR && mcpmydocs run
EOF

chmod +x ~/bin/mcpmydocs-server

# Add to Claude Code
claude mcp add --transport stdio mcpmydocs -- ~/bin/mcpmydocs-server
```

The wrapper script is necessary because the MCP server needs to run from a directory containing the database and models.

### Available MCP tools

Once connected, Claude Code has access to:

#### `search`

Semantic search with cross-encoder reranking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | (required) | The search query |
| `limit` | integer | 5 | Number of results to return (max 20) |
| `rerank` | boolean | true | Enable cross-encoder reranking |
| `candidates` | integer | 50 | Candidate pool size for reranking (max 100) |

#### `list_documents`

List all indexed documents with titles and paths. No parameters.

### Example usage in Claude Code

Ask Claude Code to search your indexed documentation:

```
Search my docs for information about database migrations
```

Claude will automatically use the `search` tool and return relevant documentation snippets.

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
│   ├── config.go     # CLI configuration
│   ├── index.go      # Index command
│   ├── run.go        # MCP server command
│   └── search.go     # Search command
├── internal/
│   ├── app/          # Application initialization
│   ├── chunker/      # Markdown chunking logic
│   ├── embedder/     # ONNX embedding generation
│   ├── logger/       # Logging utilities
│   ├── paths/        # Path resolution for models
│   ├── reranker/     # Cross-encoder reranking
│   ├── search/       # Unified search service
│   └── store/        # DuckDB storage layer
├── assets/
│   └── models/       # Downloaded ONNX models
├── Makefile
├── Brewfile
└── main.go
```

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

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests before submitting (`go test ./...`)
4. Submit a pull request

For bugs or feature requests, please [open an issue](https://github.com/mattdennewitz/mcpmydocs/issues).

## Troubleshooting

### "database not found" error

Run the index command first:
```bash
mcpmydocs index /path/to/your/docs
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

1. Ensure the wrapper script runs from a directory containing `mcpmydocs.db` and the models
2. Check Claude Code logs: `~/.claude/logs/`
3. Test the server directly: `echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | mcpmydocs run`

## License

MIT
