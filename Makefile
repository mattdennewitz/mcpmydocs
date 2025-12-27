# mcpmydocs Build System

BINARY_NAME=mcpmydocs
DIST_DIR=dist
MODELS_DIR=assets/models
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=BAAI/bge-reranker-base

.PHONY: all build clean download-models install-deps install-go-modules deps

all: install-deps install-go-modules download-models build

## Dependency Management
install-deps:
	@echo "Installing system dependencies via Homebrew..."
	brew bundle --file=Brewfile

install-go-modules:
	@echo "Downloading Go modules..."
	go mod download
	go mod tidy

## Model Management
download-models:
	@echo "Checking for models..."
	@mkdir -p $(MODELS_DIR)
	@# Download Embedding Model (ONNX)
	@if [ ! -f $(MODELS_DIR)/embed.onnx ]; then \
		echo "Downloading $(EMBED_MODEL)..."; \
		curl -L "https://huggingface.co/$(EMBED_MODEL)/resolve/main/onnx/model.onnx" -o $(MODELS_DIR)/embed.onnx; \
	fi
	@# Download Reranker Model (ONNX)
	@if [ ! -f $(MODELS_DIR)/rerank.onnx ]; then \
		echo "Downloading $(RERANK_MODEL)..."; \
		curl -L "https://huggingface.co/$(RERANK_MODEL)/resolve/main/onnx/model.onnx" -o $(MODELS_DIR)/rerank.onnx; \
	fi
	@# Download Tokenizer config
	@if [ ! -f $(MODELS_DIR)/tokenizer.json ]; then \
		echo "Downloading tokenizer config..."; \
		curl -L "https://huggingface.co/$(EMBED_MODEL)/resolve/main/tokenizer.json" -o $(MODELS_DIR)/tokenizer.json; \
	fi

## Build
build:
	@echo "Building mcpmydocs $(shell git describe --tags --always || echo "v0.1.0")..."
	@mkdir -p $(DIST_DIR)
	@go build -o $(DIST_DIR)/$(BINARY_NAME) main.go

dist-bundle: build download-models
	@echo "Creating self-contained bundle..."
	@mkdir -p $(DIST_DIR)/lib
	@mkdir -p $(DIST_DIR)/assets/models
	@# Copy models
	@cp -r $(MODELS_DIR)/* $(DIST_DIR)/assets/models/
	@# Download and copy ONNX libs
	@./scripts/download_onnx.sh
	@echo "Bundle created in $(DIST_DIR)"

clean:
	rm -rf $(DIST_DIR)
	rm -rf assets/models/*.onnx

## Release builds (cross-platform)
PLATFORMS=darwin_amd64 darwin_arm64 linux_amd64 linux_arm64
VERSION?=$(shell git describe --tags --always 2>/dev/null || echo "dev")

release:
	@echo "Building release binaries for version $(VERSION)..."
	@mkdir -p $(DIST_DIR)/release
	@for platform in $(PLATFORMS); do \
		GOOS=$$(echo $$platform | cut -d_ -f1); \
		GOARCH=$$(echo $$platform | cut -d_ -f2); \
		output="$(DIST_DIR)/release/$(BINARY_NAME)_$$platform"; \
		echo "Building $$platform..."; \
		mkdir -p $$output; \
		CGO_ENABLED=1 GOOS=$$GOOS GOARCH=$$GOARCH go build -ldflags="-s -w -X main.Version=$(VERSION)" -o $$output/$(BINARY_NAME) main.go || continue; \
		tar -czf $$output.tar.gz -C $$output $(BINARY_NAME); \
		rm -rf $$output; \
	done
	@echo "Release binaries created in $(DIST_DIR)/release/"

release-local:
	@echo "Building release binary for current platform..."
	@mkdir -p $(DIST_DIR)/release
	@platform="$$(go env GOOS)_$$(go env GOARCH)"; \
	output="$(DIST_DIR)/release/$(BINARY_NAME)_$$platform"; \
	mkdir -p $$output; \
	go build -ldflags="-s -w -X main.Version=$(VERSION)" -o $$output/$(BINARY_NAME) main.go; \
	tar -czf $$output.tar.gz -C $$output $(BINARY_NAME); \
	rm -rf $$output; \
	echo "Created $$output.tar.gz"

## Development Setup
deps: install-deps install-go-modules
	@echo "System and Go dependencies ready."
	@echo "Note: Ensure libduckdb and libonnxruntime are linked correctly in your environment."