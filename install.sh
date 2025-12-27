#!/bin/bash
set -e

# mcpmydocs installer
# Usage: curl -sSL https://raw.githubusercontent.com/mattdennewitz/mcpmydocs/main/install.sh | bash

REPO="mattdennewitz/mcpmydocs"
INSTALL_DIR="${MCPMYDOCS_INSTALL_DIR:-$HOME/.local/bin}"
ASSETS_DIR="${MCPMYDOCS_ASSETS_DIR:-$HOME/.local/share/mcpmydocs}"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

info() { echo -e "${BLUE}==>${NC} $1"; }
success() { echo -e "${GREEN}==>${NC} $1"; }
warn() { echo -e "${YELLOW}==>${NC} $1"; }
error() { echo -e "${RED}==>${NC} $1"; exit 1; }

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin) OS="darwin" ;;
        Linux) OS="linux" ;;
        *) error "Unsupported operating system: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="amd64" ;;
        arm64|aarch64) ARCH="arm64" ;;
        *) error "Unsupported architecture: $ARCH" ;;
    esac

    PLATFORM="${OS}_${ARCH}"

    # Check for available pre-built binaries
    case "$PLATFORM" in
        darwin_arm64|linux_amd64|linux_arm64)
            info "Detected platform: $PLATFORM"
            ;;
        darwin_amd64)
            error "Intel Mac binaries are not available. Please build from source:
    git clone https://github.com/${REPO}.git
    cd mcpmydocs && go build -o mcpmydocs main.go"
            ;;
        *)
            error "Unsupported platform: $PLATFORM"
            ;;
    esac
}

# Get the latest release version from GitHub
get_latest_version() {
    info "Fetching latest release..."
    VERSION=$(curl -sSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

    if [ -z "$VERSION" ]; then
        error "Could not determine latest version. Check https://github.com/${REPO}/releases"
    fi

    info "Latest version: $VERSION"
}

# Download and install the binary
install_binary() {
    BINARY_NAME="mcpmydocs_${PLATFORM}"
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY_NAME}.tar.gz"

    info "Downloading mcpmydocs..."

    mkdir -p "$INSTALL_DIR"
    TMP_DIR=$(mktemp -d)
    trap "rm -rf $TMP_DIR" EXIT

    if ! curl -sSL "$DOWNLOAD_URL" -o "$TMP_DIR/mcpmydocs.tar.gz"; then
        error "Failed to download from $DOWNLOAD_URL"
    fi

    tar -xzf "$TMP_DIR/mcpmydocs.tar.gz" -C "$TMP_DIR"

    # Find the binary (might be in a subdirectory)
    BINARY=$(find "$TMP_DIR" -name "mcpmydocs" -type f | head -1)
    if [ -z "$BINARY" ]; then
        error "Binary not found in archive"
    fi

    mv "$BINARY" "$INSTALL_DIR/mcpmydocs"
    chmod +x "$INSTALL_DIR/mcpmydocs"

    success "Installed mcpmydocs to $INSTALL_DIR/mcpmydocs"
}

# Download model files
install_models() {
    MODEL_DIR="$ASSETS_DIR/models"
    mkdir -p "$MODEL_DIR"

    if [ -f "$MODEL_DIR/embed.onnx" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
        info "Model files already exist, skipping download"
        return
    fi

    info "Downloading embedding model (~90MB)..."
    curl -sSL -o "$MODEL_DIR/embed.onnx" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

    info "Downloading tokenizer..."
    curl -sSL -o "$MODEL_DIR/tokenizer.json" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

    success "Model files installed to $MODEL_DIR"
}

# Check for ONNX Runtime
check_onnx_runtime() {
    case "$OS" in
        darwin)
            if [ -f "/opt/homebrew/lib/libonnxruntime.dylib" ] || [ -f "/usr/local/lib/libonnxruntime.dylib" ]; then
                success "ONNX Runtime found"
                return
            fi

            if command -v brew &> /dev/null; then
                warn "ONNX Runtime not found. Install with:"
                echo "    brew install onnxruntime"
            else
                warn "ONNX Runtime not found. Install Homebrew first, then run:"
                echo "    brew install onnxruntime"
            fi
            ;;
        linux)
            if ldconfig -p 2>/dev/null | grep -q libonnxruntime; then
                success "ONNX Runtime found"
                return
            fi

            warn "ONNX Runtime not found. Install from:"
            echo "    https://github.com/microsoft/onnxruntime/releases"
            echo "    Extract and copy libonnxruntime.so to /usr/local/lib/"
            ;;
    esac
}

# Print post-install instructions
print_instructions() {
    echo ""
    success "Installation complete!"
    echo ""
    echo "Add to your PATH (if not already):"
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo ""
    echo "Set the model path environment variable:"
    echo "    export MCPMYDOCS_MODEL_PATH=\"$ASSETS_DIR/models/embed.onnx\""
    echo ""
    echo "Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo "    export MCPMYDOCS_MODEL_PATH=\"$ASSETS_DIR/models/embed.onnx\""
    echo ""
    echo "Then start indexing:"
    echo "    mcpmydocs index /path/to/your/docs"
    echo ""
}

main() {
    echo ""
    echo "mcpmydocs installer"
    echo "==================="
    echo ""

    detect_platform
    get_latest_version
    install_binary
    install_models
    check_onnx_runtime
    print_instructions
}

main
