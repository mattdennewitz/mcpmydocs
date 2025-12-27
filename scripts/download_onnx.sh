#!/bin/bash
set -e

# Configuration
ONNX_VERSION="1.17.1"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
if [ "$OS" = "darwin" ]; then
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        ARCH="arm64"
    elif [ "$ARCH" = "x86_64" ]; then
        ARCH="x86_64"
    fi
elif [ "$OS" = "linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        ARCH="x64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        ARCH="aarch64"
    fi
fi

# Determine download URL
if [ "$OS" = "darwin" ]; then
    # macOS is "osx" in release names
    # e.g. onnxruntime-osx-arm64-1.17.1.tgz
    PLATFORM="osx"
    EXT="tgz"
    LIB_NAME="libonnxruntime.1.17.1.dylib" # The tarball usually contains versioned libs
    TARGET_LIB="libonnxruntime.dylib"
elif [ "$OS" = "linux" ]; then
    PLATFORM="linux"
    EXT="tgz"
    LIB_NAME="libonnxruntime.so.1.17.1"
    TARGET_LIB="libonnxruntime.so"
else
    echo "Unsupported OS for auto-download: $OS"
    exit 1
fi

BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}"
FILENAME="onnxruntime-${PLATFORM}-${ARCH}-${ONNX_VERSION}.${EXT}"
URL="${BASE_URL}/${FILENAME}"

echo "Downloading ONNX Runtime v${ONNX_VERSION} for ${OS}/${ARCH}..."
echo "Source: ${URL}"

mkdir -p dist/lib
mkdir -p dist/tmp

# Download
if [ ! -f "dist/tmp/${FILENAME}" ]; then
    curl -L "${URL}" -o "dist/tmp/${FILENAME}"
fi

# Extract
echo "Extracting..."
tar -xzf "dist/tmp/${FILENAME}" -C "dist/tmp"

# Find and copy library
# The structure inside tgz varies slightly but usually:
# onnxruntime-osx-arm64-1.17.1/lib/libonnxruntime.dylib
EXTRACTED_DIR="dist/tmp/onnxruntime-${PLATFORM}-${ARCH}-${ONNX_VERSION}"

if [ "$OS" = "darwin" ]; then
    # On macOS, we often get .dylib and .1.17.1.dylib. Copy the dylib.
    cp "${EXTRACTED_DIR}/lib/libonnxruntime.dylib" dist/lib/
    # Also copy the versioned one if it's the real file and others are symlinks?
    # Usually libonnxruntime.dylib is the one we want to link against.
elif [ "$OS" = "linux" ]; then
    cp "${EXTRACTED_DIR}/lib/${LIB_NAME}" "dist/lib/${TARGET_LIB}"
fi

# Clean up
rm -rf dist/tmp

echo "Success! Library placed in dist/lib/${TARGET_LIB}"
