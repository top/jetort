#!/usr/bin/env bash
# Build JetOrt for one deployment target. This is the only compile entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET=""

usage() {
    cat <<'EOF'
Usage: build.sh --target jetson|linux-nvidia|macos|windows-nvidia|windows-other|linux-amd|cpu

  jetson          TensorRT → CUDA → CPU, CUDA arch from Cargo.toml (Jetson Orin is 87)
  linux-nvidia    TensorRT → CUDA → CPU
  macos           CoreML → CPU
  windows-nvidia  TensorRT → CUDA → CPU
  windows-other   DirectML → CPU (including AMD GPUs)
  linux-amd       CPU (ROCm is not built)
  cpu             CPU

With no --target, a terminal asks which target to use. Without a terminal, this exits.
CI always passes --target.
EOF
}

read_cuda_arch() {
    awk '
        /^\[package.metadata.hardware\]$/ { section=1; next }
        /^\[/ { section=0 }
        section && /^cuda_arch[[:space:]]*=/ {
            gsub(/[^0-9]/, "", $0)
            print $0
            exit
        }
    ' "$PROJECT_ROOT/Cargo.toml"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)
            TARGET="${2:?--target requires a value}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "$TARGET" ]; then
    if [ -t 0 ]; then
        echo "Select a JetOrt build target:"
        echo "  1) jetson"
        echo "  2) linux-nvidia"
        echo "  3) macos"
        echo "  4) windows-nvidia"
        echo "  5) windows-other"
        echo "  6) linux-amd"
        echo "  7) cpu"
        printf "Target [1-7]: "
        read -r choice
        case "$choice" in
            1|jetson) TARGET=jetson ;;
            2|linux-nvidia) TARGET=linux-nvidia ;;
            3|macos) TARGET=macos ;;
            4|windows-nvidia) TARGET=windows-nvidia ;;
            5|windows-other) TARGET=windows-other ;;
            6|linux-amd) TARGET=linux-amd ;;
            7|cpu) TARGET=cpu ;;
            *) echo "Unknown target: $choice" >&2; exit 2 ;;
        esac
    else
        echo "No --target and stdin is not a terminal." >&2
        usage >&2
        exit 2
    fi
fi

command -v cargo >/dev/null || {
    echo "cargo is required. Install Rust before running build.sh." >&2
    exit 1
}

FEATURES=()
case "$TARGET" in
    jetson)
        FEATURES=(cuda tensorrt)
        CUDA_ARCH="${JETORT_CUDA_ARCH:-${TARGET_CUDA_ARCH:-$(read_cuda_arch)}}"
        if [ -z "$CUDA_ARCH" ]; then
            echo "Jetson builds need cuda_arch in Cargo.toml or JETORT_CUDA_ARCH." >&2
            exit 1
        fi
        export JETORT_CUDA_ARCH="$CUDA_ARCH"
        export TARGET_CUDA_ARCH="$CUDA_ARCH"
        export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
        echo "[target] jetson CUDA architecture ${CUDA_ARCH} (CMAKE_CUDA_ARCHITECTURES / TARGET_CUDA_ARCH)"
        ;;
    linux-nvidia|windows-nvidia)
        FEATURES=(cuda tensorrt)
        if [ -n "${JETORT_CUDA_ARCH:-${TARGET_CUDA_ARCH:-}}" ]; then
            CUDA_ARCH="${JETORT_CUDA_ARCH:-$TARGET_CUDA_ARCH}"
            export JETORT_CUDA_ARCH="$CUDA_ARCH"
            export TARGET_CUDA_ARCH="$CUDA_ARCH"
            export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
            echo "[target] ${TARGET} CUDA architecture ${CUDA_ARCH}"
        fi
        ;;
    macos)
        FEATURES=(coreml)
        ;;
    windows-other)
        FEATURES=(directml)
        ;;
    linux-amd|cpu)
        FEATURES=()
        echo "[target] ${TARGET}: CPU only. ROCm is not enabled."
        ;;
    *)
        echo "Unknown target: $TARGET" >&2
        usage >&2
        exit 2
        ;;
esac

# Library hints for the dynamic loader. Missing libraries do not fail the compile;
# the runtime logs a fallback when an execution provider cannot be loaded.
case "$TARGET" in
    jetson|linux-nvidia)
        echo "[hint] ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so"
        echo "[hint] TensorRT: libnvinfer.so, NvInfer.h, libonnxruntime_providers_tensorrt.so"
        ;;
    macos)
        echo "[hint] ORT dylib: libonnxruntime.dylib (ORT_DYLIB_PATH)"
        ;;
    windows-nvidia|windows-other)
        echo "[hint] ORT_DYLIB_PATH=onnxruntime.dll"
        ;;
esac

cd "$PROJECT_ROOT"
if [ "${#FEATURES[@]}" -gt 0 ]; then
    FEATURE_LIST=$(IFS=,; echo "${FEATURES[*]}")
    echo "[build] cargo build --release --no-default-features --features ${FEATURE_LIST}"
    cargo build --release --no-default-features --features "$FEATURE_LIST"
else
    echo "[build] cargo build --release --no-default-features"
    cargo build --release --no-default-features
fi

echo "[ok] $PROJECT_ROOT/target/release/jetort"
