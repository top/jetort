#!/usr/bin/env bash
# Build JetOrt and install its user systemd service. Does not install system dependencies.

set -euo pipefail

# ponytail: installs a `systemctl --user` unit for the invoking account.
# Running as root would install it for root's user session instead of the
# real operator, and would run the inference server with needless privilege.
[ "$EUID" -ne 0 ] || {
    echo "Do not run this script as root/sudo; it installs a user-level systemd service for your account." >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_SERVICE=true
START_SERVICE=true

usage() {
    cat <<'EOF'
Usage: deploy.sh [--no-service] [--no-start]

  --no-service   Build only; do not install the user systemd unit.
  --no-start     Install/update the unit but do not start or restart it.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-service) INSTALL_SERVICE=false; START_SERVICE=false ;;
        --no-start) START_SERVICE=false ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

command -v cargo >/dev/null || {
    echo "cargo is required. Install Rust before running deploy.sh." >&2
    exit 1
}

find_first() {
    for pattern in "$@"; do
        for candidate in $pattern; do
            if [ -e "$candidate" ]; then
                printf '%s' "$candidate"
                return 0
            fi
        done
    done
    return 1
}

FEATURES=()
if [ "$(uname -s)" = "Darwin" ]; then
    FEATURES=(coreml)
elif [ -e /dev/nvidia0 ] || [ -e /dev/nvhost-gpu ] || command -v nvidia-smi >/dev/null; then
    FEATURES=(cuda)
    TRT_LIBRARY=$(find_first \
        "/usr/lib/aarch64-linux-gnu/libnvinfer.so*" \
        "/usr/lib/x86_64-linux-gnu/libnvinfer.so*" \
        "/usr/local/lib/libnvinfer.so*" \
        "/usr/local/TensorRT*/lib/libnvinfer.so*" || true)
    TRT_HEADER=$(find_first \
        "/usr/include/aarch64-linux-gnu/NvInfer.h" \
        "/usr/include/x86_64-linux-gnu/NvInfer.h" \
        "/usr/include/NvInfer.h" \
        "/usr/local/TensorRT*/include/NvInfer.h" || true)
    ORT_TRT_PROVIDER=$(find_first \
        "/usr/local/lib/libonnxruntime_providers_tensorrt.so*" \
        "/usr/lib/aarch64-linux-gnu/libonnxruntime_providers_tensorrt.so*" \
        "/usr/lib/x86_64-linux-gnu/libonnxruntime_providers_tensorrt.so*" || true)

    if [ -z "$TRT_LIBRARY" ] && command -v ldconfig >/dev/null; then
        TRT_LIBRARY=$(ldconfig -p 2>/dev/null | awk '/libnvinfer\.so/{print $NF; exit}')
    fi
    if [ -z "$ORT_TRT_PROVIDER" ] && command -v ldconfig >/dev/null; then
        ORT_TRT_PROVIDER=$(ldconfig -p 2>/dev/null | awk '/libonnxruntime_providers_tensorrt\.so/{print $NF; exit}')
    fi

    if [ -n "$TRT_LIBRARY" ] && [ -n "$TRT_HEADER" ] && [ -n "$ORT_TRT_PROVIDER" ]; then
        FEATURES+=(tensorrt)
        echo "[detect] TensorRT library: $TRT_LIBRARY"
        echo "[detect] TensorRT headers: $TRT_HEADER"
        echo "[detect] ORT TensorRT provider: $ORT_TRT_PROVIDER"
    else
        echo "[skip] TensorRT feature disabled because required components are incomplete:"
        [ -n "$TRT_LIBRARY" ] || echo "       missing libnvinfer.so"
        [ -n "$TRT_HEADER" ] || echo "       missing NvInfer.h"
        [ -n "$ORT_TRT_PROVIDER" ] || echo "       missing libonnxruntime_providers_tensorrt.so"
        echo "       JetOrt will be built with CUDA fallback."
    fi
    CUDA_ARCH=$(awk '
        /^\[package.metadata.hardware\]$/ { section=1; next }
        /^\[/ { section=0 }
        section && /^cuda_arch[[:space:]]*=/ {
            gsub(/[^0-9]/, "", $0)
            print $0
            exit
        }
    ' "$PROJECT_ROOT/Cargo.toml")
    [ -n "$CUDA_ARCH" ] && FEATURES+=("sm_$CUDA_ARCH")
fi

cd "$PROJECT_ROOT"
if [ "${#FEATURES[@]}" -gt 0 ]; then
    FEATURE_LIST=$(IFS=,; echo "${FEATURES[*]}")
    echo "[build] Compiled runtime features: $FEATURE_LIST"
    echo "[build] cargo build --release --no-default-features --features $FEATURE_LIST"
    cargo build --release --no-default-features --features "$FEATURE_LIST"
else
    echo "[build] cargo build --release --no-default-features"
    cargo build --release --no-default-features
fi

TARGET_BIN="$PROJECT_ROOT/target/release/jetort"
[ -x "$TARGET_BIN" ] || {
    echo "Build succeeded but $TARGET_BIN is missing." >&2
    exit 1
}
echo "[ok] Binary: $TARGET_BIN"

if [ "$INSTALL_SERVICE" = false ]; then
    exit 0
fi

command -v systemctl >/dev/null || {
    echo "systemctl is unavailable; binary was built but no service was installed." >&2
    exit 1
}

SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$SERVICE_DIR/jetort.service"
mkdir -p "$SERVICE_DIR"

sed \
    -e "s|@PROJECT_ROOT@|$PROJECT_ROOT|g" \
    -e "s|@JETORT_BIN@|$TARGET_BIN|g" \
    "$SCRIPT_DIR/jetort.service" > "$SERVICE_FILE"

systemctl --user daemon-reload
echo "[ok] Service installed: $SERVICE_FILE"

if [ "$START_SERVICE" = true ]; then
    systemctl --user enable jetort.service >/dev/null
    systemctl --user restart jetort.service
    systemctl --user --no-pager --full status jetort.service
fi
