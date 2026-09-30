#!/usr/bin/env bash
# Jetson convenience wrapper. Build parameters live here; compilation is build.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_SERVICE=true
START_SERVICE=true

usage() {
    cat <<'EOF'
Usage: jetson.sh [--no-service] [--no-start]

Fills Jetson CUDA architecture 87 (Cargo.toml package.metadata.hardware.cuda_arch)
and calls scripts/build.sh --target jetson. Then installs the user systemd unit
and starts it, which is how the server runs on device.
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

# Jetson Orin is compute capability 8.7. build.sh forwards CUDA_ARCH to the compiler.
CUDA_ARCH="$(awk '
    /^\[package.metadata.hardware\]$/ { section=1; next }
    /^\[/ { section=0 }
    section && /^cuda_arch[[:space:]]*=/ {
        gsub(/[^0-9]/, "", $0)
        print $0
        exit
    }
' "$PROJECT_ROOT/Cargo.toml")"
CUDA_ARCH="${CUDA_ARCH:-87}"
export JETORT_CUDA_ARCH="$CUDA_ARCH"
export TARGET_CUDA_ARCH="$CUDA_ARCH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

"$SCRIPT_DIR/build.sh" --target jetson

# --- install (not a second build) ---
# The user systemd unit used to be mixed into deploy.sh. Library setup stays in
# setup_onnxruntime.sh.
if [ "$INSTALL_SERVICE" = true ]; then
    if [ "$START_SERVICE" = true ]; then
        "$SCRIPT_DIR/install-user-service.sh"
    else
        "$SCRIPT_DIR/install-user-service.sh" --no-start
    fi
elif [ "$START_SERVICE" = true ]; then
    exec "$PROJECT_ROOT/target/release/jetort" --config "$PROJECT_ROOT/config.yaml"
fi
