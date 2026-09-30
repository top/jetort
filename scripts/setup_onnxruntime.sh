#!/usr/bin/env bash
# Install and verify ONNX Runtime shared libraries. Does not install GenAI or deploy JetOrt.

set -euo pipefail

# The script keeps privilege escalation local to the individual install steps.
# handles the one privileged step (writing to $PREFIX).
[ "$EUID" -ne 0 ] || {
    echo "Do not run this script as root/sudo; it escalates only the steps that need it." >&2
    exit 1
}

PREFIX="/usr/local"
WHEEL="${CONFIG_ORT_WHEEL:-}"
CHECK_ONLY=false
FORCE=false

usage() {
    cat <<'EOF'
Usage: setup_onnxruntime.sh [--check] [--force] [--prefix DIR] [--wheel PATH_OR_URL]

  --check         Verify the existing installation without changing it.
  --force         Reinstall even when the current libraries are usable.
  --prefix DIR    Installation prefix (default: /usr/local).
  --wheel VALUE   Local wheel, wheel URL, or pip package specifier.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=true ;;
        --force) FORCE=true ;;
        --prefix)
            PREFIX="${2:?--prefix requires a directory}"
            shift
            ;;
        --wheel)
            WHEEL="${2:?--wheel requires a value}"
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

LIB_DIR="$PREFIX/lib"
SUDO=""
if [ "$EUID" -ne 0 ] && [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    command -v sudo >/dev/null || {
        echo "sudo is required to write to $PREFIX" >&2
        exit 1
    }
    SUDO="sudo"
fi

find_library() {
    local name="$1"
    local found=""
    found=$(find "$LIB_DIR" -maxdepth 1 \( -type f -o -type l \) -name "$name" 2>/dev/null | head -n 1 || true)
    if [ -z "$found" ] && command -v ldconfig >/dev/null; then
        found=$(ldconfig -p 2>/dev/null | awk -v name="$name" '$1 == name { print $NF; exit }')
    fi
    printf '%s' "$found"
}

library_usable() {
    local library="$1"
    [ -n "$library" ] && [ -e "$library" ] || return 1
    if command -v ldd >/dev/null; then
        ! ldd "$library" 2>/dev/null | grep -q "not found"
    fi
}

ORT_LIB=$(find_library "libonnxruntime.so")
CUDA_PROVIDER=$(find_library "libonnxruntime_providers_cuda.so")
TRT_PROVIDER=$(find_library "libonnxruntime_providers_tensorrt.so")
NVIDIA_DEVICE=false
if [ -e /dev/nvidia0 ] || [ -e /dev/nvhost-gpu ] || command -v nvidia-smi >/dev/null; then
    NVIDIA_DEVICE=true
fi

if library_usable "$ORT_LIB" \
    && { [ "$NVIDIA_DEVICE" = false ] || library_usable "$CUDA_PROVIDER"; } \
    && { [ -z "$CUDA_PROVIDER" ] || library_usable "$CUDA_PROVIDER"; } \
    && { [ -z "$TRT_PROVIDER" ] || library_usable "$TRT_PROVIDER"; }; then
    echo "[ok] ONNX Runtime is available: $ORT_LIB"
    [ -n "$CUDA_PROVIDER" ] && echo "[ok] CUDA provider: $CUDA_PROVIDER"
    [ -n "$TRT_PROVIDER" ] && echo "[ok] TensorRT provider: $TRT_PROVIDER"
    if [ "$FORCE" = false ]; then
        echo "[skip] Existing ONNX Runtime installation is usable."
        exit 0
    fi

    if [ "$NVIDIA_DEVICE" = true ] && ! library_usable "$CUDA_PROVIDER"; then
        echo "[missing] NVIDIA device detected but the ONNX Runtime CUDA provider is unavailable."
    fi
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "[missing] A usable ONNX Runtime installation was not found." >&2
    exit 1
fi

command -v python3 >/dev/null || {
    echo "python3 is required to extract an ONNX Runtime wheel." >&2
    exit 1
}

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/jetort-ort.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT

if [ -z "$WHEEL" ]; then
    if [ "$(uname -m)" = "aarch64" ]; then
        WHEEL=$(python3 - <<'PY'
import json
import urllib.request

url = "https://api.github.com/repos/Shattered217/Jetson-Orin-Wheels/releases/latest"
request = urllib.request.Request(url, headers={"User-Agent": "JetOrt"})
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        for asset in json.load(response).get("assets", []):
            name = asset.get("name", "")
            if "onnxruntime_gpu" in name and "aarch64.whl" in name:
                print(asset["browser_download_url"])
                break
except Exception:
    pass
PY
)
    else
        WHEEL="onnxruntime-gpu"
    fi
fi

[ -n "$WHEEL" ] || {
    echo "Could not resolve a compatible ONNX Runtime wheel; pass --wheel." >&2
    exit 1
}

if [ -f "$WHEEL" ]; then
    cp "$WHEEL" "$TEMP_DIR/"
elif [[ "$WHEEL" == http://* || "$WHEEL" == https://* ]]; then
    command -v curl >/dev/null || {
        echo "curl is required to download $WHEEL" >&2
        exit 1
    }
    curl -fL --retry 3 -o "$TEMP_DIR/$(basename "${WHEEL%%\?*}")" "$WHEEL"
else
    python3 -m pip download --no-deps --dest "$TEMP_DIR" "$WHEEL"
fi

WHEEL_FILE=$(find "$TEMP_DIR" -maxdepth 1 -name '*.whl' -type f | head -n 1)
[ -n "$WHEEL_FILE" ] || {
    echo "No wheel was downloaded." >&2
    exit 1
}

mkdir -p "$TEMP_DIR/extracted"
python3 -m zipfile -e "$WHEEL_FILE" "$TEMP_DIR/extracted"
mapfile -d '' LIBRARIES < <(
    find "$TEMP_DIR/extracted" \( -type f -o -type l \) \
        \( -name 'libonnxruntime.so*' -o -name 'libonnxruntime_providers_*.so*' \) -print0
)
[ "${#LIBRARIES[@]}" -gt 0 ] || {
    echo "The wheel contains no ONNX Runtime shared libraries." >&2
    exit 1
}

$SUDO mkdir -p "$LIB_DIR"
for library in "${LIBRARIES[@]}"; do
    $SUDO cp -L "$library" "$LIB_DIR/$(basename "$library")"
done

VERSIONED=$(find "$LIB_DIR" -maxdepth 1 -name 'libonnxruntime.so.*' -type f | head -n 1 || true)
if [ -n "$VERSIONED" ]; then
    $SUDO ln -sfn "$(basename "$VERSIONED")" "$LIB_DIR/libonnxruntime.so"
fi

if command -v ldconfig >/dev/null && [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    printf '%s\n' "$LIB_DIR" | $SUDO tee /etc/ld.so.conf.d/jetort.conf >/dev/null
    $SUDO ldconfig
fi

exec bash "$0" --check --prefix "$PREFIX"
