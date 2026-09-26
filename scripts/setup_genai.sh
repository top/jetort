#!/usr/bin/env bash
# Install or build ONNX Runtime GenAI. Does not install base ORT or deploy JetOrt.

set -euo pipefail

# ponytail: resolve $0 to an absolute path before any `cd` (e.g. into
# $SOURCE_DIR for the build) makes a relative invocation path unresolvable
# for the self-re-exec at the end of the script.
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# ponytail: this script escalates internally (via $SUDO) only for the file
# writes that need it. Running the whole script as root defeats that and
# leaves root-owned build/clone artifacts and a root-relative $HOME behind.
[ "$EUID" -ne 0 ] || {
    echo "Do not run this script as root/sudo; it escalates only the steps that need it." >&2
    exit 1
}

PREFIX="/usr/local"
CUDA_ARCH="${TARGET_CUDA_ARCH:-87}"
SOURCE_DIR="${GENAI_SRC_DIR:-$HOME/onnxruntime-genai}"
PACKAGE="${CONFIG_GENAI_URL:-}"
CHECK_ONLY=false
FORCE=false

usage() {
    cat <<'EOF'
Usage: setup_genai.sh [--check] [--force] [--prefix DIR] [--package PATH_OR_URL]
                      [--source DIR] [--cuda-arch NUMBER]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=true ;;
        --force) FORCE=true ;;
        --prefix) PREFIX="${2:?--prefix requires a directory}"; shift ;;
        --package) PACKAGE="${2:?--package requires a value}"; shift ;;
        --source) SOURCE_DIR="${2:?--source requires a directory}"; shift ;;
        --cuda-arch) CUDA_ARCH="${2:?--cuda-arch requires a value}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
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

# ponytail: release version is the suffix of the resolved .so filename
# (e.g. libonnxruntime-genai.so.0.6.0 -> 0.6.0). The core and CUDA add-on
# libraries must come from the same onnxruntime-genai release, or the
# add-on interface version check fails at inference time with a confusing
# native error. Catch the mismatch here instead.
so_version() {
    local library="$1"
    [ -n "$library" ] && [ -e "$library" ] || return 0
    local resolved
    resolved=$(readlink -f "$library" 2>/dev/null || printf '%s' "$library")
    [[ "$resolved" =~ \.so\.([0-9][0-9.]*)$ ]] && printf '%s' "${BASH_REMATCH[1]}"
}

check_version_match() {
    local core_version cuda_version
    core_version=$(so_version "$GENAI_LIB")
    cuda_version=$(so_version "$GENAI_CUDA_LIB")
    if [ -n "$core_version" ] && [ -n "$cuda_version" ] && [ "$core_version" != "$cuda_version" ]; then
        echo "[error] GenAI core ($core_version) and CUDA add-on ($cuda_version) versions differ." >&2
        echo "        Reinstall both from the same release, e.g.: $0 --force --package <matching release>" >&2
        return 1
    fi
    return 0
}

GENAI_LIB=$(find_library "libonnxruntime-genai.so")
GENAI_CUDA_LIB=$(find_library "libonnxruntime-genai-cuda.so")
NVIDIA_DEVICE=false
if [ -e /dev/nvidia0 ] || [ -e /dev/nvhost-gpu ] || command -v nvidia-smi >/dev/null; then
    NVIDIA_DEVICE=true
fi

if library_usable "$GENAI_LIB" \
    && { [ "$NVIDIA_DEVICE" = false ] || library_usable "$GENAI_CUDA_LIB"; } \
    && check_version_match; then
    echo "[ok] ONNX Runtime GenAI is available: $GENAI_LIB"
    [ -n "$GENAI_CUDA_LIB" ] && echo "[ok] GenAI CUDA provider: $GENAI_CUDA_LIB"
    if [ "$FORCE" = false ]; then
        echo "[skip] Existing GenAI installation is usable."
        exit 0
    fi
fi

if [ "$NVIDIA_DEVICE" = true ] && ! library_usable "$GENAI_CUDA_LIB"; then
    echo "[missing] NVIDIA device detected but the GenAI CUDA provider is unavailable."
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "[missing] A usable ONNX Runtime GenAI installation was not found." >&2
    exit 1
fi

ORT_LIB=$(find_library "libonnxruntime.so")
library_usable "$ORT_LIB" || {
    echo "A usable libonnxruntime.so is required. Run setup_onnxruntime.sh first." >&2
    exit 1
}

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/jetort-genai.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT

install_libraries() {
    local root="$1"
    mapfile -d '' libraries < <(
        find "$root" \( -type f -o -type l \) -name 'libonnxruntime-genai*.so*' -print0
    )
    [ "${#libraries[@]}" -gt 0 ] || return 1
    $SUDO mkdir -p "$LIB_DIR"
    for library in "${libraries[@]}"; do
        $SUDO cp -L "$library" "$LIB_DIR/$(basename "$library")"
    done
    local versioned=""
    versioned=$(find "$LIB_DIR" -maxdepth 1 -type f -name 'libonnxruntime-genai.so.*' | head -n 1 || true)
    if [ ! -e "$LIB_DIR/libonnxruntime-genai.so" ] && [ -n "$versioned" ]; then
        $SUDO ln -sfn "$(basename "$versioned")" "$LIB_DIR/libonnxruntime-genai.so"
    fi
}

if [ -n "$PACKAGE" ]; then
    if [ -d "$PACKAGE" ]; then
        install_libraries "$PACKAGE" || {
            echo "No GenAI shared libraries found in $PACKAGE" >&2
            exit 1
        }
    else
        ARCHIVE="$PACKAGE"
        if [[ "$PACKAGE" == http://* || "$PACKAGE" == https://* ]]; then
            command -v curl >/dev/null || {
                echo "curl is required to download $PACKAGE" >&2
                exit 1
            }
            ARCHIVE="$TEMP_DIR/$(basename "${PACKAGE%%\?*}")"
            curl -fL --retry 3 -o "$ARCHIVE" "$PACKAGE"
        fi
        case "$ARCHIVE" in
            *.tar.gz|*.tgz) tar -xzf "$ARCHIVE" -C "$TEMP_DIR" ;;
            *.whl|*.zip) python3 -m zipfile -e "$ARCHIVE" "$TEMP_DIR" ;;
            *.so|*.so.*) cp "$ARCHIVE" "$TEMP_DIR/" ;;
            *) echo "Unsupported GenAI package: $ARCHIVE" >&2; exit 1 ;;
        esac
        install_libraries "$TEMP_DIR" || {
            echo "The package contains no GenAI shared libraries." >&2
            exit 1
        }
    fi
else
    command -v git >/dev/null || { echo "git is required to build GenAI." >&2; exit 1; }
    command -v python3 >/dev/null || { echo "python3 is required to build GenAI." >&2; exit 1; }
    command -v curl >/dev/null || { echo "curl is required to obtain ORT headers." >&2; exit 1; }
    command -v nvcc >/dev/null || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] || {
        echo "nvcc is required to build GenAI with CUDA." >&2
        exit 1
    }

    if [ ! -d "$SOURCE_DIR/.git" ]; then
        git clone --depth 1 https://github.com/microsoft/onnxruntime-genai.git "$SOURCE_DIR"
    fi

    ORT_HOME="$TEMP_DIR/ort"
    mkdir -p "$ORT_HOME/lib" "$ORT_HOME/include"
    cp -L "$ORT_LIB" "$ORT_HOME/lib/libonnxruntime.so"

    HEADER_ROOT="/usr/local/include/onnxruntime"
    if [ -d "$HEADER_ROOT" ]; then
        cp -R "$HEADER_ROOT/." "$ORT_HOME/include/"
    else
        # ponytail: a hand-picked file list here keeps breaking whenever
        # onnxruntime_c_api.h grows a new #include (e.g. onnxruntime_error_code.h).
        # Sparse-clone the whole session header directory instead, so new
        # headers are picked up automatically.
        echo "[download] ONNX Runtime C/C++ headers"
        HEADER_CLONE="$TEMP_DIR/onnxruntime-headers"
        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/microsoft/onnxruntime.git "$HEADER_CLONE"
        git -C "$HEADER_CLONE" sparse-checkout set include/onnxruntime/core/session
        cp -R "$HEADER_CLONE/include/onnxruntime/core/session/." "$ORT_HOME/include/"
    fi

    cd "$SOURCE_DIR"
    BUILD_ARGS=(
        --config Release
        --update
        --build
        --parallel
        --skip_tests
        --skip_examples
        --skip_wheel
        --use_cuda
        --cuda_home "${CUDA_HOME:-/usr/local/cuda}"
        --ort_home "$ORT_HOME"
        --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    )
    command -v ninja >/dev/null && BUILD_ARGS+=(--cmake_generator Ninja)
    python3 build.py "${BUILD_ARGS[@]}"
    install_libraries "$SOURCE_DIR/build" || {
        echo "Build completed but produced no GenAI shared libraries." >&2
        exit 1
    }
fi

if command -v ldconfig >/dev/null && [[ "$PREFIX" == /usr* || "$PREFIX" == /opt* ]]; then
    $SUDO ldconfig
fi

exec bash "$SCRIPT_PATH" --check --prefix "$PREFIX"
