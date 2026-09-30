#!/usr/bin/env bash
# Install the JetOrt user systemd unit. Does not compile and does not install libraries.

set -euo pipefail

[ "$EUID" -ne 0 ] || {
    echo "Do not run this script as root/sudo; it installs a user-level systemd service." >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
START_SERVICE=true

usage() {
    cat <<'EOF'
Usage: install-user-service.sh [--no-start]

Installs ~/.config/systemd/user/jetort.service for the binary at target/release/jetort.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-start) START_SERVICE=false ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

TARGET_BIN="$PROJECT_ROOT/target/release/jetort"
[ -x "$TARGET_BIN" ] || {
    echo "Binary not found at $TARGET_BIN. Build first: ./scripts/build.sh --target jetson" >&2
    exit 1
}

command -v systemctl >/dev/null || {
    echo "systemctl is unavailable; the binary was not wrapped in a service." >&2
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
