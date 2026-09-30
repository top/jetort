#!/usr/bin/env bash
# deploy.sh used to compile, install a systemd unit, and start the server.
# Compilation is only scripts/build.sh. This file does not build.

echo "deploy.sh no longer builds JetOrt." >&2
echo "  Build:    ./scripts/build.sh --target <jetson|linux-nvidia|macos|windows-nvidia|windows-other|linux-amd|cpu>" >&2
echo "  Jetson:   ./scripts/jetson.sh" >&2
echo "  Service:  ./scripts/install-user-service.sh" >&2
exit 2
