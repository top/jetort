#!/usr/bin/env bash
# Stop a runaway TensorRT recompile, restore quarantined embedding engines,
# sync local sources, rebuild, restart, and smoke-test embeddings.
set -euo pipefail

HOST="${JETORT_HOST:-ray@10.0.0.2}"
REMOTE="${JETORT_REMOTE:-/home/ray/jetort}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> stop jetort on $HOST"
ssh -o ConnectTimeout=12 -o BatchMode=yes "$HOST" 'systemctl --user stop jetort || true; sleep 1; pkill -f "[t]arget/release/jetort" 2>/dev/null || true; pkill -f "[/]home/ray/jetort" 2>/dev/null || true; sleep 1; pgrep -af jetort || echo "no jetort process"'

echo "==> restore quarantined TRT caches (do not delete engines)"
ssh -o BatchMode=yes "$HOST" bash -s <<'REMOTE_RESTORE'
set -euo pipefail
cd /home/ray/jetort
for q in .cache/*.quarantine; do
  [ -d "$q" ] || continue
  base="${q%.quarantine}"
  echo "found quarantine: $q -> $base"
  if [ -d "$base" ]; then
    # Prefer restoring engine files from quarantine into base if base is empty/incomplete
    if ! find "$base" -name '*.engine' 2>/dev/null | grep -q .; then
      echo "cache $base has no .engine; moving quarantine back"
      rm -rf "$base"
      mv "$q" "$base"
    else
      # Copy any engines from quarantine that are missing in base, then keep quarantine as backup
      echo "cache $base already has engine(s); copying missing files from quarantine"
      rsync -a --ignore-existing "$q"/ "$base"/
    fi
  else
    echo "restoring $q -> $base"
    mv "$q" "$base"
  fi
done
ls -la .cache/ || true
for d in .cache/Qwen3-Embedding-0.6B .cache/*Embedding*; do
  [ -d "$d" ] || continue
  echo "--- $d ---"
  ls -la "$d" | head -20
  [ -f "$d/jetort-trt-manifest.json" ] && cat "$d/jetort-trt-manifest.json" || true
done
REMOTE_RESTORE

echo "==> rsync sources (preserve device config, .cache, and model weights)"
# Device config.yaml is source of truth for enabled models — never overwrite it.
rsync -az --delete \
  --exclude 'config.yaml' \
  --exclude 'config*.yaml' \
  --exclude '.cache/' \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude '*.engine' \
  --exclude 'quant_pipeline/' \
  "$LOCAL/" "$HOST:$REMOTE/"

echo "==> rebuild on device (scripts/jetson.sh)"
ssh -o BatchMode=yes "$HOST" "cd $REMOTE && bash scripts/jetson.sh"

echo "==> service status + startup logs"
ssh -o BatchMode=yes "$HOST" 'systemctl --user is-active jetort; journalctl --user -u jetort -n 120 --no-pager | tee /tmp/jetort-start.log | grep -E "Qwen3-Embedding|compilation in progress|NvMap|TensorRT|print_timings|quarantine|position_ids|✨|fallback|Falling" || true'

echo "==> short embeddings smoke"
ssh -o BatchMode=yes "$HOST" 'curl -sS -w "\nHTTP:%{http_code}\n" http://127.0.0.1:11433/v1/embeddings -H "Content-Type: application/json" -d "{\"model\":\"Qwen3-Embedding-0.6B\",\"input\":\"hello world\"}" | head -c 400; echo; journalctl --user -u jetort -n 30 --no-pager | grep -E "print_timings|Qwen3-Embedding|compilation|NvMap|Falling|shape outside" || true'

echo "==> done"
