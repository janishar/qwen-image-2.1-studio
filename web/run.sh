#!/usr/bin/env bash
# Run qwen image studio on its own under `helm dev`, which keeps what it stores in ./.helm.
#
#   [QWEN_MODELS=<dir holding Qwen-Image-2.1, -PE-T2I, -PE-I2I>] [HELM=<helm>] bash web/run.sh [helm dev flags]
#   bash web/run.sh stop
#
# A run first stops the helm dev this script started before, so starting again is a restart.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HELM="${HELM:-helm}"
MODELS="${QWEN_MODELS:-$HOME/models}"
RUN="$PWD/.cache/qwen-studio"
PIDFILE="$RUN/qwen-studio.pid"

# Ends the helm dev this script last started, which stops the studio before it exits.
stop() {
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q " dev -f helmstudio.yaml"; then
    echo "helm dev: stopping the qwen image studio started before ($pid)"
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do sleep 0.2; done
  fi
  rm -f "$PIDFILE"
}
if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

if ! command -v "$HELM" >/dev/null; then
  echo "helm is not installed. Install it with helmstudio's installer, or set HELM to its path." >&2
  exit 1
fi
if ! .venv/bin/python -c "import helm_runtime_sdk" 2>/dev/null; then
  echo "helm-runtime-sdk is not installed in .venv: run uv sync" >&2
  exit 1
fi

stop
mkdir -p "$RUN"
echo $$ >"$PIDFILE" # exec keeps this pid, so it is helm dev's
exec "$HELM" dev -f helmstudio.yaml -venv .venv \
  -link "qwen_image=$MODELS/Qwen-Image-2.1" \
  -link "pe_t2i=$MODELS/Qwen-Image-2.1-PE-T2I" \
  -link "pe_i2i=$MODELS/Qwen-Image-2.1-PE-I2I" \
  "$@"
