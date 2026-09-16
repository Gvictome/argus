#!/usr/bin/env bash
# Start the ARGUS node.  >>> RUN THIS ON THE PI <<<
#
#   ./scripts/run_node.sh                  normal demo
#   ./scripts/run_node.sh --every-frame    classify stationary subjects too
#   ./scripts/run_node.sh --threat         add the dangerous-person stage
#   ./scripts/run_node.sh --model yolov8s  bigger, slower detector
#   ./scripts/run_node.sh --video media/vtest.avi   a recording instead of the camera
#
# Then open http://<this-pi>:8000/dashboard from any browser that can reach it.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] && source .venv/bin/activate

export NODE_NAME="${NODE_NAME:-Node A}"
export CAMERA_ZONE="${CAMERA_ZONE:-Front door}"
export AUTH_REQUIRED="${AUTH_REQUIRED:-false}"
export FL_ENABLED="${FL_ENABLED:-false}"          # rounds are triggered by hand
export THREAT_ENABLED="${THREAT_ENABLED:-false}"  # ~100ms/frame on the Pi's CPU
export RECORD_EVENTS="${RECORD_EVENTS:-true}"
export MOTION_GATE="${MOTION_GATE:-true}"
export OBJECT_MODEL="${OBJECT_MODEL:-yolov8n}"
# Next.js takes 3001 when 3000 is busy, and a browser sends the exact origin.
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001}"

while [ $# -gt 0 ]; do
  case "$1" in
    --every-frame) export MOTION_GATE=false; shift ;;
    --threat)      export THREAT_ENABLED=true; shift ;;
    --no-record)   export RECORD_EVENTS=false; shift ;;
    --model)       export OBJECT_MODEL="$2"; shift 2 ;;
    --video)       export CAMERA_SOURCE="$2"; shift 2 ;;
    --port)        export PORT="$2"; shift 2 ;;
    -h|--help)     sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "node   : ${NODE_NAME} (${CAMERA_ZONE})"
echo "model  : ${OBJECT_MODEL}   motion gate: ${MOTION_GATE}   threat: ${THREAT_ENABLED}"
echo "address: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT:-8000}/dashboard"
echo
exec python main.py
