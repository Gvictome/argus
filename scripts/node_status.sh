#!/usr/bin/env bash
# Watch the node.  >>> RUN THIS ON THE PI <<<
#
#   ./scripts/node_status.sh          one line per second, Ctrl+C to stop
#   ./scripts/node_status.sh --once   a single reading
#
# fps/ms are real detection throughput. tracks rises the moment something
# is in frame; events rises once it leaves.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

while true; do
  python - <<'PY' || echo "  node not answering on :8000"
import json, urllib.request
try:
    with urllib.request.urlopen("http://localhost:8000/api/detection/status", timeout=10) as r:
        d = json.load(r)
    with urllib.request.urlopen("http://localhost:8000/api/status", timeout=10) as r:
        s = json.load(r)
except Exception as exc:
    raise SystemExit(f"  {exc}")
w, c = d.get("worker") or {}, d.get("collector") or {}
print(f"fps={w.get('detect_fps', 0):5.1f} ms={w.get('detect_ms', 0):5.0f} "
      f"cap={w.get('capture_fps', 0):5.1f} | tracks={c.get('open_tracks', 0):2d} "
      f"events={c.get('emitted', 0):4d} motion={c.get('motion_events', 0):3d} "
      f"stills={c.get('snapshots_saved', 0):4d} | {d.get('backend')} {d.get('model')} "
      f"gate={d.get('motion_gate')} rate={d.get('motion_rate')} | {s.get('model_version')}"
      + (f" | CAMERA: {w.get('camera_error')}" if w.get("camera_error") else ""))
PY
  [ "$ONCE" = "1" ] && break
  sleep 1
done
