#!/usr/bin/env bash
# Run one federated round end to end.  >>> RUN THIS ON THE PI <<<
# The node must already be running (scripts/run_node.sh in another shell).
#
#   ./scripts/fl_trial.sh                 3 rounds, seeded corpus
#   ./scripts/fl_trial.sh --rounds 5
#   ./scripts/fl_trial.sh --no-bootstrap  only real labelled events
#
# Starts the aggregator, seeds samples, trains locally, joins the round,
# then reports whether the validation gate accepted the averaged model.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

API="http://localhost:8000"
ROUNDS=3; EPOCHS=4; SAMPLES=4000; BOOTSTRAP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --no-bootstrap) BOOTSTRAP=0; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

curl -sf -m 5 "$API/api/status" >/dev/null || {
  echo "The node is not answering on :8000. Start scripts/run_node.sh first." >&2
  exit 1
}

echo "== aggregator (${ROUNDS} rounds) =="
python sim/fl.py serve --rounds "$ROUNDS" --clients 1 --address 127.0.0.1:8080 \
  > /tmp/argus_fl_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 4

if [ "$BOOTSTRAP" = "1" ]; then
  # A round needs 200 labelled samples; a short capture gives a handful.
  # These rows are tagged synthetic and never appear in the event list.
  echo "== seeding ${SAMPLES} synthetic samples (disclose this) =="
  curl -sX POST "$API/api/federated/samples/bootstrap" \
       -H 'Content-Type: application/json' \
       -d "{\"site\":\"driveway\",\"n\":${SAMPLES}}" | python -m json.tool | head -12
fi

echo "== local training =="
curl -sX POST "$API/api/federated/head/train" \
     -H 'Content-Type: application/json' -d '{"epochs":6}' | python -m json.tool | head -20

echo "== joining the round =="
curl -sX POST "$API/api/federated/round" -H 'Content-Type: application/json' \
     -d "{\"server\":\"127.0.0.1:8080\",\"epochs\":${EPOCHS}}" | python -m json.tool

echo "== running (detection keeps going throughout) =="
for _ in $(seq 1 200); do
  python - <<'PY' && break || sleep 3
import json, sys, urllib.request
with urllib.request.urlopen("http://localhost:8000/api/federated/round", timeout=20) as r:
    s = json.load(r)
with urllib.request.urlopen("http://localhost:8000/api/detection/status", timeout=20) as r:
    w = (json.load(r).get("worker") or {})
print(f"  running={s['running']}  detect_fps={w.get('detect_fps')}")
sys.exit(0 if (not s["running"] and s.get("result")) else 1)
PY
done

echo
echo "== result =="
curl -s "$API/api/federated/round" | python -m json.tool
echo
echo "== node =="
curl -s "$API/api/status" | python -m json.tool | head -8
echo
echo "model_version changes only when the gate accepts the update."
echo "A rejection is the gate working: the averaged model scored worse here."
