#!/usr/bin/env bash
# Is this node ready for node day?  >>> RUN THIS ON THE PI <<<
#
#   ./scripts/node_preflight.sh               all checks, read-only
#   ./scripts/node_preflight.sh --fix-perms   also chmod 600 deploy/node.env
#
# Prints [ ok ] / [WARN] / [FAIL] per check and exits 1 if any FAIL.
# Never prints the password: only whether it is set and long enough.
set -uo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=deploy/node.env
MIN_PW_LEN=12
fails=0
ok()   { printf '[ ok ] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
bad()  { printf '[FAIL] %s\n' "$*"; fails=$((fails + 1)); }

# Value of KEY in node.env, without sourcing the file (it holds a secret).
envval() { grep -E "^[[:space:]]*$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^"//' -e 's/"$//'; }

echo "== code =="
echo "  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null), $(git log -1 --format='%h %s' 2>/dev/null)"
dirty=$(git status --short --untracked-files=no 2>/dev/null | wc -l)
if [ "$dirty" = 0 ]; then ok "no local edits to tracked files"; else warn "$dirty tracked file(s) edited locally (git status)"; fi
if git merge-base --is-ancestor fb3dd05 HEAD 2>/dev/null; then ok "Gate B fixes present (fb3dd05)"
else warn "Gate B fixes (fb3dd05) not in this checkout"; fi

echo; echo "== deploy/node.env =="
if [ ! -f "$ENV_FILE" ]; then
  bad "missing: cp deploy/node.env.example deploy/node.env, then edit it"
else
  perm=$(stat -c '%a' "$ENV_FILE")
  if [ "$perm" = 600 ]; then ok "permissions 600"
  elif [ "${1:-}" = "--fix-perms" ]; then chmod 600 "$ENV_FILE" && ok "permissions fixed to 600"
  else bad "permissions $perm (should be 600; rerun with --fix-perms)"; fi

  name=$(envval NODE_NAME)
  if [ -n "$name" ]; then ok "NODE_NAME=$name (must differ on every node)"; else bad "NODE_NAME not set"; fi
  if [ "$(envval AUTH_REQUIRED)" = "true" ]; then ok "AUTH_REQUIRED=true"; else bad "AUTH_REQUIRED is not true"; fi
  pw=$(envval ARGUS_ADMIN_PASSWORD)
  if [ -z "$pw" ]; then bad "ARGUS_ADMIN_PASSWORD not set"
  elif [ "${#pw}" -lt "$MIN_PW_LEN" ]; then bad "ARGUS_ADMIN_PASSWORD shorter than $MIN_PW_LEN characters"
  else ok "ARGUS_ADMIN_PASSWORD set (${#pw} chars)"; fi
  unset pw
  if [ "$(envval FL_ENABLED)" = "true" ]; then ok "FL_ENABLED=true"; else bad "FL_ENABLED is not true"; fi

  fl=$(envval FL_SERVER_URL); api=$(envval FL_CENTRAL_API)
  echo "  FL_SERVER_URL=$fl  FL_CENTRAL_API=$api"
  host=${fl%:*}; port=${fl##*:}
  if [ -n "$host" ] && timeout 3 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then ok "aggregator reachable at $fl"
  else warn "aggregator not reachable at $fl (not started yet, wrong IP, or PC firewall)"; fi
fi

echo; echo "== python =="
if [ -x .venv/bin/python ]; then
  ok ".venv present"
  # flwr/ultralytics/cv2/numpy are required; picamera2 only on camera nodes.
  if ! .venv/bin/python - <<'PY'
import importlib, sys
missing = []
for mod, required in (("flwr", 1), ("ultralytics", 1), ("cv2", 1), ("numpy", 1), ("picamera2", 0)):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod:12s} MISSING ({exc.__class__.__name__})")
        if required:
            missing.append(mod)
sys.exit(1 if missing else 0)
PY
  then bad "required Python package missing (pip install -r requirements.txt)"; fi
else
  bad ".venv missing (docs/guides/02-set-up-the-pi.md step 3)"
fi

echo; echo "== camera =="
if command -v rpicam-hello >/dev/null; then
  cams=$(timeout 10 rpicam-hello --list-cameras 2>&1)
  if grep -qE '^[0-9]+ :' <<<"$cams"; then ok "camera listed"
  else warn "no camera listed (busy because the node is running? otherwise reseat the ribbon)"; fi
fi

echo; echo "== service =="
if systemctl is-enabled argus-node >/dev/null 2>&1; then ok "argus-node enabled"; else warn "argus-node not installed/enabled"; fi
if systemctl is-active argus-node >/dev/null 2>&1; then ok "argus-node running"; else warn "argus-node not running"; fi
status=$(curl -s -m 5 localhost:8000/api/status)
if [ -n "$status" ]; then
  # No quotes inside f-string braces: Bookworm's Python 3.11 rejects that.
  echo "$status" | python3 -c '
import json, sys
d = json.load(sys.stdin)
s = d.get("samples") or {}
vals = [d.get(k) for k in ("node", "accelerator", "fps", "model_version")] + [s.get("labelled")]
print("  node=%s accel=%s fps=%s model=%s labelled=%s" % tuple(vals))'
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 localhost:8000/api/detection/status)
  if [ "$code" = 401 ]; then ok "auth enforced (anonymous request got 401)"
  else bad "anonymous request got $code, expected 401: auth is OFF"; fi
else
  warn "API not answering on :8000"
fi

echo; echo "== health =="
vcgencmd measure_temp 2>/dev/null | sed 's/^/  /'
thr=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
if [ "$thr" = "0x0" ]; then ok "no undervoltage or throttling"; else warn "get_throttled=$thr (undervoltage or heat; use the 27 W PSU)"; fi
df -h / | awk 'NR==2 {print "  disk: " $3 " used of " $2 " (" $5 ")"}'
if timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes; then ok "clock synced"; else warn "clock not NTP-synced (check: date)"; fi
journal=$(grep -hs '^Storage=' /etc/systemd/journald.conf /etc/systemd/journald.conf.d/*.conf | tail -1)
echo "  journald ${journal:-Storage=auto (default: logs written to the SD card)}"

echo
if [ "$fails" = 0 ]; then echo "PREFLIGHT PASSED"; else echo "PREFLIGHT: $fails FAIL(s)"; fi
exit $(( fails > 0 ))
