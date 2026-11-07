#!/usr/bin/env bash
# Where does YOLO actually run on this node?  >>> RUN THIS ON THE PI <<<
#
#   ./scripts/hailo_check.sh
#
# Read-only. Answers four questions, in order, and prints one verdict:
#   1. Is the AI HAT+ 2 (Hailo-10H) on the PCIe bus?
#   2. Is the runtime installed and can it talk to the chip?
#   3. Is there a compiled Hailo model where ARGUS looks for one?
#   4. Which backend did the running node pick? (/api/status "accelerator")
# ARGUS uses the Hailo only when 1-3 all hold; otherwise it runs YOLO on the
# CPU (NCNN if exported). The verdict line is what goes on the poster.
set -uo pipefail
cd "$(dirname "$0")/.."

ok()   { printf '[ ok ] %s\n' "$*"; }
bad()  { printf '[FAIL] %s\n' "$*"; }
info() { printf '[info] %s\n' "$*"; }

board=0; runtime=0; model=0

echo "== 1. board on PCIe =="
if lspci 2>/dev/null | grep -i hailo; then ok "Hailo device on PCIe"; board=1
else bad "no Hailo device in lspci (HAT seated? PCIe ribbon? power off and reseat)"; fi
if ls -l /dev/hailo* 2>/dev/null; then info "device node present"; else info "no /dev/hailo* node"; fi

echo; echo "== 2. runtime =="
# Capture first: `dpkg -l | grep -q` under pipefail can report a false miss (SIGPIPE).
pkgs=$(dpkg -l 2>/dev/null | awk '/hailo/ {print $2, $3}')
[ -n "$pkgs" ] && printf '  %s\n' "$pkgs"
if grep -q '^hailo-h10-all ' <<<"$pkgs"; then ok "hailo-h10-all installed (right package for AI HAT+ 2)"
elif grep -q '^hailo-all ' <<<"$pkgs"; then bad "hailo-all is the Hailo-8/8L package; AI HAT+ 2 needs hailo-h10-all"
else info "no hailo apt package (sudo apt install hailo-h10-all && sudo reboot)"; fi
if command -v hailortcli >/dev/null; then
  if out=$(timeout 15 hailortcli fw-control identify 2>&1); then
    echo "$out" | sed 's/^/  /'; ok "hailortcli talks to the chip"; runtime=1
  else
    echo "$out" | tail -3 | sed 's/^/  /'; bad "hailortcli could not identify the board"
  fi
else
  bad "hailortcli not installed"
fi

echo; echo "== 3. compiled model where ARGUS looks (src/detection/__init__.py _HAILO_CANDIDATES) =="
for p in models/yolov8n_hailo_model models/yolo11n_hailo_model yolov8n_hailo_model; do
  if [ -d "$p" ] && [ -n "$(find "$p" -name '*.hef' -print -quit)" ]; then ok "found $p (with .hef)"; model=1
  elif [ -d "$p" ]; then bad "$p exists but holds no .hef: incomplete export"; fi
done
if [ -f yolov8n_hailo_model.hef ]; then ok "found yolov8n_hailo_model.hef"; model=1; fi
[ "$model" = 1 ] || bad "no Hailo export on this node (compile on x86 with scripts/export_hailo.py --hw-arch hailo10h)"
ls -d models/*ncnn_model 2>/dev/null | sed 's/^/  CPU export: /'
# Precompiled demo models shipped by the Hailo apt packages (not used by ARGUS as is).
if [ -d /usr/share/hailo-models ]; then
  info "precompiled HEFs from apt: $(ls /usr/share/hailo-models | tr '\n' ' ')"
fi

echo; echo "== 4. what the running node uses =="
accel=$(curl -s -m 5 localhost:8000/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("accelerator","?"), "fps", d.get("fps","?"))' 2>/dev/null)
if [ -n "$accel" ]; then info "accelerator from /api/status: $accel"
else info "node not answering on :8000 (start it to see the live backend)"; fi

echo
if [ "$board$runtime$model" = "111" ]; then
  echo "VERDICT: Hailo path is possible. Claim it only if step 4 says Hailo-10H."
elif [ "$board$runtime" = "11" ]; then
  echo "VERDICT: AI HAT+ 2 installed and working, but YOLO runs on the CPU: no compiled Hailo model."
else
  echo "VERDICT: YOLO runs on the CPU. The Hailo board is not usable yet (see FAIL lines)."
fi
