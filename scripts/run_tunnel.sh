#!/usr/bin/env bash
#
# Start the Cloudflare Tunnel for ARGUS, but only if it is safe to.
#
# The tunnel publishes a live camera feed and the controls to enroll and
# delete faces. This script exists because the failure mode it prevents
# is silent: a tunnel started with AUTH_REQUIRED unset looks identical to
# a working one, right up until someone finds the URL.
#
# Usage:
#   scripts/run_tunnel.sh              # preflight, then run
#   scripts/run_tunnel.sh --check      # preflight only
#
set -euo pipefail

CONFIG="${CLOUDFLARED_CONFIG:-$HOME/.cloudflared/config.yml}"
API_URL="${ARGUS_URL:-http://localhost:8000}"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

fail() { echo "  [FAIL] $*" >&2; FAILED=1; }
ok()   { echo "  [ ok ] $*"; }
warn() { echo "  [warn] $*"; }

FAILED=0
echo "ARGUS tunnel preflight"
echo "----------------------"

# --- 1. Is ARGUS running? --------------------------------------------
if ! curl -fsS -m 5 "$API_URL/api/status" >/dev/null 2>&1; then
    fail "ARGUS is not responding at $API_URL — start it first (python main.py)"
else
    ok "ARGUS is up at $API_URL"
fi

# --- 2. Is app-level auth actually enforced? -------------------------
# Asking the API rather than reading the environment: the running server
# may have been started from a different shell, and its opinion is the
# one that matters.
if [[ $FAILED -eq 0 ]]; then
    ME="$(curl -fsS -m 5 "$API_URL/api/auth/me" 2>/dev/null || echo '{}')"
    if echo "$ME" | grep -q '"auth_required":true'; then
        ok "AUTH_REQUIRED is on"
    else
        fail "AUTH_REQUIRED is OFF — refusing to expose an open camera feed.
         Restart ARGUS with:
             AUTH_REQUIRED=true ARGUS_ADMIN_PASSWORD='<a real password>' python main.py"
    fi

    # Prove it, rather than trusting the flag: an unauthenticated read of
    # a protected endpoint must be refused.
    CODE="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API_URL/api/faces" || echo 000)"
    if [[ "$CODE" == "401" ]]; then
        ok "protected endpoints reject unauthenticated requests"
    else
        fail "GET /api/faces returned $CODE, expected 401 — auth is not enforced"
    fi
fi

# --- 3. Secrets --------------------------------------------------------
if [[ "${SECRET_KEY:-}" == "dev-secret-change-in-production" || -z "${SECRET_KEY:-}" ]]; then
    warn "SECRET_KEY is unset or the development default — set a real one"
fi
if [[ -n "${ARGUS_ADMIN_PASSWORD:-}" && ${#ARGUS_ADMIN_PASSWORD} -lt 12 ]]; then
    warn "ARGUS_ADMIN_PASSWORD is under 12 characters"
fi

# --- 4. cloudflared ----------------------------------------------------
if ! command -v cloudflared >/dev/null 2>&1; then
    fail "cloudflared is not installed — see config/cloudflared-argus.yml"
else
    ok "cloudflared $(cloudflared --version 2>/dev/null | head -1)"
fi

if [[ ! -f "$CONFIG" ]]; then
    fail "no tunnel config at $CONFIG
         cp config/cloudflared-argus.yml $CONFIG   (then edit the hostname)"
else
    ok "config found at $CONFIG"
    if grep -q "yourdomain.com" "$CONFIG" 2>/dev/null; then
        fail "$CONFIG still has the placeholder hostname 'yourdomain.com'"
    fi
    CREDS="$(grep -E '^credentials-file:' "$CONFIG" | awk '{print $2}' || true)"
    if [[ -n "$CREDS" && ! -f "$CREDS" ]]; then
        fail "credentials file not found: $CREDS  (run: cloudflared tunnel create argus-demo)"
    fi
fi

echo
if [[ $FAILED -ne 0 ]]; then
    echo "Preflight FAILED — not starting the tunnel." >&2
    exit 1
fi
echo "Preflight passed."
echo
echo "Reminder: app-level auth is the second gate, not the first."
echo "Add a Cloudflare Access policy for this hostname so unauthenticated"
echo "requests never reach the Pi. Zero Trust -> Access -> Applications."
echo

[[ $CHECK_ONLY -eq 1 ]] && exit 0

echo "Starting tunnel..."
exec cloudflared tunnel --config "$CONFIG" run
