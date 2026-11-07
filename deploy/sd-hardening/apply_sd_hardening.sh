#!/usr/bin/env bash
# SD card wear hardening for an ARGUS node.  >>> RUN THIS ON THE PI <<<
#
#   ./deploy/sd-hardening/apply_sd_hardening.sh            DRY RUN: show the plan, change nothing
#   sudo ./deploy/sd-hardening/apply_sd_hardening.sh --apply
#   sudo ./deploy/sd-hardening/apply_sd_hardening.sh --undo
#
# What --apply does (each step is skipped if already true):
#   1. journald in RAM (Storage=volatile, 64 MB cap)        takes effect now
#   2. /tmp as a 256 MB tmpfs                                after reboot
#   3. noatime on the root filesystem                        after reboot
#   4. swap file off, only on 8 GB Pis (dphys-swapfile)      takes effect now
# /etc/fstab is backed up first and checked with `findmnt --verify`; if the
# check fails the backup is put back. --undo reverses all four steps.
# Do it on ONE node first, reboot it, run scripts/node_preflight.sh, then the rest.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOURNAL_SRC="$HERE/journald-argus.conf"
JOURNAL_DST=/etc/systemd/journald.conf.d/90-argus-volatile.conf
STATE=/etc/argus-sd-hardening.state
FSTAB=/etc/fstab
TMPFS_LINE='tmpfs /tmp tmpfs defaults,noatime,nosuid,nodev,size=256m 0 0'
MIN_RAM_KB_FOR_NO_SWAP=7000000

MODE=dry
case "${1:-}" in
  --apply) MODE=apply ;;
  --undo)  MODE=undo ;;
  ""|--dry-run) MODE=dry ;;
  *) echo "usage: $0 [--apply|--undo|--dry-run]" >&2; exit 2 ;;
esac
if [ "$MODE" != dry ] && [ "$(id -u)" != 0 ]; then echo "run with sudo for --apply/--undo" >&2; exit 1; fi

say() { if [ "$MODE" = dry ]; then echo "[plan] $*"; else echo "[do]   $*"; fi; }
root_opts() { findmnt -no OPTIONS /; }
ram_kb() { awk '/MemTotal/ {print $2}' /proc/meminfo; }
swap_active() { systemctl is-active dphys-swapfile >/dev/null 2>&1; }

backup_fstab() {
  local b
  b="$FSTAB.argus-bak-$(date +%Y%m%d-%H%M%S)"
  cp -a "$FSTAB" "$b"
  echo "fstab_backup=$b" >> "$STATE"
  echo "$b"
}

# findmnt --verify can report warnings on a stock Pi fstab (non-zero exit),
# so judge the edit against the original's own result, not against zero:
# fail only if the edited file produces messages the original did not.
verify_or_restore() {
  local backup="$1" before after
  before=$(findmnt --verify --tab-file "$backup" 2>&1 | grep -viE 'summary|^[[:space:]]*$' || true)
  after=$(findmnt --verify --tab-file "$FSTAB" 2>&1 | grep -viE 'summary|^[[:space:]]*$' || true)
  if [ "$before" != "$after" ]; then
    cp -a "$backup" "$FSTAB"
    echo "fstab check FAILED (new findmnt messages), original restored from $backup:" >&2
    diff <(echo "$before") <(echo "$after") >&2 || true
    exit 1
  fi
}

apply_journald() {
  if [ -f "$JOURNAL_DST" ]; then echo "[skip] journald already volatile"; return; fi
  say "install $JOURNAL_DST (journal in RAM) and restart systemd-journald"
  if [ "$MODE" = apply ]; then
    install -D -m 644 "$JOURNAL_SRC" "$JOURNAL_DST"
    systemctl restart systemd-journald
  fi
}

apply_fstab() {
  local need_tmp=0 need_noatime=0 b
  if [ "$(findmnt -no FSTYPE /tmp 2>/dev/null)" != tmpfs ] && \
     ! grep -qE '^[^#]*[[:space:]]/tmp[[:space:]]' "$FSTAB"; then need_tmp=1; fi
  if [[ "$(root_opts)" != *noatime* ]] && grep -qE '^[^#]*[[:space:]]/[[:space:]]' "$FSTAB"; then need_noatime=1; fi
  if [ "$need_tmp" = 1 ]; then say "add to $FSTAB: $TMPFS_LINE"; else echo "[skip] /tmp already tmpfs or in fstab"; fi
  if [ "$need_noatime" = 1 ]; then say "add noatime to the / entry in $FSTAB"; else echo "[skip] / already noatime"; fi
  [ "$MODE" = apply ] && [ $((need_tmp + need_noatime)) -gt 0 ] || return 0

  b=$(backup_fstab)
  # Build the new file from the untouched backup, then replace in one write.
  awk -v fix="$need_noatime" 'BEGIN{OFS="\t"}
       fix==1 && !/^[[:space:]]*#/ && $2=="/" && $4 !~ /noatime/ {$4=$4",noatime"} {print}' "$b" > "$FSTAB.argus-new"
  [ "$need_tmp" = 1 ] && echo "$TMPFS_LINE" >> "$FSTAB.argus-new"
  cat "$FSTAB.argus-new" > "$FSTAB"
  rm -f "$FSTAB.argus-new"
  verify_or_restore "$b"
  echo "       fstab updated and verified (backup $b). Reboot to apply."
}

apply_swap() {
  if ! swap_active; then echo "[skip] no dphys-swapfile swap active"; return; fi
  if [ "$(ram_kb)" -lt "$MIN_RAM_KB_FOR_NO_SWAP" ]; then
    echo "[keep] swap file left on: under 8 GB RAM, the node may need it"; return
  fi
  say "turn off the swap file (dphys-swapfile) on this 8 GB Pi"
  if [ "$MODE" = apply ]; then
    dphys-swapfile swapoff || true
    systemctl disable --now dphys-swapfile
    echo "swapfile_disabled=1" >> "$STATE"
  fi
}

do_undo() {
  local b
  if [ -f "$JOURNAL_DST" ]; then
    rm -f "$JOURNAL_DST"; systemctl restart systemd-journald; echo "[undo] journald back to default"
  fi
  b=$(grep -s '^fstab_backup=' "$STATE" | head -1 | cut -d= -f2- || true)
  if [ -n "$b" ] && [ -f "$b" ]; then cp -a "$b" "$FSTAB"; echo "[undo] fstab restored from $b (reboot to apply)"; fi
  if grep -qs '^swapfile_disabled=1' "$STATE"; then
    systemctl enable --now dphys-swapfile; echo "[undo] swap file back on"
  fi
  rm -f "$STATE"
}

echo "ARGUS SD hardening: mode=$MODE  RAM=$(( $(ram_kb) / 1024 )) MB  / opts=$(root_opts)"
if [ "$MODE" = undo ]; then
  do_undo
else
  apply_journald
  apply_fstab
  apply_swap
fi
if [ "$MODE" = dry ]; then echo "Dry run only. Nothing changed. To apply: sudo $0 --apply"; fi
exit 0
