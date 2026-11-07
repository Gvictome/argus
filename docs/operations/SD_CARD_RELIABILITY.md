# SD card reliability

One SD card has already been corrupted by an unclean stop. At the showcase a
node runs for hours, gets power-cycled, and sees a constant stream of
people. This page is what keeps the card alive, and what to do when it dies
anyway. Written 2026-11-06 for node day (11-07) and the showcase (11-22).

## 1. What writes to the card today

Checked in the code on `fix/gate-b` (fb3dd05):

| Writer | How often | Notes |
|---|---|---|
| Detection worker | never per frame | Frames stay in RAM; the MJPEG stream is encoded in memory |
| Event collector | once per event | One JPEG still + one crop in `data/snapshots/`, one line in `data/training/samples.jsonl`, SQLite rows |
| Event recorder (`RECORD_EVENTS=true`) | every frame **while a clip records** | Up to 120 s per clip, `media/events/`. Nothing deletes old clips yet |
| System journal | every log line | Pi OS keeps it on the card by default |
| uvicorn | warnings only | `log_level="warning"`, no access log |

So there are no per-frame writes in normal detection. The two heavy
writers are event clips and the journal. At a busy booth "an event" is
almost always happening, so clip recording becomes close to continuous.

**Showcase setting:** `RECORD_EVENTS=false` in `deploy/node.env` unless clip
playback is part of the demo. If it is, keep it on and check free space
(`df -h /`) at every break.

## 2. Staged hardening (not applied)

`deploy/sd-hardening/apply_sd_hardening.sh`. Without arguments it only
prints what it would do.

| Step | Effect | When |
|---|---|---|
| journald in RAM (`Storage=volatile`, 64 MB) | No log writes to the card | Immediately |
| `/tmp` as tmpfs (256 MB) | Temp files in RAM | After reboot |
| `noatime` on `/` | Reading a file no longer writes its access time | After reboot |
| Swap file off (8 GB Pis only) | No swap writes | Immediately |

```bash
./deploy/sd-hardening/apply_sd_hardening.sh               # dry run
sudo ./deploy/sd-hardening/apply_sd_hardening.sh --apply  # do it
sudo reboot
./scripts/node_preflight.sh                               # journald line should say volatile
sudo ./deploy/sd-hardening/apply_sd_hardening.sh --undo   # if anything misbehaves
```

`/etc/fstab` is backed up before any change and checked with
`findmnt --verify`; a failed check restores the backup. Apply it on one node,
reboot, confirm it comes back, then do the others.

The cost of volatile logs: they vanish on reboot. Save them first when it matters:
`journalctl -u argus-node -b > ~/argus-node-$(date +%F-%H%M).log`.

## 3. Clean shutdown, every time

* Press the Pi 5 **power button once**. Pi OS shuts down cleanly. Or:
  `sudo systemctl stop argus-node && sudo poweroff`.
* Wait until the green LED stops flashing (about 10 s) before pulling power.
* The service gives the node 45 s to close the camera and finish a clip
  (`TimeoutStopSec=45` in `deploy/argus-node.service`).
* At the booth, power every node from one strip you control, taped down.
  Do not let anyone "just unplug it".
* After an unexpected power cut, check the next boot is clean:
  `journalctl -b -p err` and `./scripts/node_preflight.sh`.

## 4. Which card

Use a **high-endurance** card, 64 GB, A2 if possible, from a real retailer
(fake cards are common on marketplaces):

* Samsung PRO Endurance 64 GB
* SanDisk MAX Endurance 64 GB (or High Endurance)

These are rated for continuous video recording, which is the closest
match to a camera node. The Raspberry Pi branded A2 card is fast but not
sold as high-endurance. Buying cards is Geo's call (money).

## 5. The showcase spare: byte image of a known-good card

Do this once a node passes node day (preflight passed, round 1 done).
The image is a complete copy: OS, code, `deploy/node.env` (with the admin
password), enrolled faces and events. Treat it as secret: keep it on the
PC or an encrypted drive, not in a shared folder.

**Make the image (Windows PC, free tools):**

1. Shut the node down cleanly (section 3). Take the card out.
2. Put it in the PC's card reader. **If Windows offers to format a disk,
   click Cancel.** That is the Linux partition; formatting it destroys the card.
3. Win32 Disk Imager: Image File `D:\argus-images\node-A-2026-11-07.img`,
   tick **Read Only Allocated Partitions**, click **Read**.
4. Record a checksum next to it:
   `Get-FileHash D:\argus-images\node-A-2026-11-07.img -Algorithm SHA256 | Out-File D:\argus-images\node-A-2026-11-07.sha256.txt`
5. Put the card back in the node and boot it. It should come up unchanged.

**Make the spare card:**

1. Raspberry Pi Imager: *Choose OS* > *Use custom* > the `.img`; pick the
   spare card; write. Skip the customisation prompt (it would change
   settings). Imager verifies after writing.
2. Label the spare: "ARGUS node A spare 11-07".
3. **Boot-test it**: put it in node A, power on, run
   `./scripts/node_preflight.sh` and `.\scripts\check_nodes.ps1`. A spare
   that has never booted is not a spare.
4. Put the original card back.

**Notes:**

* The spare is a clone of node A: same hostname, SSH keys and `NODE_NAME`.
  Use it only in node A's Pi, never at the same time as the original.
  The router gives the same IP because DHCP reservations follow the Pi's
  network hardware, not the card.
* The spare card must be the same size or larger than the original.
* Redo the image after any change you want to keep (new code, re-enrolled faces).
* Alternative without a PC reader: `rpi-clone` on the Pi with the spare in
  a USB card reader. Only if Win32 Disk Imager is not available.
