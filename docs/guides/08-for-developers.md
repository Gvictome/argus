# 8 · For developers

Where things live, how to change them, and what is genuinely unfinished.

---

## Two repositories

| Repo | Purpose | Branch rule |
|---|---|---|
| `Gvictome/argus` | Backend: camera, detection, events, learning, API, built-in page | Work on **`feat/threat-detection`** |
| `adam9798/argus-dashboard` | The product dashboard (Next.js) | **Work on your branch and open a pull request. Never push to its main.** |

The backend repo's default branch on GitHub is `master`, which is a stale
mirror with none of this work. Always check out `feat/threat-detection`.

---

## Layout

```
src/
  camera/       opening cameras and video files, per board
  detection/    motion, YOLO, tracking, annotation, recording
    worker.py   the background loop that owns the camera
  federated/    the learned head, samples, rounds, the gate
    collector.py    detections -> events -> training rows
    model_state.py  the validation gate and version history
    node_client.py  this node's participant in a round
  api/          routes, the dashboard contract, trial endpoints
scripts/        one-command runners and tooling
sim/            the node simulator (Flower, timing model)
docs/           documentation; docs/guides are these guides
static/         the built-in live page (one HTML file)
```

---

## Running it

| Task | Command |
|---|---|
| Node, on a Pi | `./scripts/run_node.sh` |
| Node against a video, anywhere | `python scripts/mock_demo.py --reset` |
| Tests | `python -m pytest -q` |
| A federated round | `./scripts/fl_trial.sh` |
| What the detector sees | `python scripts/check_classes.py --camera` |
| Build these PDFs | `python scripts/build_docs_pdf.py --guides` |

Tests are fast (~30s) because nothing heavyweight loads at import.

**Gate any push on the test command's exit code, not on reading its
output.** A grep for "failed" matches a passing summary line too, which is
how two failing tests once got pushed.

---

## Settings that change behaviour

| Variable | Default | Effect |
|---|---|---|
| `MOTION_GATE` | `true` | `false` runs the detector on every frame |
| `OBJECT_MODEL` | `yolov8n` | Larger models need a matching export |
| `OBJECT_BACKEND` | `auto` | Picks an optimised export if present |
| `CAMERA_SOURCE` | camera | A video file, for repeatable runs |
| `ARGUS_DATA_DIR` | `data` | Moves database, samples and model together |
| `CAMERA_SWAP_RB` | `false` | Only if colours look wrong |
| `THREAT_ENABLED` | `true` | Costly stage; off for demos |

---

## Design decisions worth knowing before changing things

**The camera is owned by one background worker.** Nothing else may read it
directly — a second reader gets empty frames. Anything needing a picture
goes through the worker's latest frame.

**One event per track, not per frame.** Otherwise a person standing still
generates hundreds of near-identical rows and teaches the model that
whatever loiters longest matters most.

**The still is captured at the subject's largest**, not when the track
closes — by then they are half out of frame.

**A shared model must pass the gate.** Never apply a federated result
directly.

**Exports have a fixed input size.** Asking a 640-pixel export for 416
returns nothing at all, very quickly — which reads as an enormous speed-up.
The code now forces the export's own size.

---

## What has actually been measured

On a laptop, limited to four cores, on a pedestrian video:

| Change | Result |
|---|---|
| Removing face recognition | 0.86 → 23.4 frames per second |
| Smaller working frame for motion | 17.8 → 22.9 at 1080p |
| Smaller detector input | ~60% faster, ~5% fewer detections |

**On the Pi, only two things are measured:** the accelerator runs a large
detector at 76 frames per second through its own tool, and the converted
detector works. No end-to-end frame rate on the Pi has been recorded. Do not
quote laptop numbers as Pi numbers.

---

## Open work

1. **Connect the accelerator.** ARGUS does not use the Hailo yet. It needs
   the detector compiled into the chip's format, which requires an x86
   Linux machine. Until then the Pi uses its processor.
2. **Measure on the Pi.** Every performance number so far is from a laptop.
3. **Parcels.** "Package" currently means backpack, handbag or suitcase.
   Real doorstep parcels need a custom-trained detector.
4. **Averaging across dissimilar sites** produced a worse model in testing.
   The gate catches it; making it actually work is research.
5. **Dashboard authentication.** The dashboard does not send its token to
   the backend. Fine on a private network, must be fixed before exposing
   anything to the internet.

---

## Documentation

- `docs/guides/` — these guides, plain language, one topic each
- `docs/QUICKSTART.md` — every command, labelled by machine
- `docs/OVERVIEW.md` — the system in one pass, for a technical reader
- `docs/DEMO_RUNBOOK.md` — the full runbook with measured numbers
- `sim/README.md` — the simulator, and what it can and cannot prove
