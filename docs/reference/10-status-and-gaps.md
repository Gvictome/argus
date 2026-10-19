# 10 · Status, known gaps, and what to do next

As of 2026-10-18. Showcase 2026-11-22; feature freeze 2026-11-08.

**The second node has not arrived.** Every federated claim for the showcase is
made with one real node plus simulated MEVA peers (section 10.6).

---

## 10.1 What works and has been run

| Capability | Evidence |
|---|---|
| Camera → motion → YOLO → overlay → live stream, continuous and unattended | First end-to-end hardware run on the Pi, 2026-09-20 |
| Event tracking, one event per track, stills and crops | 322 hand-labelled real events from that run |
| Event-triggered clips with pre- and post-roll | `media/events/` on the Pi; 18 tests |
| Threat stage | 0.926 precision, 0.796 recall at the chosen operating point (section 4.3) |
| Local head training through the gate | API and tests |
| Federated round Pi ↔ laptop aggregator | 2 rounds on 2026-09-20, balanced accuracy 0.75 |
| Multi-client Flower simulation | 3 clients × 3 rounds on 2026-10-02, section 2.7 |
| Robust aggregation in the Flower servers | Wired 2026-10-05; 3 clients × 2 rounds over gRPC through `RobustFedAvg`; 11 strategy tests (section 5.10) |
| Gate loss tie-break | 2 tests; the 2026-09-20 round-2 case is now rejected (section 5.6) |
| Fine-taxonomy make classifier | 0.832 over 49 makes on Stanford Cars; FedAvg balanced accuracy 0.860 → 0.894 in simulation (section 5.14) |
| Model versions and rollback | Node archive, rollback API, aggregator version resume; 8 tests (sections 5.6, 5.11) |
| Cron retries a round that never happened | 2 tests through the real runner path (section 5.9) |
| MEVA benchmark | 9 cameras converted to 19-feature events with whole-clip holdout (`scripts/meva_to_events.py`) |
| Federated vs base, local-only, FedAvg, fine-tune | 10 seeds with 95% intervals: federation alone loses to local-only (-0.047 macro), federation + fine-tune ties it (0.727 vs 0.723) and beats the no-head baseline (0.612). `docs/research/fl-vs-base-meva.md` |
| Hybrid demo (one real node + simulated MEVA peers) | `scripts/demo_hybrid.py`; run end to end on the laptop with the stand-in node, 7 clients, a poisoned peer rejected every round; upgrades to two real nodes with `--real-nodes 2` |
| Base detector decision | YOLOv8n locked 2026-10-18 (`docs/research/detector-comparison.md`) |
| Test suite | 547 passed, 3 skipped, 0 failed on 2026-10-18 (with media/vtest.avi present; the previously known failure passes) |

## 10.2 Gaps that block the research claim

| Gap | Why it matters | Fix |
|---|---|---|
| **One physical node** (parts missed the 10/12 date and are still not here) | FedAvg over one client averages it with itself. That is not federation | The hybrid demo: one real node + 4-8 simulated MEVA peers (10.6). Second Pi 5 + AI HAT+ 2 when the parts arrive; the demo upgrades with `--real-nodes 2` |
| **Robust filters idle below 3 contributors** | With one real node alone only the structural and norm checks run | The hybrid demo has 5-9 contributors, so direction and trimmed-mean are active; but only one is a real device, and the showcase must say so |
| **Federated-vs-local evidence: answered, and the answer is not flattering** | Federation alone is 0.047 macro balanced accuracy below local-only; with a per-site fine-tune it ties local-only. Package recall collapses in the shared model because the MEVA anomaly rule is camera-relative | Report it as is. Next: add the Pi events as a camera, re-run with a site-independent anomaly rule, second pass through our own YOLO |
| **Pi events not yet in the benchmark** | The 322 labelled events are the only data from ARGUS's own camera; transfer to it is the claim the showcase needs | Pi is off: `scp` them (`docs/operations/ACCESS_MAP.md`), add as a tenth camera in `sim/meva_experiment.py` |
| **Class imbalance in real data** | `routine_vehicle` had 0 samples and `routine_animal` 1; balanced accuracy pinned at 0.75 | Ingest street footage with `scripts/ingest_video.py --label-by-class`, then review the labels in the dashboard |

## 10.3 Gaps in the running system

| Gap | Risk | Fix |
|---|---|---|
| **Federated and dashboard routes have no auth**, even with `AUTH_REQUIRED=true` | Anyone on the network can relabel data, seed rows, or trigger training; the tunnel preflight does not notice | Add `require_auth` to the `APIRouter`s in `trial_routes.py` and `dashboard_routes.py`; probe one in `run_tunnel.sh` |
| **Scheduler idle check is never fed** | `notify_motion()` has no caller, so "only while idle" is always true | Call it from the collector or worker when motion is seen |
| **No clip retention** | `media/events/` grows until the SD card fills; a card already corrupted once | A retention sweep by age or total size (Mohammed, in progress) |
| **Hailo unused** | Every Pi number is CPU (NCNN) | Custom compile is future work (the Dataflow Compiler was never installed; the 10/18 hard stop passed). Base detector locked to YOLOv8n on CPU. Prebuilt Hailo model-zoo `yolov8n_v5.x.hef` files are on disk but not wired in |
| **Dashboard does not send its token** | Blocks turning auth on with the Next.js dashboard | Frontend change in `adam9798/argus-dashboard` |
| **Tokens in memory** | Every restart logs everyone out | Acceptable for the demo |
| **Encryption at rest is a placeholder** | `SecurityService.encrypt/decrypt` are TODOs | Out of scope for the showcase; state it honestly |
| **"Package" means backpack, handbag or suitcase** | Real parcels are not detected | Needs a custom-trained detector |

## 10.4 Housekeeping

| Item | Where |
|---|---|
| The GitHub default branch (`master`) is a stale mirror. `main` was brought up to the showcase hardening line on 2026-10-19 | Decide whether to retire `master` |
| `docs/project/PROPOSAL.md` and `docs/project/SHOWCASE_PRD.md` still state 24-30 FPS; the target is now 8-12 FPS | Revise |
| `docs/project/PROPOSAL.md` still lists face requirements FR-4/FR-5/PR-4/PR-7 and objective O2 as complete | Mark "implemented, measured, disabled for performance" and keep the 0.9996 LFW result |
| `QUICKSTART.md`, `OVERVIEW.md`, guide 6, `FEDERATED_ARCHITECTURE.md` and `JETSON_MIGRATION.md` say rounds run only while idle / nothing is moving | True only once `notify_motion()` is wired |
| `tests/test_federated_infra.py::test_an_unknown_label_is_rejected` fails instead of skipping without `media/vtest.avi` | Add the same skip its sibling test has |
| FastAPI `on_event` is deprecated | Move to a lifespan handler when convenient |
| Older docs (`ARCHITECTURE.md`, `BOOTUP.md`, `WEB_APP_ARCHITECTURE.md`, `PROMETHEUS*.md`) date from January and describe the earlier design | Treat this reference as current where they disagree |
| The 2026-10-05 work and the fine-taxonomy branch are committed and pushed; nothing is single-machine any more | Done |

## 10.5 The plan of record (from the 2026-10-04 status report)

| Item | Owner | Dates |
|---|---|---|
| Wire `robust.py` into the Flower servers | Giovanny | **Done 2026-10-05** (uncommitted, see 10.4) |
| Parts for the second node ordered | Georgios | Were due Oct 12; **still not here on Oct 18** |
| Alert delivery (Telegram, Discord) and dashboard label capture | Saifeddine | Oct 6 - 11 |
| Clip retention service; login on two API routes | Mohammed | In progress |
| Second enclosure print and closed-case heat test | Mohammed | Before Oct 13 |
| Federated vs. local-only comparison | Giovanny | **Done 2026-10-18**, `docs/research/fl-vs-base-meva.md` |
| Hailo compile on the VM | Giovanny | **Not done; hard stop passed.** Future work; YOLOv8n on CPU is the showcase detector |
| Build node B | Christian | Oct 13 - 18; **blocked on parts** |
| Simulation matrix: 3 datasets × 3 strategies (FedAvg, FedTrimmedAvg, robust) × 3 attacks × 5 seeds, 10 clients = 135 runs | Giovanny | Oct 19 - Nov 1 |
| Decision: two-node live demo or fallback | Team | Oct 25. The fallback is the MEVA-backed hybrid demo (not CIFAR-10); see `DEMO_RUNBOOK.md` section H4 |
| Feature freeze | | Nov 8 |
| Rehearsal, poster, backup video | Team | Nov 9 - Nov 21 |
| Showcase | | Nov 22 |

Production operation of the federated loop (the schedule, model versions,
rollback) is documented separately in `docs/operations/PRODUCTION_FL.md`.

Datasets for the matrix (decided 2026-10-12, `docs/research/datasets.md`):
MEVA as the main benchmark, one client per security camera; ShanghaiTech
Campus as the anomaly test set; CIFAR-10 with Dirichlet partitioning only
for comparison with published results.

## 10.6 What was done 10/12 - 10/18, and what is still open

**Done**

| What | Where |
|---|---|
| Fine-taxonomy branch merged into the hardening line; four branches pushed to GitHub | `feat/fine-taxonomy`, `feat/meva-converter`, `fix/showcase-hardening`, `fix/showcase-hardening-2` |
| MEVA: 9 cameras converted (was 1) | `data/datasets/meva/events/`; `scripts/meva_to_events.py` |
| Base detector locked: YOLOv8n | `docs/research/detector-comparison.md` |
| Hybrid demo launcher, poisoned-peer toggle, per-round evaluation | `scripts/demo_hybrid.py`, `sim/poison.py`, `sim/metrics.py`, `sim/fl.py` (`--eval-store`, `--poison`, `--node-id`) |
| Federated vs base vs local-only vs FedAvg + fine-tune on MEVA | `sim/meva_experiment.py`, `docs/research/fl-vs-base-meva.md` |
| Demo script and runbook rewritten for the hybrid demo | `docs/project/DEMO_SCRIPT.md`, `docs/operations/DEMO_RUNBOOK.md` section H |

**Still open**

| Item | Blocked on | Next step |
|---|---|---|
| Node B | Parts not delivered | `python scripts/demo_hybrid.py --peers 6 --rounds 5 --real-nodes 2` once built |
| Copy the Pi events; freeze the base holdout | Pi was off | `scp` per `ACCESS_MAP.md`; add as a camera in the experiment |
| Detector speed on the Pi | Pi was off | `python scripts/compare_detectors.py --skip-accuracy --models yolov8n` on the Pi |
| systemd unit on the Pi; verify a scheduled round | Pi was off | `sudo systemctl enable --now argus-node`, then `curl localhost:8000/api/federated/status` |
| Run the hybrid demo against the real Pi | Pi was off | `docs/operations/DEMO_RUNBOOK.md` H2 |
| Threat test set and threshold re-tune | Not started | `docs/research/threat-eval.md` |
| Hailo custom compile | Dataflow Compiler never installed | Future work |
| Second pass through ARGUS's own YOLO on MEVA video | Needs video downloads | `scripts/ingest_video.py` per camera |
| ShanghaiTech anomaly test; the 135-run matrix | Not started | Due 11/01 |
| `PROPOSAL.md` still states 24-30 FPS and lists face requirements as complete | Uncommitted edit on `perf/stream-frame-rate` is not ours | Team to revise |
