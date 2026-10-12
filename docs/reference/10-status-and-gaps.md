# 10 · Status, known gaps, and what to do next

As of 2026-10-05. Showcase 2026-11-22; feature freeze 2026-11-08.

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
| Test suite | 328 / 328 on 2026-10-05 |

## 10.2 Gaps that block the research claim

| Gap | Why it matters | Fix |
|---|---|---|
| **One physical node** | FedAvg over one client averages it with itself. That is not federation | Second Pi 5 + AI HAT+ 2. Meanwhile, simulation with 3 or more clients is the primary evidence |
| **Robust filters idle below 3 nodes** | Wired in, but with one Pi only the structural and norm checks run | Node B (parts in hand by Oct 12, build Oct 13-18); the simulation matrix runs 10 clients |
| **No federated-vs-local evidence** | The sponsor's actual question is whether federated beats a node training alone. The fine-taxonomy sims found local-only ahead of plain FedAvg | Three arms (untrained, local-only, federated) on one frozen holdout, leave-one-site-out; add per-site fine-tuning after FedAvg as a fourth arm |
| **Class imbalance in real data** | `routine_vehicle` had 0 samples and `routine_animal` 1; balanced accuracy pinned at 0.75 | Ingest street footage with `scripts/ingest_video.py --label-by-class`, then review the labels in the dashboard |

## 10.3 Gaps in the running system

| Gap | Risk | Fix |
|---|---|---|
| **Federated and dashboard routes have no auth**, even with `AUTH_REQUIRED=true` | Anyone on the network can relabel data, seed rows, or trigger training; the tunnel preflight does not notice | Add `require_auth` to the `APIRouter`s in `trial_routes.py` and `dashboard_routes.py`; probe one in `run_tunnel.sh` |
| **Scheduler idle check is never fed** | `notify_motion()` has no caller, so "only while idle" is always true | Call it from the collector or worker when motion is seen |
| **No clip retention** | `media/events/` grows until the SD card fills; a card already corrupted once | A retention sweep by age or total size (Mohammed, in progress) |
| **Hailo unused** | Every Pi number is CPU (NCNN) | Compile on the x86_64 Linux VM with `scripts/export_hailo.py --hw-arch hailo10h`; hard stop 2026-10-18, then present the CPU numbers |
| **Dashboard does not send its token** | Blocks turning auth on with the Next.js dashboard | Frontend change in `adam9798/argus-dashboard` |
| **Tokens in memory** | Every restart logs everyone out | Acceptable for the demo |
| **Encryption at rest is a placeholder** | `SecurityService.encrypt/decrypt` are TODOs | Out of scope for the showcase; state it honestly |
| **"Package" means backpack, handbag or suitcase** | Real parcels are not detected | Needs a custom-trained detector |

## 10.4 Housekeeping

| Item | Where |
|---|---|
| The GitHub default branch (`master`) is a stale mirror and `main` is behind too. The work is on `feat/threat-detection`, not merged | GitHub settings, or merge when ready |
| `docs/project/PROPOSAL.md` and `docs/project/SHOWCASE_PRD.md` still state 24-30 FPS; the target is now 8-12 FPS | Revise |
| `docs/project/PROPOSAL.md` still lists face requirements FR-4/FR-5/PR-4/PR-7 and objective O2 as complete | Mark "implemented, measured, disabled for performance" and keep the 0.9996 LFW result |
| `QUICKSTART.md`, `OVERVIEW.md`, guide 6, `FEDERATED_ARCHITECTURE.md` and `JETSON_MIGRATION.md` say rounds run only while idle / nothing is moving | True only once `notify_motion()` is wired |
| `tests/test_federated_infra.py::test_an_unknown_label_is_rejected` fails instead of skipping without `media/vtest.avi` | Add the same skip its sibling test has |
| FastAPI `on_event` is deprecated | Move to a lifespan handler when convenient |
| Older docs (`ARCHITECTURE.md`, `BOOTUP.md`, `WEB_APP_ARCHITECTURE.md`, `PROMETHEUS*.md`) date from January and describe the earlier design | Treat this reference as current where they disagree |
| **Uncommitted on this laptop (2026-10-05):** robust strategy wiring (`src/federated/robust_strategy.py`, `central_server/*`, `sim/fl.py`, `src/federated/server.py`, `run_central_server.ps1`), the gate loss tie-break, version archive and rollback (`model_state.py`, `trial_routes.py`), the cron retry fix (`rounds.py`, `scheduler.py`), `deploy/`, `docs/operations/PRODUCTION_FL.md`, `ingest_video.py --label-by-class`, `scripts/measure_training_cost.py`, runbook and doc edits, and their tests | Commit and push, gated on pytest's exit code |
| The fine-taxonomy branch (`feat/fine-taxonomy`) exists only on the other development machine | Push it so it is not a single-machine artefact |

## 10.5 The plan of record (from the 2026-10-04 status report)

| Item | Owner | Dates |
|---|---|---|
| Wire `robust.py` into the Flower servers | Giovanny | **Done 2026-10-05** (uncommitted, see 10.4) |
| Parts for the second node ordered | Georgios | In hand by Oct 12 |
| Alert delivery (Telegram, Discord) and dashboard label capture | Saifeddine | Oct 6 - 11 |
| Clip retention service; login on two API routes | Mohammed | In progress |
| Second enclosure print and closed-case heat test | Mohammed | Before Oct 13 |
| Federated vs. local-only comparison (three arms) | Giovanny | Oct 5 - 18 |
| Hailo compile on the VM | Giovanny | Hard stop Oct 18 |
| Build node B | Christian | Oct 13 - 18 |
| Simulation matrix: 3 datasets × 3 strategies (FedAvg, FedTrimmedAvg, robust) × 3 attacks × 5 seeds, 10 clients = 135 runs | Giovanny | Oct 19 - Nov 1 |
| Decision: two-node live demo or fallback | Team | Oct 25 |
| Feature freeze | | Nov 8 |
| Rehearsal, poster, backup video | Team | Nov 9 - Nov 21 |
| Showcase | | Nov 22 |

Production operation of the federated loop (the schedule, model versions,
rollback) is documented separately in `docs/operations/PRODUCTION_FL.md`.

Datasets for the matrix (decided 2026-10-12, `docs/research/datasets.md`):
MEVA as the main benchmark, one client per security camera; ShanghaiTech
Campus as the anomaly test set; CIFAR-10 with Dirichlet partitioning only
for comparison with published results.
