# 10 · Status, known gaps, and what to do next

As of 2026-10-02. Showcase 2026-11-22; feature freeze 2026-11-08.

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
| Test suite | 303 / 303 on 2026-10-02 |

## 10.2 Gaps that block the research claim

| Gap | Why it matters | Fix |
|---|---|---|
| **One physical node** | FedAvg over one client averages it with itself. That is not federation | Second Pi 5 + AI HAT+ 2. Meanwhile, simulation with 3 or more clients is the primary evidence |
| **`robust.py` is not wired into any server** | The project's central contribution is written and tested but disconnected | Call `RobustAggregator` from `ArgusStrategy.aggregate_fit` in `central_server/fl_aggregator.py` |
| **No federated-vs-local evidence** | The sponsor's actual question is whether federated beats a node training alone | Three arms (untrained, local-only, federated) on one frozen holdout, leave-one-site-out |
| **Class imbalance in real data** | `routine_vehicle` had 0 samples and `routine_animal` 1; balanced accuracy pinned at 0.75 | Capture or ingest vehicle and animal footage (`scripts/ingest_video.py`) |
| **The gate ignores loss** | A worse model at equal balanced accuracy is accepted (0.1941 → 0.2529 on 2026-09-20) | Add a loss tie-break in `model_state.accept_candidate` |

## 10.3 Gaps in the running system

| Gap | Risk | Fix |
|---|---|---|
| **Federated and dashboard routes have no auth**, even with `AUTH_REQUIRED=true` | Anyone on the network can relabel data, seed rows, or trigger training; the tunnel preflight does not notice | Add `require_auth` to the `APIRouter`s in `trial_routes.py` and `dashboard_routes.py`; probe one in `run_tunnel.sh` |
| **Scheduler idle check is never fed** | `notify_motion()` has no caller, so "only while idle" is always true | Call it from the collector or worker when motion is seen |
| **No clip retention** | `media/events/` grows until the SD card fills; a card already corrupted once | A retention sweep by age or total size |
| **Hailo unused** | Every Pi number is CPU (NCNN) | Compile on the x86_64 Linux VM with `scripts/export_hailo.py --hw-arch hailo10h`; hard stop 2026-10-18, then present the CPU numbers |
| **Dashboard does not send its token** | Blocks turning auth on with the Next.js dashboard | Frontend change in `adam9798/argus-dashboard` |
| **Tokens in memory** | Every restart logs everyone out | Acceptable for the demo |
| **Encryption at rest is a placeholder** | `SecurityService.encrypt/decrypt` are TODOs | Out of scope for the showcase; state it honestly |
| **"Package" means backpack, handbag or suitcase** | Real parcels are not detected | Needs a custom-trained detector |

## 10.4 Housekeeping

| Item | Where |
|---|---|
| The GitHub default branch (`master`) is a stale mirror and `main` is behind too. The work is on `feat/threat-detection`, not merged | GitHub settings, or merge when ready |
| `docs/PROPOSAL.md` and `docs/SHOWCASE_PRD.md` still state 24-30 FPS; the target is now 8-12 FPS | Revise |
| `docs/PROPOSAL.md` still lists face requirements FR-4/FR-5/PR-4/PR-7 and objective O2 as complete | Mark "implemented, measured, disabled for performance" and keep the 0.9996 LFW result |
| `QUICKSTART.md`, `OVERVIEW.md`, guide 6, `FEDERATED_ARCHITECTURE.md` and `JETSON_MIGRATION.md` say rounds run only while idle / nothing is moving | True only once `notify_motion()` is wired |
| `tests/test_federated_infra.py::test_an_unknown_label_is_rejected` fails instead of skipping without `media/vtest.avi` | Add the same skip its sibling test has |
| FastAPI `on_event` is deprecated | Move to a lifespan handler when convenient |
| Older docs (`ARCHITECTURE.md`, `BOOTUP.md`, `WEB_APP_ARCHITECTURE.md`, `PROMETHEUS*.md`) date from January and describe the earlier design | Treat this reference as current where they disagree |

## 10.5 The plan of record (from the 2026-09-27 status report)

| Item | Dates |
|---|---|
| Wire `robust.py` into the Flower server | Sep 28 - Oct 4 |
| Order the second Pi 5 + AI HAT+ 2 | by Sep 29 |
| Disk retention; push the dashboard branch | Oct 1 - Oct 4 |
| Three-arm ablation harness | Oct 5 - Oct 18 |
| Hailo compile on the VM | Oct 5 - Oct 18 |
| Simulation matrix: 3 datasets × 3 strategies (FedAvg, FedTrimmedAvg, robust) × 3 attacks × 5 seeds, 10 clients = 135 runs | Oct 19 - Nov 1 |
| Feature freeze | Nov 8 |
| Rehearsal, poster, backup video | Nov 9 - Nov 21 |
| Showcase | Nov 22 |

Datasets for the matrix, in order: CIFAR-10 with Dirichlet partitioning
plus FEMNIST (comparability with published results), CASAS smart-home
data (real per-home heterogeneity), then VIRAT or UCF-Crime run through
ARGUS's own YOLO and `features.py` (real multi-site data in ARGUS's own
feature space, filling the empty classes).
