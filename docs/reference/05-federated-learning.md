# 5 · Training and federated learning

This is the part of ARGUS that learns. YOLO and the threat model are
frozen. What learns is a small head that decides whether a detection is
routine for the site it was seen at, trained on that site's labelled
events, and improved by averaging with other sites without sharing any
footage.

---

## 5.1 The data path, end to end

```
process_frame detections
        │
EventCollector.observe          src/federated/collector.py   IoU + class tracking, one row per track
        │ track closes
features.extract(track)         src/federated/features.py    19 numbers
        │
SampleStore.append              src/federated/store.py       data/training/samples.jsonl  (label = None)
        │ a person confirms or corrects (dashboard, API, ingest_video.py)
SampleStore.set_label
        │
FederatedHead.fit               src/federated/head.py        local training on a copy
        │
node_client  ◄──── Flower round ────►  aggregator (FedAvg)   src/federated/node_client.py
        │ global.npz
model_state.accept_candidate    src/federated/model_state.py the validation gate
        │ accepted
data/fl_head.npz  →  live head, version fl-rN
```

---

## 5.2 `features.py`: what the head sees

The head does not see pixels. It sees a **detection in context**, 19
numbers, all normalised into roughly [-1, 1] at the source so no scaler has
to be fitted or shipped:

| Index | Feature | How it is computed |
|---|---|---|
| 0-3 | class one-hot | person, vehicle, animal, package |
| 4-5 | width, height | box size as a fraction of the frame |
| 6 | sqrt(area) | |
| 7 | aspect | `min(w/h, 6) / 6` |
| 8-9 | centre x, centre y | fraction of the frame |
| 10-11 | velocity x, y | `tanh(10 × displacement)` between the last two updates |
| 12 | speed | `tanh(10 × |displacement|)` |
| 13 | dwell | seconds present / 30, capped at 1 |
| 14-15 | hour of day | sin and cos, so 23:59 sits next to 00:00 |
| 16-17 | day of week | sin and cos |
| 18 | detector confidence | |

Five output labels: `routine_person` (0), `routine_vehicle` (1),
`routine_animal` (2), `routine_package` (3), `anomaly` (4).

`class_index(detection_type, coco_id)` maps a detection onto the four
classes. Packages are checked by COCO id first (24, 26, 28). Motion and
unknown return `None` and are skipped rather than fed in as zeros.

**`features.py` and `sim/workload.py` must stay in lockstep.** The
simulator generates vectors in this exact layout.

---

## 5.3 `collector.py`: one event per track, not per frame

`EventCollector.observe(detections, frame_shape, frame=...)` is called by
`process_frame` on every frame.

* **Association.** Each detection joins the open track of the same class
  with the highest IoU above 0.3. If none overlaps, it falls back to
  centre distance within 2.5 box-widths, because a fast car can move
  further than its own width between detections and IoU alone would break
  it into one-frame tracks.
* **Closing.** A track unseen for 2 s closes. Tracks with fewer than 3
  frames are dropped as detector flicker (`dropped_short`).
* **One row per track.** A person standing in frame for a minute is one
  training row, not 1,800. Otherwise the head learns that whatever
  loiters longest matters most.
* **Stills.** The still saved for an event is the frame where the subject
  was **largest** (updated only when the box grows 15%), not the last
  frame, by which time the subject is half out of shot. A crop of just the
  subject is saved alongside, so an image model can be trained later
  without re-recording. Both go to `data/snapshots/`.
* **Motion events.** Sustained motion over 0.2% of the frame for at least
  1 s, with no classified object, becomes a `motion` event. It is how a
  person YOLO missed still leaves a record. Long motion is cut at 60 s.
* Every emitted row is also logged in the `events` table
  (`event_type="detection"`) with a human-readable feature summary.

---

## 5.4 `store.py`: the sample store

`SampleStore(path)` is an append-only JSONL file,
`data/training/samples.jsonl`. One line per sample:

```json
{"id": "18f2c3a9b1-00042", "n": 43, "t": 1727800000.0,
 "x": [0.0, 1.0, 0.0, 0.0, 0.31, ...], "y": null,
 "meta": {"snapshot": "...jpg", "crop": "...jpg", "track_id": 7, "frames": 41,
          "duration_s": 3.2, "source": "camera", "synthetic": false}}
```

`n` is the stable integer id the dashboard addresses. `y` is `null` until a
person labels it. JSONL survives a restart, can be inspected with `tail`,
and needs no migration when a field is added.

**Why labels come from a person:** the simulator derives labels from the
same rule that generated its data, which is circular. On a real node the
label has to come from outside the model.

`bootstrap(site, n)` seeds rows from the simulator's generator, tagged
`synthetic: true`, so a demo can train before enough real events exist.
`stats()` reports real and synthetic counts separately. `holdout_split()`
is the one deterministic 80/20 split shared by the client and the gate, so
a candidate is always scored on rows it did not train on.

---

## 5.5 `head.py`: the model

`FederatedHead` is a pure-numpy MLP:

```
19 → 384 → 384 → 128 → 5      ReLU, softmax, cross-entropy, Adam (lr 3e-3), batch 64
```

About 200K parameters, deliberately. Constraint CON-3 in the proposal caps
the federated model there because training runs on the Pi 5 CPU at night.
No torch import, so its memory use is honest, and Flower's `NumPyClient`
wants lists of arrays anyway.

| Method | |
|---|---|
| `fit(X, y, epochs, batch=64, seed)` | Mini-batch Adam; returns the last batch loss |
| `evaluate(X, y)` | (loss, accuracy) |
| `balanced_accuracy(X, y)` | Mean per-class recall. **Report this, never plain accuracy**: 254 of the first 322 real events were person, so always guessing person scores about 79% |
| `predict`, `infer_one` | Class probabilities / one label |
| `get_weights`, `set_weights`, `save`, `load` | Flower and `.npz` transport |

`src/federated/head.py` mirrors `sim/head.py`. Keep them identical, or the
simulator stops predicting what the node does.

---

## 5.6 `model_state.py`: the validation gate

Nothing reaches the live head without passing
`accept_candidate(app, candidate_path, source, tolerance=0.01)`:

1. Score the live head and the candidate on this node's held-out 20%.
2. Accept if the candidate's balanced accuracy is no more than
   `tolerance` below the live head's. The very first model is accepted
   outright.
3. On acceptance, swap the weights in one assignment, save
   `data/fl_head.npz`, bump the version (`local-rN` or `fl-rN`).
4. Record every decision, accepted or not, in `data/fl_model_state.json`
   (last 50 kept).

`load_model_state(app)` restores the accepted head at startup, so an
update survives a restart.

**Known bug:** the gate compares balanced accuracy only. In the 2026-09-20
hardware run, round 2's averaged model had worse loss (0.1941 → 0.2529)
and was accepted, because 0.75 against 0.75 is not a regression. It needs
a loss tie-break.

---

## 5.7 Local training: `POST /api/federated/head/train`

Implemented in `src/api/trial_routes.py`. Trains a **copy** of the live
head on the labelled training split, off the event loop, then sends the
copy through the gate with `source="local-train"`. Refuses with 409 below
`min_samples` (200).

---

## 5.8 A federated round: `rounds.py` and `node_client.py`

`run_round_blocking(app, server, epochs=6)` is the one definition of a
round. The cron and the manual trigger (`POST /api/federated/round`) both
follow it so the 02:00 path cannot drift from the demo path.

1. Require 200 labelled samples.
2. Save the live weights to `data/fl_rounds/<timestamp>/init.npz`.
3. Launch `python -m src.federated.node_client --store ... --init ... --out ...
   --server ... --epochs ... --node-name ... [--central-api ...]` as its own
   process, with BLAS pinned to one thread (`ARGUS_TRAIN_THREADS`).
4. The client registers with the aggregator's REST API if
   `--central-api` is given, then joins the Flower round:
   * `fit()` loads the server's weights, trains on the node's labelled
     training split, returns weights plus real metrics (loss, accuracy,
     balanced accuracy, samples, seconds) and its `node_id`.
   * `evaluate()` receives the **aggregated** weights, scores them, and
     writes them to `global.npz`. After the last round that file holds the
     final global model.
5. Back in the API process, `global.npz` goes through the gate with
   `source="federated"`.

Everything lands in `data/fl_rounds/<timestamp>/` (`init.npz`,
`global.npz`, `events.jsonl`), so any round can be inspected afterwards.

Only weights travel. No frames, stills or feature vectors leave the node.

---

## 5.9 `scheduler.py`: the cron

`FLScheduler` runs only when `FL_ENABLED=true`.

* Due every `FL_ROUND_INTERVAL_DAYS` (14) at `FL_ROUND_HOUR` (02:00).
* The last-round time is persisted to `data/fl_state.json`. Without that,
  a node rebooted weekly would restart a 14-day countdown forever and
  never train.
* A round overdue at startup runs after a 5-minute grace period, not a
  full interval later.
* It wakes at least hourly rather than sleeping for 14 days, which
  survives suspend, NTP clock steps on an RTC-less board, and lets `stop()`
  return promptly.
* A failed round does **not** advance the schedule, so it retries at the
  next wake.
* `is_idle()` defers a round if motion was seen in the last 10 minutes.
  **But nothing calls `notify_motion()`, so this check always passes
  today.**

---

## 5.10 `robust.py`: Byzantine-robust aggregation (not wired in)

Plain FedAvg lets one broken or hostile node move the global model for
everyone. `RobustAggregator.aggregate(global_weights, updates)` screens
**deltas** (client minus global), cheapest filter first:

| Filter | Rejects | Defends against |
|---|---|---|
| 1 Structural | wrong shapes, NaN/Inf, zero samples | a broken or mismatched node |
| 2 Norm | update norm > 3× the median | scaled "model replacement" attacks, a diverged optimiser |
| 3 Direction | cosine to the consensus direction < 0 | sign-flip, label-flip, targeted poisoning |
| 4 Trimmed mean | the top and bottom 10% per coordinate | residual outliers |

Filters 3 and 4 need a majority to be honest, so they are skipped below 3
contributors and the report says so. `validate_candidate()` is the matching
promotion check.

21 tests in `tests/test_robust_aggregation.py` cover it. **No server
imports it yet**: the central aggregator, the simulator server and the
legacy server all run plain FedAvg. Wiring it into
`central_server/fl_aggregator.py`'s `ArgusStrategy.aggregate_fit` is the
planned next step and the core of the research claim.

---

## 5.11 The aggregator: `central_server/fl_aggregator.py`

One process, two jobs:

* **Flower server** on `ARGUS_FL_ADDRESS` (default `0.0.0.0:8080`).
  `ArgusStrategy` is FedAvg plus a hook that records per-client metrics
  and saves a checkpoint after every round. Metrics are weighted by sample
  count (without that, Flower drops them and every dashboard reads zero).
  On start it resumes from the latest checkpoint.
* **Flask REST API** on 8090:

| Route | |
|---|---|
| `GET /api/status` | server state |
| `GET /api/nodes`, `POST /api/nodes/register`, `POST /api/nodes/<id>/heartbeat` | node registry |
| `GET /api/training/history`, `/api/training/round/<n>`, `/api/training/summary` | per-round results |
| `GET /api/model/latest`, `/api/model/checkpoints` | global model versions |
| `GET/POST /api/config` | runtime config |
| `GET /metrics`, `/api/prometheus/targets` | Prometheus |

Set `ARGUS_AUTH_TOKEN` to require a bearer token on the protected routes.
Config comes from `central_config.json` overridden by `ARGUS_FL_ADDRESS`,
`ARGUS_MIN_CLIENTS`, `ARGUS_ROUNDS`. `docker-compose.yml` runs it with
Prometheus (9090), Grafana (3000) and a Cloudflare tunnel.

---

## 5.12 The simulator: `sim/`

| File | Role |
|---|---|
| `workload.py` | Synthetic per-site events in the 19-feature layout. Each site has its own class mix, active hours, never-routine classes and loitering limit, so the data is non-IID in both label mix and the rule itself |
| `head.py` | Copy of the node's head |
| `fl.py` | `serve` (FedAvg Flower server) and `client --site` (trains on generated data). Separate OS processes, one BLAS thread each |
| `pipeline.py` | The per-frame loop: `cpu` stages burn a real core, `accel` stages sleep (the Hailo works over PCIe while the CPU is free), `measured` stages time the real head |
| `profiles.py` | Stage timings, each with its source |
| `run_node.py` | One node at a target FPS with a live Flower round against it; reports the FPS hit, Prometheus on 9108 |

The three sites:

| Site | Class mix (person/vehicle/animal/package) | Active hours | Rule |
|---|---|---|---|
| `driveway` | 40 / 42 / 10 / 8 | 06-22 | Night presence is the signal |
| `backdoor` | 46 / 4 / 14 / 36 | 08-19 | A vehicle here is always wrong |
| `yard` | 22 / 3 / 56 / 19 | 07-20 | Tolerant of loitering, not of people at night |

Profiles:

| Profile | CPU ms | Accelerator ms | Serial ceiling | Pipelined |
|---|---|---|---|---|
| `pi5-hailo` (target build, default) | 27 | 40 | 14.9 FPS | 25.0 FPS |
| `pi5-threat-on-cpu` (risk) | 63 | 28 | 11.0 FPS | 15.9 FPS |
| `pi5-cpu-only` (fallback) | 405 | 0 | 2.5 FPS | 2.5 FPS |
| `x86-dev` (measured) | 140 | 0 | 7.1 FPS | 7.1 FPS |

What it answered: the sponsor spec rejected every 8 GB board on the claim
that a node "stops detecting while it trains". That claim came from Jetson
unified memory. With detection on the Hailo and a 200K-parameter head on
the CPU, the simulator measured a round costing about 0.1% of frame rate.

What it cannot answer: whether the head is accurate on real data, because
the team wrote both the generator and the model.

---

## 5.13 Results so far

| Run | Setup | Result |
|---|---|---|
| Hardware, 2026-09-20 | One Pi, laptop aggregator, 2 rounds, 322 hand-labelled events | Balanced accuracy 0.75. `routine_vehicle` had 0 samples and `routine_animal` 1, so only three of five classes were represented |
| Simulation, 2026-10-02 | 3 synthetic sites, 3 rounds, `sim/fl.py` | Balanced accuracy 0.820 → 0.829 → 0.838 |

Neither is yet evidence that federated beats local-only training. That
needs the three-arm comparison (untrained / local-only / federated, on one
frozen holdout, ideally leave-one-site-out) in the 2026-09-27 plan.
