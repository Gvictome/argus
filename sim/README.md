# ARGUS node simulator

A single ARGUS node running its detection loop at a target frame rate while a
**real Flower federated round** runs against it, so you can watch what the
round does to the frame rate.

```bash
python sim/run_node.py                                    # one node, 12 FPS
python sim/run_node.py --clients 3 --rounds 6             # real FedAvg
python sim/run_node.py --profile pi5-threat-on-cpu        # the risk case
python sim/run_node.py --pipelined                        # overlap the Hailo
python sim/run_node.py --cores 4                          # match Pi core count
```

Metrics land on `http://127.0.0.1:9108/metrics` (`--port 0` to disable).

---

## The question it answers

`ARGUS_Materials_List_B_Sponsor_Spec.md` rejected every 8 GB board against
"REQ 3", on the claim that under 16 GB *"a live security node stops detecting
while it trains. That is a functional failure, not a slowdown."*

That claim was written about **Jetson unified memory**, where CPU and GPU share
one pool and a Flower client holding GPU memory genuinely starves inference.
The architecture actually being built is different: detection is offloaded to
the Hailo-10H, which has its own 8 GB, and the trainer is a ~200K parameter
head on the CPU. This simulator measures whether REQ 3 transfers. It does not.

---

## Why it is not just a spreadsheet

The thing that makes the numbers mean anything:

| Stage type | What the sim does | Why |
|---|---|---|
| `cpu` | calibrated numpy matmul loop | Burns a real core. Contends with the trainer through the OS scheduler, exactly as the real pipeline would. |
| `accel` | `time.sleep()` | The Hailo works over PCIe; the CPU is free meanwhile. Sleeping is the honest model. |
| `measured` | real forward pass, timed | The federated head is a real model, not a budget. |

If CPU stages were also sleeps, a concurrent training round would show zero
impact — which is the entire question. The trainer runs as a **separate OS
process** talking gRPC, not a thread, so contention is real.

> **One BLAS thread per process is mandatory** and is forced at the top of
> `run_node.py` and `fl.py`. numpy's BLAS is multithreaded by default, so a
> single "CPU stage" would otherwise saturate every core; with the trainer
> doing the same the two oversubscribe the machine and both collapse. The
> first run of this sim showed 393 ms for a 12 ms sleep before this was fixed.

---

## Profiles

Timings come from Rev 2.2 Table 11 and `docs/HAILO_PIPELINE.md`, **with ArcFace
removed** per the 2026-09-14 scope change. Each stage carries its source.

| Profile | CPU | Accel | Serial ceiling | Pipelined |
|---|---:|---:|---:|---:|
| `pi5-hailo` (target build) | 27 ms | 40 ms | **14.9 FPS** | 25.0 FPS |
| `pi5-threat-on-cpu` (risk) | 63 ms | 28 ms | **11.0 FPS** | 15.9 FPS |
| `pi5-cpu-only` (fallback) | 405 ms | 0 ms | 2.5 FPS | 2.5 FPS |
| `x86-dev` (measured) | 140 ms | 0 ms | 7.1 FPS | 7.1 FPS |

`pi5-threat-on-cpu` is the case `HAILO_PIPELINE.md` §3 warns about: an
uncompiled threat model is *"a second CPU YOLO pass at ~36 ms/frame"*. Table 11
has no threat line at all, so this is the gap between the proposal and the code.

---

## The federated head

Replaces the CIFAR-10 stand-in. It does not classify pixels — YOLO already
emits person/vehicle/animal/package, and reproducing that adds nothing. It
classifies a **detection in context**: what was found, how big, where, how long
it lingered, what time it was. A delivery van at 14:00 is routine; the same van
at 03:00 is not. That judgement is site-specific, which is what makes it worth
federating.

- 19 features → 384 → 384 → 128 → 5 classes, **205,445 params** (CON-3 bound ~200K)
- Labels are the five from `flower_training_report_2026-04-24.md:18`:
  routine person / vehicle / animal / package, plus anomaly
- Pure numpy, so `NumPyClient` is a direct fit and the RSS figure stays honest

### Genuinely non-IID

`flower_training_report_2026-04-24.md:36` partitioned CIFAR-10 with
`IidPartitioner` — *"each node gets a random, balanced sample"*. That is the
easiest possible split; it validates plumbing, not learning.

Here each of the three sites has both a different **label** distribution and a
different **conditional** distribution — the anomaly rule itself differs:

| Site | Mix | Rule |
|---|---|---|
| `driveway` | vehicles + people all day | night presence is the signal |
| `backdoor` | packages + staff | a vehicle here is *always* wrong |
| `yard` | mostly animals | tolerant of loitering, not of people at night |

Watch the per-site balanced-accuracy spread in the report. That spread is the
non-IID effect, and it is the thing a 5-node deployment has to survive.

---

## Reading the output

Three phases, same pipeline throughout:

```
  phase         frames     fps   p50 ms   p95 ms   p99 ms
  baseline          96   14.44     69.2     70.1     76.7
  fl               510   14.19     68.7     76.4    112.9  (-1.7%)
  recovery          72   14.44     68.9     72.1     77.6  (+0.0%)
```

`fl` is the window where the Flower round was running. The delta against
baseline is the answer to REQ 3.

The stage table compares measured against budget, so you can see which budget
lines the simulator actually reproduces and which drift.

---

## What this does not model

Stated plainly because the numbers will get quoted:

- **CPU stage times are budgets from Rev 2.2 Table 11, not Pi measurements.**
  No Pi 5 + Hailo-10H benchmark exists in this repo yet. `DETECTION_ACCURACY.md`
  §6 is explicit that its x86 figures are *"a ceiling, not a prediction"*.
- **The head is numpy.** Keeping TensorFlow in the federated loop adds roughly
  1 GB of framework RSS this does not model.
- **These are this machine's cores, not the Pi's.** `--cores 4` matches the
  count, not the per-core performance.
- **Memory pressure is not simulated.** The sim shows CPU contention, which is
  what actually binds here; it is not evidence about 8 GB vs 16 GB on its own.

To replace the budgets with real numbers, run `scripts/benchmark_classes.py`
on the Pi and edit `sim/profiles.py`.

---

## Files

| File | Role |
|---|---|
| `profiles.py` | Hardware timing profiles, every line sourced |
| `workload.py` | Non-IID per-site event generator |
| `head.py` | The ~200K param federated head (numpy MLP) |
| `pipeline.py` | Frame loop; real CPU burn vs accelerator sleep |
| `fl.py` | Flower server + client, run as separate processes |
| `run_node.py` | Orchestrator, Prometheus surface, report |
| `_run/fl_events.jsonl` | Raw per-round events from the last run |
