# Federated head vs base, local-only and FedAvg, on MEVA (2026-10-18)

The sponsor's question: does federating the head beat a node training alone?
This is the first answer on real security-camera data. **Short version: on
MEVA, federation alone does not beat local training. Federation plus a
per-site fine-tune ties local-only, and a camera that has seen no local data
does about as well as the no-head baseline.** The numbers are below; read the
caveats before quoting any of them.

## What was run

`sim/meva_experiment.py`, 10 seeds, 9 MEVA cameras (G299, G326, G328, G329,
G330, G331, G336, G339, G340) as 9 clients. Results file:
`docs/research/data/meva_fl_experiment.json`. Reproduce:

```
python sim/meva_experiment.py --seeds 10 --cold-seeds 5 --workers 10 --out data/benchmarks/meva_fl_experiment.json
python sim/meva_experiment.py --markdown data/benchmarks/meva_fl_experiment.json
```

About 8 minutes on 12 cores. The experiment is in-process, with the same
`RobustAggregator` and the same numpy head the Flower server and clients use,
so many seeds are affordable. The gRPC path was exercised separately by the
hybrid demo; see `docs/operations/DEMO_RUNBOOK.md`.

| Setting | Value |
|---|---|
| Input | The 19 features per tracked event from MEVA ground-truth tracks (`scripts/meva_to_events.py`); label = routine class or anomaly by the converter's rule |
| Holdout | Whole clips, 20% of each camera's clips, **reseeded per seed** (so the interval covers training noise and holdout choice). Every arm sees the same holdout within a seed |
| Training cap | At most 4,000 training events per camera (random subsample), so the two cameras with 12-15k events do not drown the rest |
| Rounds | 8, with 3 local epochs each. Scored after round 3, 5 and 8 |
| Local-only | Same number of epochs as the federated arm at each checkpoint, so every arm has an equal training budget |
| Fine-tune | 3 extra local epochs per camera from the federated model |
| Robust | Production `RobustAggregator` defaults (norm 3x median, cosine 0, trim 10%); all 9 clients honest |
| Cold-start | The robust model trained **without** camera c, scored on c. 5 seeds only (9x the work) |
| Detector-only | No head at all: label every detection `routine_<its class>`. This is the node's behaviour with no head installed, and it can never say anomaly |
| Untrained | The shared random initial head |
| Intervals | Mean +/- 95% t-interval over seeds |

**Metrics.** Balanced accuracy and per-class recall only; plain accuracy is
never quoted because person dominates (43-99% of each camera's events).
*Macro* = mean over cameras of each camera's balanced accuracy (each camera
counts equally; this is the "does it work at my site" number). *Pooled* =
every holdout event pooled, classes weighted equally.

## Results

| Arm | Seeds | Macro bal. acc | Pooled bal. acc | person | vehicle | package | anomaly |
|---|---|---|---|---|---|---|---|
| untrained | 10 | 0.185 +/- 0.087 | 0.207 +/- 0.051 | 0.050 +/- 0.074 | 0.170 +/- 0.219 | 0.292 +/- 0.337 | 0.317 +/- 0.264 |
| detector-only | 10 | 0.612 +/- 0.017 | 0.750 +/- 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| local-only, 3 rounds of epochs | 10 | 0.719 +/- 0.014 | 0.816 +/- 0.006 | 0.998 | 0.990 | 0.993 | 0.282 +/- 0.021 |
| FedAvg, 3 rounds | 10 | 0.660 +/- 0.009 | 0.559 +/- 0.007 | 1.000 | 1.000 | 0.000 | 0.237 +/- 0.026 |
| Robust, 3 rounds | 10 | 0.659 +/- 0.009 | 0.559 +/- 0.007 | 1.000 | 1.000 | 0.000 | 0.235 +/- 0.027 |
| local-only, 5 | 10 | 0.720 +/- 0.015 | 0.813 +/- 0.008 | 0.996 | 0.990 | 0.980 | 0.286 +/- 0.027 |
| FedAvg, 5 rounds | 10 | 0.661 +/- 0.008 | 0.565 +/- 0.010 | 1.000 | 0.998 | 0.022 +/- 0.041 | 0.242 +/- 0.027 |
| Robust, 5 rounds | 10 | 0.660 +/- 0.009 | 0.560 +/- 0.007 | 1.000 | 0.999 | 0.000 | 0.239 +/- 0.026 |
| local-only, 8 | 10 | 0.723 +/- 0.008 | 0.816 +/- 0.007 | 0.995 | 0.991 | 0.981 +/- 0.015 | 0.296 +/- 0.025 |
| FedAvg, 8 rounds | 10 | 0.658 +/- 0.012 | 0.611 +/- 0.060 | 0.999 | 0.998 | 0.210 +/- 0.245 | 0.235 +/- 0.025 |
| Robust, 8 rounds | 10 | 0.676 +/- 0.017 | 0.677 +/- 0.077 | 1.000 | 0.999 | 0.490 +/- 0.315 | 0.220 +/- 0.028 |
| **Robust 8 + per-site fine-tune** | 10 | **0.727 +/- 0.011** | 0.814 +/- 0.008 | 0.999 | 0.996 | 0.987 +/- 0.013 | 0.276 +/- 0.026 |
| FedAvg 8 + per-site fine-tune | 10 | 0.726 +/- 0.011 | 0.814 +/- 0.007 | 0.999 | 0.996 | 0.985 +/- 0.013 | 0.275 +/- 0.026 |
| Cold-start (camera never seen) | 5 | 0.627 +/- 0.047 | 0.549 +/- 0.015 | 1.000 | 0.998 | 0.000 | 0.200 +/- 0.059 |

Recall cells without an interval have a half-width under 0.01. `routine_animal`
is absent from MEVA and is not scored. Paired differences (same seed, same
holdout, so tighter than comparing the rows above):

| Comparison | Macro | Pooled |
|---|---|---|
| Robust 8 minus local-only 8 | -0.047 +/- 0.015 | -0.138 +/- 0.077 |
| Robust 8 + fine-tune minus local-only 8 | +0.003 +/- 0.008 | -0.001 +/- 0.004 |
| Robust 8 minus FedAvg 8 | +0.018 +/- 0.019 | +0.067 +/- 0.072 |
| Robust 8 + fine-tune minus FedAvg 8 + fine-tune | +0.001 +/- 0.001 | +0.001 +/- 0.001 |
| Robust 8 minus detector-only | +0.064 +/- 0.029 | -0.073 +/- 0.077 |
| Robust 8 + fine-tune minus detector-only | +0.115 +/- 0.020 | +0.064 +/- 0.008 |
| Cold-start minus untrained (5 seeds) | +0.463 +/- 0.164 | +0.382 +/- 0.082 |

## What it says

1. **Federation alone loses to local-only.** At every checkpoint, FedAvg and
   robust aggregation score 0.05-0.06 macro balanced accuracy below a node
   that simply trains on its own events (paired, round 8: -0.047 +/- 0.015).
   Same direction as the synthetic simulations of 10/04.
2. **Per-site fine-tuning recovers all of it, and no more.** Robust +
   fine-tune (0.727) is statistically indistinguishable from local-only
   (0.723; paired difference +0.003 +/- 0.008). **This data does not show
   federated training beating local-only.** What it shows is that the
   federated model is a sound starting point that fine-tuning turns into a
   local-quality model.
3. **The global model's failure is the package class.** Package recall is
   0.00 to 0.49 for the federated models and 0.98 for local ones. The cause is
   in the labels, not the optimiser: a bag is `routine_package` at G331 and
   G339 but **anomaly** at G299, where under 2% of events are bags (the
   converter's rare-class rule). The same 19 numbers carry opposite labels at
   different cameras and the model has no site identity, so the average
   learns neither. Local and fine-tuned models learn their camera's answer.
   This is the real argument for per-site fine-tuning, and it is a property of
   the labelling rule, so treat it as a lesson about the benchmark as much as
   about FL.
4. **A head beats no head by a small, real margin.** Robust + fine-tune is
   +0.115 +/- 0.020 macro and +0.064 +/- 0.008 pooled over the detector-only
   baseline, and the whole gain is anomaly recall, 0.00 to about 0.28. That
   recall is low in absolute terms: roughly 7 in 10 anomalous events are still
   called routine.
5. **Rounds 3, 5 and 8 barely differ** for local-only, FedAvg and fine-tune
   (all within one interval). Eight rounds only show in the robust arm, where
   package recall climbs from 0.00 to 0.49 with a wide interval (+/- 0.315);
   that is a few seeds recovering the class, not a trend to quote.
6. **Cold start is about the detector-only baseline.** A camera never seen by
   the federation scores 0.627 +/- 0.047 macro (5 seeds), which overlaps
   detector-only (0.612 +/- 0.017). Against the untrained head (0.185) the
   gain is large, but untrained is a poor reference. "A new node starts useful
   without local data" holds only in the sense that it is no worse than not
   having a head.
7. **Robust aggregation does no harm when nobody is attacking.** Across 80
   honest rounds (10 seeds x 8) it rejected one update (G339, one round, one
   seed): 1 in 720 updates, a false positive, harmless here. Robust 8 minus
   FedAvg 8 is +0.018 +/- 0.019 macro, i.e. no measurable cost or benefit on
   honest data; its value is the attack case (`sim/report/` and the hybrid
   demo).

## Caveats (state these out loud)

- **MEVA is daytime only and has no animals.** Rule (b) of the anomaly rule
  (off-hours) is close to vacuous, and `routine_animal` cannot be scored.
- **"Anomaly" means rare at this camera, not harmful.** The label is a
  function of class, activity and hour, using ground-truth activity labels the
  live node never has. A head with anomaly recall 0.28 is partly learning a
  labelling rule, not detecting danger. The number says nothing about threat
  detection.
- **The labels are camera-relative by construction** (finding 3). That
  inflates the case for fine-tuning and depresses FedAvg; a benchmark whose
  anomaly labels came from humans, or from one site-independent rule, could
  come out differently.
- **Class imbalance is severe.** G299, G326, G329 and G330 are 97-99% person.
  Several cameras have under 10 anomaly events in a holdout (G326 has none and
  is scored on person alone); their per-camera balanced accuracy is noisy, and
  G329 sits at 0.50-0.54 in every arm.
- **Unequal cameras.** Training is capped at 4,000 events per camera. Without
  the cap, G299 and G330 would hold over half the data; with it, FedAvg
  weights run from about 300 to 4,000 samples. Neither choice is the
  "natural" one.
- **Ground-truth tracks, not our detector.** The second pass through ARGUS's
  own YOLO (`scripts/ingest_video.py` on MEVA video) is not done; the gap it
  would show is detector error on top of these numbers.
- **One site.** All 9 cameras are the same facility, so non-IID-ness is
  between viewpoints, not between neighbourhoods. Transfer to a home camera is
  unmeasured until the Pi's 322 labelled events (or node B) are in.
- **Intervals cover seed noise, not dataset noise.** The 10 seeds reshuffle
  initialisation, batch order and which clips are held out, but the clip pool
  is fixed. Cold-start has 5 seeds, so its interval is wide.
- **Not run:** ShanghaiTech anomaly transfer, the 135-run matrix, attacks in
  this experiment (all clients honest), real gRPC rounds at 10 seeds.

## What to say at the showcase

> On real security-camera data, a shared model trained by federation is a
> good starting point but does not beat a camera training alone. After a
> short local fine-tune it matches local training and beats having no head at
> all, mainly by catching about 28% of unusual events that a detector alone
> catches none of. We found the shared model fails on anything whose meaning
> depends on the site, which is why every node keeps a local fine-tune step.

Do not say "federated learning improves accuracy". The data does not show it.

## Next

1. Add the Pi's 322 labelled events (or node B) as a tenth camera and re-run;
   the experiment accepts any store whose rows carry `meta.clip`.
2. Second pass through ARGUS's YOLO on MEVA video.
3. Re-label anomaly with a site-independent rule and re-run, to separate
   "federation does not help" from "the labels are camera-relative".
4. Add a poisoned peer to the arms (the demo shows it live; this would put a
   number on it).
