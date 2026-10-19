# ARGUS Demo Script

What the demo shows, what to say, what to type, and what the codebase
looks like when someone asks. Rewritten 2026-10-18 for the **one real node
plus simulated peers** reality: the second node's parts have not arrived, so
every federated claim below is made with one real node and eight or fewer
simulated MEVA cameras, and says so.

Runtime: **8 minutes** for the full walkthrough, **90 seconds** for the
short version if a judge is passing through.

---

## The 30-second version

> Every security camera makes you choose: send your footage to a company
> and get a system that improves, or keep it private and get one that
> never learns. ARGUS does both. Video is processed and stored entirely
> on this device, nothing leaves it. But every two weeks it sends what it
> *learned* to a shared model, as numbers, never images. And because we do
> not control the devices contributing, we built the part that makes that
> safe: screening that stops one broken device from poisoning everyone's
> model.

---

## Part 1 - Detection, live (2 min)

Open the dashboard (`http://<node>:8000/dashboard`). Point at the badges
before anything else: **on-device, frame rate, which backend is live, which
model loaded.**

> Everything you are about to see runs on this board. No internet
> connection is involved in any of it.

### 1.1 Walk into frame

> Motion fires first. It is nearly free, and it is the gate. Only when
> something moves does the object model run, so a still room costs almost
> nothing. That green box is the person class. One event per tracked
> object, not one per frame.

### 1.2 The classes

| Class | Source | What triggers it |
|-------|--------|------------------|
| `person` | YOLOv8n | A person |
| `vehicle` | YOLOv8n | Bicycle, car, motorcycle, bus, train, truck |
| `animal` | YOLOv8n | Bird, cat, dog, horse, sheep, cow |
| `package` | YOLOv8n | Backpack, handbag, suitcase (COCO has no parcel class) |
| `anomaly` | The federated head | "Unusual for this site", decided from 19 numbers per event |

> Four things the detector can see, and one judgement the federated model
> adds: is this routine *for this camera*. A van at 2 pm is routine; the
> same van at 3 am is not. That site-specific judgement is the part worth
> federating.

**Not in this demo, and say so if asked:** face recognition (removed
2026-09-14, it cut the frame rate 12-27x) and, unless it was tested on the
day, the threat stage (`THREAT_ENABLED=false`; it is expensive on the Pi's
CPU until a Hailo build exists). The base detector is **YOLOv8n**, locked
2026-10-18 (`docs/research/detector-comparison.md`).

---

## Part 2 - Recording (1 min)

> The cameras run continuously. The disk does not.

Show `/api/clips` or the clips list.

> We only write footage around actual detections: the clip starts before
> the trigger and continues after it, and someone standing around produces
> one clip, not three hundred. Motion alone does not trigger recording; it
> fires on clouds and branches.

---

## Part 3 - Federated learning, the hybrid demo (4 min)

This is the part that is research rather than assembly. Spend the time, and
**open with the honest framing**:

> Federation needs more than one device. We have built one physical node.
> So the federation you are about to see is **this one real node plus
> simulated peers**: each peer is a real security camera from the public
> MEVA dataset (CC-BY), with its own real, differently-behaved footage and
> its own labels, running the same training code. The node on the table is
> the real one. When the second node's parts arrive, it joins with the same
> command.

### 3.1 The problem and the inversion

> To improve a security model the normal way, you collect everyone's
> footage centrally. That means shipping video of people's homes to a
> company.

```
   Day 0        every device has the same model
   Days 1-13    each device learns from its own camera, nothing is sent
   Day 14       each device sends only MODEL CHANGES, numbers
   server       screens them, combines the safe ones, sends back a model
```

> The model travels to the data instead of the data travelling to the
> model. No image crosses the network.

```bash
curl http://<host>:8000/api/federated/status
```

> The next-due time matters: "it runs every two weeks" is unfalsifiable
> without it. The schedule is written to disk, so a reboot does not reset
> the 14-day clock.

### 3.2 Launch it - one command

On the laptop (aggregator + simulated MEVA peers), with the Pi pointed at it:

```bash
python scripts/demo_hybrid.py --peers 6 --rounds 5
```

The launcher prints the exact command to run on the Pi (`node_client
--server <laptop-ip>:8080`). **Before the showcase, rehearse with the
stand-in** (the same pipeline on a video file, on the laptop, no Pi
needed):

```bash
python scripts/demo_hybrid.py --standin media/vtest.avi --peers 6 --rounds 5
```

Allow about four minutes from launch to the first round: every Python
process imports the Flower stack (about 40 s each on this laptop).
**Start it before you start talking, not when you reach this slide.**

What the console shows each round:

```
round 3: balanced accuracy 0.551   loss 0.255
  recall: routine_person 1.00  routine_vehicle 1.00  routine_package 0.00  anomaly 0.21
  per camera: G330 0.72  G299 0.73  G336 0.73  G339 0.52  G328 0.69
  updates: accepted 6, rejected 1   [direction + trimmed-mean filters ON]
  REJECTED mallory: update norm 69.06 exceeds 3.0x median 8.504
```

Read it left to right: **balanced accuracy** (never plain accuracy, because
person dominates every camera), **recall per class** on footage no node
trained on (whole clips held out), **per camera**, then **which updates the
aggregator accepted**.

Say this about the numbers:

> Round 0 is the untrained model. Balanced accuracy climbs with each round
> as one real node and six cameras average what they learned. These are
> short demo runs; the proper comparison is a separate experiment with ten
> seeds and confidence intervals, and its honest result is below.

### 3.3 The attack - the poisoned peer toggle

```bash
python scripts/demo_hybrid.py --standin media/vtest.avi --peers 6 --rounds 5 --poison signflip
```

> One extra peer, `mallory`, sends the opposite of what it learned,
> boosted five times. Against plain federated averaging that drags the
> shared model backwards for everyone. Watch the console: `mallory` is
> rejected every round, and the accuracy line keeps climbing.

Four filters, on the poster:

| Filter | Catches |
|--------|---------|
| Structural | Malformed data, NaN or infinite values (`--poison nan`) |
| Magnitude | Updates far larger than peers, trying to overwrite the model |
| Direction | Updates opposing everyone else, deliberate sabotage |
| Trimmed mean | Remaining outliers |

> Then a final check on the node: we score the candidate model on this
> camera's own held-out events, and if it is worse than the one it would
> replace, we throw it away and keep the old one.

Two lines that land with a technical judge:

> We compare model *changes*, not model weights. Raw weights are dominated
> by the shared starting point, so two updates that disagree completely
> still look almost identical.
>
> And consensus uses the median, not the mean, because the mean is exactly
> what an attacker is trying to drag.

### 3.4 Say the limitations out loud

> Two of those filters compare devices against each other, so they need at
> least three contributors. **With one real node and six simulated peers
> they are active, but that is seven contributors of which only one is a
> real device.** With the real node alone the aggregator would fall back to
> the checks that work regardless. We are not claiming a seven-node
> deployment.

> What the experiment on the nine MEVA cameras found (10 seeds, 95%
> intervals, `docs/research/fl-vs-base-meva.md`): federation alone does
> **not** beat a camera training on its own. Federation followed by a short
> local fine-tune **matches** local training (0.727 vs 0.723 macro balanced
> accuracy) and beats having no head at all (0.727 vs 0.612). The shared
> model fails on anything whose meaning depends on the site, which is why
> every node keeps a local fine-tune step. We do not claim federation
> improves accuracy, because on this data it does not.

Volunteering this is worth more than being caught on it.

### 3.5 If asked: when node B arrives

> Same command, one more number: `--real-nodes 2`. The aggregator waits for
> one more client and node B joins with the node client pointed at the
> laptop. Nothing else changes. Two real nodes is still short of the three
> the direction and trimmed-mean filters need, so the simulated peers stay.

---

## Part 4 - Privacy and security (1 min)

| Leaves the device | Never leaves the device |
|-------------------|-------------------------|
| Model weight arrays | Camera frames |
| Sample counts | Recorded clips |
| Loss and accuracy | Stills and crops |
| A device ID | The events themselves |

If asked about storage, answer straight:

> Federated learning means data does not *leave* the device. It does not
> mean nothing is stored. Each device keeps its events, stills and clips
> locally, on hardware the customer owns. "We never store your footage"
> would be false. "Your footage never leaves your device" is true.

---

## Queries - every command, in demo order

### Setup

```bash
python scripts/fetch_models.py                    # threat weights, checksum-verified (optional)
python -m pytest tests/ -q
python main.py
```

### Health and status

```bash
curl localhost:8000/api/status
curl localhost:8000/api/detection/status
```

| Field | If it is wrong |
|-------|----------------|
| `"backend": "cpu"` | Expected today: no Hailo build exists (custom compile is future work). Check the NCNN export loaded |
| `"fps": 0.0` | Fewer than two frames processed yet |
| `worker.camera_error` | Camera did not open; see the runbook, section E |

### Recordings and federation

```bash
curl localhost:8000/api/clips
curl localhost:8000/api/federated/status
```

### The hybrid demo

```bash
python scripts/demo_hybrid.py --standin media/vtest.avi --peers 6 --rounds 5               # rehearsal, no Pi
python scripts/demo_hybrid.py --peers 6 --rounds 5                                         # Pi joins the laptop
python scripts/demo_hybrid.py --standin media/vtest.avi --peers 6 --rounds 5 --poison signflip
python scripts/demo_hybrid.py --peers 6 --rounds 5 --real-nodes 2                          # once node B exists
```

Logs for a run: `sim/_run/hybrid/` (`events.jsonl`, `summary.json`,
`server.log`, one log per peer).

### Authentication (only when exposed)

```bash
AUTH_REQUIRED=true ARGUS_ADMIN_PASSWORD='...' python main.py
curl -X POST localhost:8000/api/auth/login \
     -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"..."}'
curl -H "Authorization: Bearer <token>" localhost:8000/api/events
scripts/run_tunnel.sh --check
```

### Reproducing the numbers

```bash
python sim/meva_experiment.py --seeds 10 --cold-seeds 5 --workers 10    # federated vs local, ~8 min
python scripts/benchmark_classes.py --dataset <split>
python scripts/tune_threat_threshold.py --dataset <split> --negatives <dir>
```

---

## If the demo breaks

| Symptom | Do this |
|---------|---------|
| Hybrid demo never starts rounds | It is waiting for a client. Check `sim/_run/hybrid/server.log`; the launcher prints how many clients it expects. The Pi missing is the usual cause |
| Pi cannot reach the laptop | Firewall on port 8080, or the wrong laptop IP. Run the stand-in instead and say the Pi is the same code |
| Round is slow | Every process imports Flower (about 40 s). Start the launcher before the talk |
| Threat stage misbehaving | `THREAT_ENABLED=false` and restart |
| Frame rate sagging | Raise `detect_every` on the stream URL, or re-export NCNN at 416 |
| Camera dead | `POST /api/cameras/init`; check `/api/cameras` |
| Everything wrong | Play the backup video and keep talking |

---

## Codebase breakdown

Counts change quickly; regenerate before quoting them
(`git ls-files | xargs wc -l` or `pytest --collect-only -q | tail -1`).
The shape is stable: tests are the largest single body of code, which is
the point, because on a demo day the failures that matter are the silent
ones.

### Where to look for each claim

| Claim | File |
|-------|------|
| Cascade, gating, classes | `src/detection/__init__.py` |
| Event tracking, features | `src/federated/collector.py`, `features.py` |
| **Robust aggregation** | `src/federated/robust.py` |
| Robust aggregation wired into Flower | `src/federated/robust_strategy.py` |
| 14-day schedule, persistence | `src/federated/scheduler.py` |
| Validation gate, model versions | `src/federated/model_state.py` |
| MEVA converter | `src/federated/meva.py`, `scripts/meva_to_events.py` |
| **Hybrid demo launcher** | `scripts/demo_hybrid.py` |
| Poisoned-peer attacks, round evaluation | `sim/poison.py`, `sim/metrics.py` |
| Federated vs local experiment | `sim/meva_experiment.py` |
| Authentication | `src/api/auth.py` |
| Adversarial tests | `tests/test_robust_aggregation.py` |

---

## Questions to expect

**"How accurate is it?"**
> We never quote plain accuracy, because person is most of every security
> camera. On nine real MEVA cameras the federated head, after a local
> fine-tune, scores 0.73 balanced accuracy against 0.61 for the detector
> alone, with all of the gain in catching unusual events (recall about 0.28
> against 0). That is "unusual at this camera", not "dangerous".

**"Is this actually federated? You have one device."**
> One real device and simulated peers, and we say so on the first slide. The
> simulated peers are real cameras and run the same training and
> aggregation code, so the screening, the rounds and the numbers are real;
> the network of independent owners is not. The second node's parts are on
> order and it joins with one flag.

**"Does federating beat training on the device alone?"**
> Not by itself, and not on this data. Federation plus a short local
> fine-tune ties local training and beats having no model. Federation's
> argument here is privacy and the shared starting point, not accuracy.

**"Why not just use the cloud?"**
> Because the price of cloud intelligence is your footage. We wanted both
> properties, and federated learning is how you get them.

**"Could someone poison your model?"**
> That is the exact problem we worked on. Against plain federated averaging,
> trivially: one sign-flipped update. In the demo you saw one such peer
> rejected every round. We also have a test that shows the attack
> succeeding without the screening and failing with it. The limit: the
> subtle attack, a quiet label flip under the norm limit, is only partly
> caught.

**"What does it not do?"**
> No night vision, and MEVA is daytime only. No faces. No Hailo
> acceleration yet: a custom compile needs the Hailo toolchain and the Pi
> runs on CPU. No signed updates, so we defend against bad updates but not
> forged identities. No encryption at rest yet.

**"Is this your own model?"**
> The object model is a standard architecture (YOLOv8n). The threat model is
> a third-party graduation thesis artifact, attributed in `NOTICE.md`, and
> the reason the project is AGPL-3.0. Our contribution is the integration,
> the measurement, and the aggregation defence.
