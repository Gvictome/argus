# ARGUS demo runbook

Prototype node + `adam9798/argus-dashboard`, end to end: capture, classify,
label, federate, update.

**Two machines.** The Pi runs the node; your laptop runs the dashboard and the
FL aggregator.

| | Pi 5 prototype | Laptop |
|---|---|---|
| Runs | node API, camera, detection, FL client | dashboard, FL aggregator |
| Port | 8000 | 3000 (dashboard), 8080 (aggregator) |

Same LAN. Get the Pi's address once:

```bash
hostname -I | awk '{print $1}'        # e.g. 192.168.1.42
```

---

## Rehearse this once before demo day

Not on demo day. The first run always finds something.

---

## 1 · Pi: start the node

```bash
cd ~/argus && git pull
source .venv/bin/activate
pip install "flwr>=1.36" prometheus-client

export NODE_NAME="Node A"
export CAMERA_ZONE="Front door"
export CORS_ORIGINS="http://localhost:3000"
export AUTH_REQUIRED=false            # LAN only — see §7
export FL_ENABLED=false               # 02:00 scheduler off; we trigger by hand

uvicorn "src.api.app:create_app" --factory --host 0.0.0.0 --port 8000
```

**Check the accelerator.** This is the most common silent failure and it reads
downstream as "detection is broken":

```bash
curl -s localhost:8000/api/status | python -m json.tool
```

Want `"accelerator": "Hailo-10H"`. If it says `"CPU"`, no Hailo export was
found — see [`HAILO_PIPELINE.md`](HAILO_PIPELINE.md). The demo still works on
CPU, just at a few FPS.

---

## 2 · Laptop: start the dashboard

```bash
git clone https://github.com/adam9798/argus-dashboard
cd argus-dashboard
npm install
cp .env.local.example .env.local
```

Edit `.env.local` — **this is the only configuration step**:

```
NEXT_PUBLIC_API_BASE_URL=http://192.168.1.42:8000
AUTH_COOKIE_SECURE=false
```

```bash
npm run dev          # http://localhost:3000
```

**Do not run `npm run mock-backend`.** That is the throwaway stand-in; the
whole point is that the Pi is now the backend.

Log in with the node's admin credentials (`scripts/set_password.py` on the Pi
if you need to set one). The status strip should show live FPS, `Hailo-10H`,
a model version, and 1 node online.

---

## 3 · Capture

Point the camera at a doorway or driveway. Do these three, deliberately:

1. **Walk straight through** — normal transit
2. **Walk in, stop, linger 10+ seconds, leave** — this is the one that matters;
   dwell is the feature the head keys on
3. **Have someone drive or cycle past** — a second class

The collector emits **one sample per track**, not per frame. One person
walking through is one row, not thirty. Tracks under 3 frames are discarded as
detector flicker.

Events appear in the dashboard list within a second or two. Verify on the Pi
if the list stays empty:

```bash
curl -s localhost:8000/api/events | python -m json.tool | head -30
```

If this returns rows but the dashboard is empty, the dashboard is pointed at
the wrong host or CORS is blocking — check the browser console, not the Pi.

---

## 4 · Label — the part that makes it honest

Click an event, then **Confirm** or **Correct** in the detail panel.

This is not decoration. The simulator derives labels from the same rule that
generates its data, which is fine for timing and worthless as evidence about
accuracy. On the node the label has to come from outside the model, and this
is where it comes from. Every click writes a label the next FL round trains on.

- **Confirm** — the call was right
- **Correct** — flips it, or set a specific class

Label the lingering pass as an anomaly and the ordinary walk-throughs as
routine. That contrast is the entire signal.

> **Say this out loud during the demo.** A flagged event is the federated
> head's judgement, not a face match. Face recognition is out of scope; no
> biometric template is ever computed. The red indicator means "unusual for
> this site", which is a stronger privacy position, not a weaker one.

---

## 5 · Seed enough data to train on

Twenty minutes of walking gives ~30 samples. The head has 205,445 parameters.

```bash
curl -sX POST localhost:8000/api/federated/samples/bootstrap \
     -H 'Content-Type: application/json' \
     -d '{"site": "driveway", "n": 5000}'
```

Seeded rows are tagged `synthetic: true` and **never appear in the dashboard**
— that list only ever shows what the camera actually saw.

**Disclose this.** The defensible claim is: *"the pipeline captures and trains
on real events, demonstrated alongside a synthetic corpus sized to make
training meaningful."* Not "we trained on real data."

---

## 6 · Train, then federate

**Local first.** Always. It separates a modelling problem from a network
problem, which otherwise look identical:

```bash
curl -sX POST localhost:8000/api/federated/head/train \
     -H 'Content-Type: application/json' -d '{"epochs": 8}'
```

Quote **balanced accuracy**, not accuracy. `DETECTION_ACCURACY.md` §0 explains
why: on skewed data, a model that fires on everything scores well on plain
accuracy without looking at anything.

**Laptop — aggregator:**

```bash
cd argus
python sim/fl.py serve --rounds 5 --clients 1 --address 0.0.0.0:8080
```

**Pi — join:**

```bash
curl -sX POST localhost:8000/api/federated/round \
     -H 'Content-Type: application/json' \
     -d '{"server": "192.168.1.50:8080", "epochs": 6}'
```

**Watch:**

```bash
curl -s localhost:8000/api/federated/round | python -m json.tool
```

Each round: server sends global weights → Pi injects them → trains on its own
samples → returns **weights only, never footage** → server runs FedAvg → sends
the averaged model back.

That return trip is the update coming back, and it is the privacy claim made
concrete: **the aggregator never sees a frame.** Say it while it is happening.

Afterwards, the dashboard's `model_version` increments and re-labelled events
reflect the retrained head.

### Two nodes

One client means FedAvg averages a single update — the full path runs but the
accuracy curve is just local training. For real averaging, add a second client
from the laptop (Flower clients are hardware-agnostic):

```bash
python sim/fl.py client --site backdoor --address 192.168.1.50:8080
```

Start the server with `--clients 2`. The sites have different anomaly rules, so
this is genuinely non-IID — unlike the Phase 1 CIFAR-10 result, which used
`IidPartitioner` and structurally cannot show this.

---

## 7 · Before any tunnel

`AUTH_REQUIRED=false` is fine on an isolated LAN and unacceptable anywhere
else. A tunnel publishes a live camera feed pointed at real people. Read
[`CLOUDFLARE_TUNNEL.md`](CLOUDFLARE_TUNNEL.md): it requires Cloudflare Access
**and** `AUTH_REQUIRED=true`, and `scripts/run_tunnel.sh` refuses to start
without the second.

---

## If it breaks

| Symptom | Cause |
|---|---|
| Status strip stuck "connecting" | `NEXT_PUBLIC_API_BASE_URL` wrong, or Pi not reachable. `curl` it from the laptop. |
| Status strip offline, curl works | CORS. `CORS_ORIGINS` must include `http://localhost:3000`. |
| Login fails | No admin account. `python scripts/set_password.py` on the Pi. |
| Event list empty, `/api/events` has rows | Dashboard pointed at the mock, or wrong host. |
| `/api/events` empty too | Nothing tracked. Check `/api/detection/status`; if `backend: none`, YOLO never loaded. |
| `accelerator: CPU` | No Hailo export present. |
| Round 409s | Not enough labelled samples. Bootstrap (§5). |
| Round starts, never finishes | Aggregator not reachable — firewall on 8080, or wrong laptop IP. |

---

## Known gaps — say them before someone finds them

- **No Pi 5 + Hailo-10H benchmark exists.** Every FPS number in Rev 2.2 is a
  budget, not a measurement. `scripts/benchmark_classes.py` on the Pi is what
  replaces them.
- **The threat model may not be compiled to Hailo.** If not it is a second CPU
  YOLO pass at ~36 ms/frame. The simulator puts the ceiling at **10.8 FPS**
  uncompiled vs **14.9 FPS** compiled — the difference between hitting the
  12 FPS target and missing it.
- **`LocalTrainer` still fine-tunes YOLO**, the pre-F-1 design, contradicting
  CON-3's 200K-parameter bound. This flow bypasses it, but it is still wired
  into `FLScheduler` — so the 02:00 scheduled round does something different
  from the manual one. Reconcile before trusting the scheduler.
- **Fast movers can still fragment.** Association is IoU with a centre-distance
  fallback; something crossing the frame in two frames may split. Raise
  `reach_factor` if a car reliably produces two events.
- **`requirements.txt` pins `flwr>=1.13.0`** but 1.36 removed the API 1.13
  documented. Pin the range you actually test against.

---

## Demo-day card

| | |
|---|---|
| Node | `uvicorn "src.api.app:create_app" --factory --host 0.0.0.0 --port 8000` |
| Dashboard | `npm run dev` (**not** `mock-backend`) |
| Health | `curl -s localhost:8000/api/status` |
| Events | `curl -s localhost:8000/api/events` |
| Seed | `curl -sX POST .../samples/bootstrap -d '{"site":"driveway","n":5000}'` |
| Train | `curl -sX POST .../head/train -d '{"epochs":8}'` |
| Aggregator | `python sim/fl.py serve --rounds 5 --clients 1 --address 0.0.0.0:8080` |
| Join | `curl -sX POST .../federated/round -d '{"server":"<laptop>:8080"}'` |
| Round status | `curl -s localhost:8000/api/federated/round` |
