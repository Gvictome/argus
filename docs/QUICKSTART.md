# ARGUS quick start

Every block says which machine it runs on. **PI** is a Raspberry Pi Connect
shell; **LAPTOP** is PowerShell on Windows.

---

## 1 · PI — start the node

```bash
cd ~/argus
git pull
./scripts/run_node.sh
```

It prints the address to open and then stays running. Leave the window alone.

| Want | Command |
|---|---|
| Classify subjects that hold still | `./scripts/run_node.sh --every-frame` |
| Bigger detector | `./scripts/run_node.sh --model yolov8s` |
| Run against a recording, no camera | `./scripts/run_node.sh --video media/vtest.avi` |
| Add the dangerous-person stage | `./scripts/run_node.sh --threat` |

First run only, to get the fast CPU path:

```bash
python scripts/export_cpu_models.py --format ncnn
```

---

## 2 · Watch it

Open **`http://<pi-address>:8000/dashboard`** in any browser that can reach
the Pi. Live feed with boxes, detection FPS every second, what is in frame,
and events with the still saved for each.

Or from a shell:

### PI

```bash
cd ~/argus && ./scripts/node_status.sh
```

One line per second:

```
fps= 9.9 ms=   27 cap= 10.0 | tracks= 2 events=  14 motion=  3 stills=  14 | cpu-ncnn yolov8n gate=True rate=0.98 | fl-r2
```

`tracks` rises when something is in frame; `events` rises once it leaves.

---

## 3 · PI — check what the camera sees

```bash
cd ~/argus && source .venv/bin/activate
python scripts/check_classes.py --camera --save /tmp/seen.jpg
```

Prints the ARGUS classes and the raw COCO labels behind them, and writes an
annotated copy you can open. Point it at a file instead to test without the
camera:

```bash
python scripts/check_classes.py media/vtest.avi
```

**The four classes:** person · vehicle (bicycle, car, motorcycle, bus,
train, truck) · animal (bird, cat, dog, horse, sheep, cow) · package
(backpack, handbag, suitcase). COCO has no parcel class, so a cardboard box
detects as nothing — use a backpack.

---

## 4 · PI — a federated round

Node running in shell 1, then:

```bash
cd ~/argus && ./scripts/fl_trial.sh
```

It starts the aggregator, seeds samples, trains locally, joins the round,
and reports the gate's decision. Watch `detect_fps` while it runs —
training is a separate process on one core, so detection keeps going.

`model_version` moves to `fl-r1`, `fl-r2`… only when the gate **accepts**
the averaged model. A rejection means it scored worse on this node's own
held-out samples, and the node kept the model it had. That is the gate
working.

> The seeded samples are synthetic and never appear in the event list. Say
> so when presenting: the pipeline captures and trains on real events,
> demonstrated alongside a corpus large enough to train on.

---

## 5 · LAPTOP — the product dashboard (optional)

```powershell
cd $HOME\Documents\argus\argus
.\scripts\run_dashboard.ps1 -Node 10.0.0.140
```

Then open `http://localhost:3000` (or `:3001`) and log in as `admin`.

Set the password on the **PI**:

```bash
python scripts/set_password.py --user admin
```

**If it says SYSTEM OFFLINE**, your laptop cannot reach the Pi, or the port
Next.js chose is missing from `CORS_ORIGINS`. The node's own page at
`http://<pi>:8000/dashboard` has no such dependency.

---

## If it breaks

| Symptom | Fix |
|---|---|
| Only `person` detected | Subject is holding still → `--every-frame`; or it is too small in frame |
| Nothing detected at all | `python scripts/check_classes.py --camera` — separates model from scene |
| `camera_error` in status | `python -c "from picamera2 import Picamera2"`; if it mentions numpy → `pip install "numpy>=1.26,<2"` |
| People look blue | `CAMERA_SWAP_RB=true ./scripts/run_node.sh` |
| Round returns 409 | Fewer than 200 labelled samples — let `fl_trial.sh` seed them |
| Dashboard offline, curl works | Wrong port in `CORS_ORIGINS` (Next.js often uses 3001) |
| `backend: cpu` on the Pi | `python scripts/export_cpu_models.py --format ncnn` |
