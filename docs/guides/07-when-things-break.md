# 7 · When things break

Every failure we actually hit, and what fixed it.

**PI** = terminal on the Raspberry Pi. **LAPTOP** = PowerShell on Windows.

---

## Start here · PI

```bash
cd ~/argus
./scripts/node_status.sh --once
```

One line tells you most of it: frame rate, whether the camera is fine,
which detector is loaded, and whether anything is in view.

---

## The camera

### "camera failed to initialize"

Work through these in order · PI:

```bash
cd ~/argus && source .venv/bin/activate
python -c "from picamera2 import Picamera2; print('ok')"
rpicam-hello --list-cameras
```

| Result | Cause | Fix |
|---|---|---|
| Error mentions **numpy** | A package upgrade broke the camera library | `pip install --ignore-installed "numpy>=2.2,<3"`, restart the node |
| `No module named picamera2` | Library missing, or the environment cannot see it | `sudo apt install -y python3-picamera2`, and rebuild the environment with `--system-site-packages` |
| No cameras listed | Ribbon cable | Power off, reseat both ends |
| `Device or resource busy` | Something else has the camera | Close it, restart the node |

### Everyone looks blue

Red and blue are swapped. It should be right by default; if your camera
really does hand back the other order · PI:

```bash
CAMERA_SWAP_RB=true ./scripts/run_node.sh
```

---

## Detection

### Only people are detected

Most likely the subject is holding still. The detector only runs on frames
where something moved · PI:

```bash
./scripts/run_node.sh --every-frame
```

Then check what the camera is actually seeing:

```bash
python scripts/check_classes.py --camera --save /tmp/seen.jpg
```

Open `/tmp/seen.jpg` — it has the boxes drawn on.

### Nothing at all is detected

```bash
python scripts/check_classes.py media/vtest.avi
```

If that finds things, the detector is fine and the problem is the scene:
the subject is too small, too fast, or a cardboard box (which is not
something the detector knows).

If that finds nothing either, the detector did not load. Check `backend` in
the status line — `none` means it failed to start.

### A box appears on screen but no event is saved

Events need three consecutive frames. A brief flicker is discarded on
purpose, otherwise the list fills with noise.

---

## Speed

### Frame rate is very low

| Check | Meaning |
|---|---|
| `backend` says `cpu` | No optimised detector. Run `python scripts/export_cpu_models.py --format ncnn` |
| You used `--every-frame` | Every frame is analysed; that is the cost |
| You used `--threat` | The extra stage costs around 100ms per frame on the Pi |
| Screen sharing is on | Sharing the Pi's screen takes processing power from the camera |

---

## The dashboard

### "SYSTEM OFFLINE" · LAPTOP

```powershell
curl.exe -s -m 5 http://10.0.0.140:8000/api/status
```

| Result | Meaning | Fix |
|---|---|---|
| Returns text | Network fine; wrong port allowed | Check the address bar — if it says 3001, restart the node with the startup script, which permits it |
| Times out | Laptop cannot reach the Pi | Different networks. Use the Pi's built-in page instead |

### Login fails

The password is stored on the Pi, not the laptop · PI:

```bash
python scripts/set_password.py --user admin
```

### Events do not update

The dashboard loads them once. Refresh the page.

---

## Learning

### The round returns an error about samples

It needs a few hundred labelled examples and you have a handful. Use
`./scripts/fl_trial.sh`, which prepares practice data automatically.

### The round never finishes

The coordinator is not reachable. If you used the script, it starts one
itself; if you started one by hand, check it is still running and that the
address matches.

### The update was rejected

Working as designed. The shared model scored worse on this camera's own
examples, so the camera kept the one it had.

---

## Scripts will not run

### `$'\r': command not found` · PI

The files arrived with Windows line endings:

```bash
cd ~/argus && git pull
chmod +x scripts/*.sh
```

### `No module named 'src'` · PI

You are running a script from outside the project. Run from `~/argus`, or
use the versions in `scripts/`, which handle this themselves.

### `ModuleNotFoundError: uvicorn` · PI

The environment is not active in that window:

```bash
cd ~/argus && source .venv/bin/activate
```

---

## Last resort

Check the window running the node. Errors are printed there, and it is
usually the fastest answer in the building.
