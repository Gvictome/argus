# 2 · Setting up the Pi

From a Raspberry Pi with nothing on it, to a camera that is watching.

Every command below is labelled. **PI** means a terminal on the Raspberry
Pi (a Raspberry Pi Connect shell is fine). **LAPTOP** means PowerShell on
Windows. Paste whole blocks.

---

## Before you start

You need: a Raspberry Pi 5, its camera connected by the ribbon cable, power,
and a network connection. The AI HAT+ accelerator is optional at this stage.

---

## Step 1 — system packages · PI

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-picamera2 python3-opencv build-essential cmake
```

`python3-picamera2` is the camera library. It must come from `apt`, not from
`pip` — building it yourself on a Pi rarely ends well.

---

## Step 2 — get the code · PI

```bash
git clone https://github.com/Gvictome/argus.git ~/argus
cd ~/argus
git checkout feat/threat-detection
```

**The branch matters.** The repository's default branch is old and does not
have any of this.

---

## Step 3 — Python environment · PI

```bash
cd ~/argus
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`--system-site-packages` is not optional — without it the environment cannot
see the camera library you installed in step 1, and the camera will never
start.

Expect this to take a while. It downloads a lot.

---

## Step 4 — check the camera · PI

```bash
python -c "from picamera2 import Picamera2; print('camera library ok')"
rpicam-hello --list-cameras
```

You want the words `camera library ok` and at least one camera listed.

**If the first line complains about numpy**, this fixes it:

```bash
pip install --ignore-installed "numpy>=2.2,<3"
```

**If no cameras are listed**, power the Pi off and reseat the ribbon cable at
both ends. Nothing further will work until this passes.

---

## Step 5 — make detection fast · PI

```bash
cd ~/argus && source .venv/bin/activate
python scripts/export_cpu_models.py --format ncnn
```

This converts the detector into a format the Pi's processor handles much
better. It takes about a minute and you only do it once. It is picked up
automatically from then on.

---

## Step 6 — set a password · PI

```bash
python scripts/set_password.py --user admin
```

It asks twice and shows nothing as you type. You need this to log into the
dashboard later.

---

## Step 7 — start it · PI

```bash
cd ~/argus
./scripts/run_node.sh
```

It prints the address to open, then stays running. **Leave this window
alone** — closing it stops the camera.

---

## Step 8 — confirm it is alive · PI

Open a second terminal:

```bash
cd ~/argus
./scripts/node_status.sh
```

One line per second. What to look for:

| Reading | Means |
|---|---|
| `fps=` above 0 | It is processing frames |
| `tracks=` rising when you step in front | It sees you |
| `events=` rising after you leave | It saved what it saw |
| `CAMERA:` appearing | Something is wrong with the camera |

Press `Ctrl+C` to stop watching. That does not stop the camera.

---

## Step 9 — look at it

Open **`http://<your-pi-address>:8000/dashboard`** in a browser on any
machine that can reach the Pi. Find the address with:

### PI

```bash
hostname -I
```

Use the first number, for example `10.0.0.140`, so the address is
`http://10.0.0.140:8000/dashboard`.

---

## If the Pi and your laptop are on different networks

This is common when using Raspberry Pi Connect. Your browser cannot reach
the Pi directly, so:

- Everything still works from the Pi's own shells.
- To see it in a browser, either use Pi Connect's screen sharing and open
  the address on the Pi itself, or put both machines on the same network.

---

*Next: guide 3 runs the demo.*
