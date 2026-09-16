# 4 · The dashboards

There are two, and they do different jobs.

---

## The built-in page — nothing to install

**`http://<pi-address>:8000/dashboard`**

Served by the Pi itself, so it needs no setup, no login, and no separate
program. It is the operator's view of a camera that is running right now.

| What you see | What it means |
|---|---|
| Live video with boxes | What the camera sees, labelled |
| **Detect FPS** | Frames actually analysed per second, updated every second |
| **Capture FPS** | Frames arriving from the camera |
| **ms/frame** | How long one frame takes to analyse |
| **In frame** | How many things are being tracked right now |
| **Events** | How many have been saved since it started |
| **Model** | Which version of the learned model is live |
| A green or amber badge | Amber says "Detecting 2" when things are in view |

Use this one for demos and for checking the camera is healthy.

---

## The product dashboard — runs on your laptop

A separate app (`adam9798/argus-dashboard`) with login, event history, and
the buttons that teach the system.

### Start it · LAPTOP

```powershell
cd $HOME\Documents\argus\argus
.\scripts\run_dashboard.ps1 -Node 10.0.0.140
```

Replace the address with your Pi's. The script clones it if needed, installs
it the first time, points it at the Pi, and starts it. Then open
**`http://localhost:3000`**.

Log in as **admin**, with the password you set on the Pi.

### What each part does

**Status strip** — frame rate, accelerator, model version, cameras online.
Three states: connecting, online, offline.

**Event list** — every saved event, newest first. Red marks something the
system judged unusual for this location.

**Detail panel** — click an event to see everything about it, and two
buttons:

- **Confirm** — the system got it right.
- **Correct** — it got it wrong. Type what it actually was, in plain
  English: "false alarm", "delivery driver", "suspicious person". It
  understands ordinary phrases.

**Those two buttons are how it learns.** Every click becomes a labelled
example that the next training round uses. Without them, the system has only
its own guesses to learn from, which teaches it nothing.

---

## Things that commonly go wrong

**"SYSTEM OFFLINE" but the Pi is fine.**
Two causes:

1. Your laptop cannot reach the Pi at all. Test it · LAPTOP:
   ```powershell
   curl.exe -s -m 5 http://10.0.0.140:8000/api/status
   ```
   No answer means different networks — use the built-in page instead.

2. The port. If port 3000 was busy, the dashboard quietly uses **3001**, and
   the Pi only allows the exact addresses it was told about. Check the
   address bar. The startup script already permits both.

**The event list does not update.**
It loads once when the page opens. Refresh it.

**Login fails.**
The password lives on the Pi. Set it there:

### PI

```bash
cd ~/argus && source .venv/bin/activate
python scripts/set_password.py --user admin
```

---

## Which to use when

| Situation | Use |
|---|---|
| Demo, or checking the camera works | Built-in page |
| Reviewing history, labelling events | Product dashboard |
| Laptop cannot reach the Pi | Built-in page, on a machine that can |
| Showing the product to someone | Product dashboard |

---

*Next: guide 5 explains what it detects and how to test it.*
