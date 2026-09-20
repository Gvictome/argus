# 6 · How the learning works

The part that makes ARGUS more than a camera with a detector attached.

---

## The idea in one paragraph

Several cameras, in different places, each learn from what they see. Instead
of sending their video somewhere to be pooled, each one sends only **what it
learned** — a list of numbers. Those get averaged and sent back, so every
camera benefits from what the others saw, and **no footage ever leaves any
device**. That is federated learning.

---

## What exactly is being learned

Not "what is a person" — the detector already knows that, and it does not
change.

What is learned is **what is normal at this particular camera**.

A small model looks at each event and its circumstances:

- what it was (person, vehicle, animal, package)
- how big it appeared, and where in the frame
- how fast it moved, and how long it stayed
- what time of day and day of week

and answers one question: **is this routine here, or unusual?**

A van in the driveway at 2pm is routine. The same van at 3am is not. A
person at the front door at noon is routine. Someone standing at the side of
the house for four minutes is not. That judgement depends entirely on the
location, which is exactly why each camera has to learn its own — and why
sharing what they learn is worth doing.

---

## Where the answers come from

The system cannot mark its own homework. Labels come from a person, through
**Confirm** and **Correct** in the dashboard. Every click becomes an example.

This is the honest part of the design: if the system generated its own
labels from a rule, then measured itself against that same rule, the
resulting accuracy figure would mean nothing.

---

## A round, step by step

1. The camera trains on its own labelled events.
2. It sends the **updated numbers** — never images — to a coordinator.
3. The coordinator averages everyone's numbers.
4. The averaged result comes back to each camera.
5. **The gate:** each camera tests the shared model against its own held-out
   examples. If it does worse than the one already running, it is
   **rejected** and the camera keeps what it had.

Step 5 matters. An average that suits four other sites can be wrong for
yours. Without the gate, one bad round quietly degrades a working camera.

---

## The two halves

**The camera** trains and sends. **The central server** averages and keeps
the history. They are different machines.

### The central server · LAPTOP

```powershell
cd $HOME\Documents\argus\argus
.\scripts\run_central_server.ps1
```

It prints the two lines to set on the Pi. It keeps every version of the
combined model in `central_server/checkpoints`, and it **stays up between
sessions** — a camera on a fortnightly schedule connects days later, so a
server that exited after the first session would not be there when it
mattered. Each session resumes from the last checkpoint.

Watch it at `http://localhost:8090/api/training/summary`, which lists every
round with its loss and accuracy, and `http://localhost:8090/api/nodes`,
which lists the cameras that have trained and what they scored.

### The camera · PI

```bash
export FL_ENABLED=true
export FL_SERVER_URL=192.168.1.50:8080        # the laptop
export FL_CENTRAL_API=http://192.168.1.50:8090
./scripts/run_node.sh
```

With `FL_ENABLED=true` the node runs rounds **on its own schedule** — every
14 days at 2am by default, only while nothing is moving, and it catches up
if the Pi was switched off across the due date. That is the cron.

## Running one by hand · PI

The node must already be running in another shell.

```bash
cd ~/argus
./scripts/fl_trial.sh
```

That does everything: starts a coordinator, prepares training data, trains
locally, joins the round, and reports the result. It runs **exactly** the
same round the schedule runs, so the demo and the unattended 2am version
cannot drift apart.

## Training on saved video · PI

Recordings are training data. Replay a clip through the detector and label
everything it finds in one go:

```bash
python scripts/ingest_video.py media/events/primary_human_2026.mp4 --label routine_person
python scripts/ingest_video.py trespasser.mp4 --label anomaly --every-frame
```

Each event comes out the same shape the live camera makes: the same
measurements, a still, and a crop of the subject. Timestamps come from the
clip's own frame rate, so a two-minute video replayed in fifteen seconds
does not record everything as moving four times too fast.

### Reading the result

| Field | Meaning |
|---|---|
| `accepted: true` | The shared model was better or equal — it is now live |
| `accepted: false` | It was worse here — the camera kept its own |
| `current` | How the running model scored |
| `candidate` | How the shared model scored |
| `model_version` | Changes to `fl-r1`, `fl-r2`… each time one is accepted |

**A rejection is not a failure.** It is the safety check doing its job, and
worth saying out loud when demonstrating.

---

## Watch the frame rate while it trains

Training happens in a separate process, on one processor core, so the camera
keeps watching throughout. Keep `./scripts/node_status.sh` visible during a
round — the frame rate barely moves. On a laptop the measured cost was 0.1%.

That is the whole argument for doing this on the device at all: a camera
that stops watching in order to learn is not a security camera.

---

## About the practice data

A camera needs a few hundred labelled examples before training is
meaningful. A short demo produces a dozen. So the trial script adds
**synthetic practice data** — realistic made-up events — to make the round
work.

**Say this when presenting.** The accurate sentence is: "it captures and
trains on real events, demonstrated alongside a synthetic set large enough
to train on." Those synthetic rows are marked as such and never appear in
the event list.

---

## What is not solved yet

- **Averaging across very different sites can hurt.** In testing, a round
  combining two dissimilar locations produced a worse model, and the gate
  correctly rejected it. Handling that properly is real research, not a
  setting to change.
- **More rounds help.** Three rounds starting from an untrained partner is
  not enough to converge.

---

*Next: guide 7 is what to do when something breaks.*
