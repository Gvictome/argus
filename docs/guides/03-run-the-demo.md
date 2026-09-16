# 3 · Running the demo

What to run, in order, and what to say while it runs.

**PI** = a terminal on the Raspberry Pi. **LAPTOP** = PowerShell on Windows.

---

## Rehearse this once before the day. The first run always finds something.

---

## Shell 1 — start the camera · PI

```bash
cd ~/argus
./scripts/run_node.sh
```

Leave it running for the whole demo.

---

## Shell 2 — the live numbers · PI

```bash
cd ~/argus
./scripts/node_status.sh
```

Put this window where people can see it. It updates every second and it is
the proof that everything else is real.

---

## The screen — open the page

**`http://<pi-address>:8000/dashboard`**

Live video with boxes, the speed, what is in frame, and recent events with
photos.

---

## The walkthrough

### 1. Show it watching

Point at the live view. Walk into frame.

> "It is running the detection on the Pi itself. Nothing is going to a
> server — no internet needed."

The `tracks` number in shell 2 rises the moment you appear.

### 2. Show it remembering

Walk out of frame. An event appears in the list with a photo of you.

> "One event per person, not one per frame. It saved a still, and the event
> went into a database on the device."

### 3. Show what it knows

Have a backpack ready, and a phone with a photo of a car on it.

> "It knows four things: people, vehicles, animals, and packages."

Hold the backpack up. Hold the car photo up, filling the screen.

**If something does not register:** hold it closer so it fills a third of
the view, and hold still for two seconds. A cardboard box will not work —
the detector has no idea what one is.

### 4. Show the learning · PI, shell 3

```bash
cd ~/argus
./scripts/fl_trial.sh
```

While it runs:

> "Each camera trains on its own events and sends only what it learned —
> numbers, not footage. Those get averaged with other cameras and sent
> back. No video ever leaves the device."

Point at shell 2 while this happens:

> "The frame rate is not dropping. Training runs on a separate core, so the
> camera never stops watching to learn."

When it finishes it prints whether the update was **accepted** or
**rejected**:

> "It only accepts the shared model if it does not make this camera worse
> on its own data. A rejection is the safety check working, not a failure."

### 5. Show the privacy claim

> "Everything you have seen — detection, storage, learning — happened on a
> £80 computer in this room. The only thing that ever leaves is a set of
> numbers, and you cannot reconstruct a picture from them."

---

## What to say if asked

**"How accurate is it?"**
On a standard benchmark image the detector is confident and correct. On the
learning side, accuracy is measured against events labelled by a person, and
the number is shown when a round runs. We do not quote a single headline
figure because it depends on the site.

**"Is that real-time?"**
Yes — the frame rate on screen is measured, not a target.

**"Does it recognise me?"**
No, and deliberately. No face recognition at all. It flags things that are
*unusual for this location*, which is a different claim and a safer one.

**"What is the accelerator for?"**
It runs the detector far faster than the Pi's processor. It is installed and
tested at 76 frames per second. Connecting it to ARGUS itself is the next
piece of work.

---

## Before you present

- Seeded training data is **synthetic**. Say so. The honest sentence is:
  "it captures and trains on real events, demonstrated alongside a synthetic
  set large enough to train on."
- Frame rates measured on a laptop are not Pi numbers. Quote what shell 2
  shows.
- If recording is on, check disk space. Clips are a few megabytes each.

---

*Next: guide 4 covers the dashboards.*
