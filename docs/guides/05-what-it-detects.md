# 5 · What it detects, and how to test it

---

## The four things it knows

| Class | What sets it off |
|---|---|
| **person** | a person |
| **vehicle** | bicycle, car, motorcycle, bus, train, truck |
| **animal** | bird, cat, dog, horse, sheep, cow |
| **package** | backpack, handbag, suitcase |

Everything else the detector could recognise — chairs, bottles, phones,
about sixty others — is thrown away before it reaches you. Those four are
what a security camera cares about.

**A cardboard box is not on the list.** The detector was trained on a
standard set of photographs that has no "parcel" in it. A real delivery box
will usually detect as nothing at all. Use a backpack to demonstrate the
package class. Teaching it real parcels means training a custom model, which
is a separate piece of work.

---

## How something becomes an event

1. **Movement is noticed.** Comparing consecutive frames is cheap, so this
   runs constantly.
2. **The detector runs.** By default only on frames where something moved.
3. **The thing is followed** from frame to frame.
4. **When it leaves, one event is saved** — not one per frame. A person
   crossing the view is a single event.
5. **Anything lasting under three frames is thrown away** as a flicker.

### The one that catches people out

By default the detector only runs when something is **moving**. Hold a
backpack perfectly still in front of the camera and nothing happens at all,
even though the same picture as a file detects fine.

To classify things that hold still · PI:

```bash
cd ~/argus
./scripts/run_node.sh --every-frame
```

This costs speed — every frame goes through the detector — but it is the
right setting when testing with props.

---

## Testing what it sees

### The direct check · PI

```bash
cd ~/argus && source .venv/bin/activate
python scripts/check_classes.py --camera --save /tmp/seen.jpg
```

This takes a frame from the running camera, runs the detector on it, and
prints what it found. It also saves a copy with the boxes drawn on, which
you can open and look at.

Output looks like:

```
ARGUS classes: {'human': 4, 'vehicle': 1}
COCO labels:   {'person(0)': 4, 'bus(5)': 1}
```

The top line is ARGUS's four classes. The bottom line is what the detector
actually called them. That distinction tells you whether something was
missed, or found and put in a different bucket.

### Test without the camera · PI

```bash
python scripts/check_classes.py media/vtest.avi
python scripts/check_classes.py some-photo.jpg
```

Useful for proving the detector works when the camera is not cooperating.

---

## Making props work

| Problem | Fix |
|---|---|
| Too small in frame | Hold it closer — roughly a third of the view |
| Moving too fast | Hold still for two seconds |
| Using a phone screen | Full brightness, tilt away from lights, fill the screen |
| Holding it still, nothing happens | Restart with `--every-frame` |
| Using a cardboard box | Use a backpack instead |

---

## What gets saved

| Thing | Where it goes |
|---|---|
| The event and its measurements | a small database file on the Pi |
| A still photo, taken when the subject was biggest | `data/snapshots/` |
| A video clip with a few seconds either side | `media/events/` |
| Long stretches of movement with nothing identified | logged separately, in chunks |

The photo is taken at the subject's largest, not at the end — by the time
something leaves the view it is usually half out of frame.

Clips only record if recording is on, which it is by default with
`run_node.sh`. Keep an eye on disk space if you leave it running overnight.

---

*Next: guide 6 explains the learning.*
