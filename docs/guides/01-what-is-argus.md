# 1 · What ARGUS is

A security camera that thinks on the device, and learns what is normal for
the place it watches — without sending video anywhere.

---

## The problem

A normal smart camera sends your video to a company's servers. Someone
else's computer decides what matters, you pay monthly for the privilege,
and it stops working when the internet does.

ARGUS keeps all of that on a small computer in your house.

---

## What it does, in order

**1. It watches.** A Raspberry Pi with a camera attached, running all the
time.

**2. It notices movement.** Comparing each frame with the one before is
cheap, so this runs constantly.

**3. It recognises things.** A model called YOLO looks at the picture and
says what is in it: a person, a vehicle, an animal, a package. It runs on
the Pi itself. No internet involved.

**4. It follows them.** If the same person is in thirty frames, that is
one event, not thirty. The system tracks each thing until it leaves.

**5. It saves what matters.** Each event gets a row in a small database, a
still photo, and optionally a video clip with a few seconds before and
after. Nothing else is kept.

**6. It learns what is normal here.** A small model looks at each event in
context — what it was, where, how fast, how long it stayed, what time of
day — and decides whether that is routine for *this* camera. A delivery
van at 2pm is routine. The same van at 3am is not.

**7. Cameras teach each other, without sharing video.** Every so often, a
camera sends what it has *learned* — a set of numbers, not footage — to be
averaged with what other cameras learned. It gets the combined result back.
This is called federated learning. **No image ever leaves the device.**

---

## The pieces

| Piece | Plain meaning |
|---|---|
| **Node** | One camera plus the Pi that runs it |
| **Detector (YOLO)** | The model that finds objects in a picture |
| **Event** | One thing that happened: a person crossed the view |
| **The head** | The small model that judges "normal here or not" |
| **Round** | One session of cameras sharing what they learned |
| **The gate** | A safety check: a shared update is only accepted if it does not make this camera worse |
| **Hailo** | An add-on chip that runs the detector far faster than the Pi's processor |

---

## What you see

**The built-in page** — open the Pi's address in a browser. Live video with
boxes drawn around what it sees, the speed it is running at, and a list of
recent events with photos.

**The dashboard** — a separate app with login, event history, and buttons to
confirm or correct what the system decided. Those corrections are how it
learns.

---

## What makes it different

- **Your video never leaves the house.** Not to us, not to anyone.
- **No subscription.** You buy the hardware once.
- **It works offline.** Unplug the internet and it keeps watching.
- **It adapts to your site.** What counts as unusual at your back door is
  different from a driveway, and it learns that difference.

---

## Honest limits, as of now

- **It does not recognise faces.** That was removed deliberately: it was
  slow, and identity is not the point. "Unusual for this site" is not "I
  know who that is."
- **Package means a backpack, handbag or suitcase.** The detector has no
  idea what a cardboard box is.
- **The fast chip is not in use yet.** The Hailo accelerator is installed
  and tested, but the detector still needs to be converted into its format.
  Until then the Pi does the work with its ordinary processor.

---

*Next: guide 2 sets up a Pi from scratch.*
