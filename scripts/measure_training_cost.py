#!/usr/bin/env python3
"""
What does training cost the detector?  Run this ON THE PI.

Training is supposed to be free: the local head trains in a worker thread
and a federated round runs as its own process, so detection is meant to
carry on at full rate. That was measured at 0.1% on a laptop and never on
the Pi, which has four cores instead of eight and no headroom to spare.
This samples the node's own throughput before, during and after a run and
prints the difference.

    python scripts/measure_training_cost.py                     # local train
    python scripts/measure_training_cost.py --mode round        # full round
    python scripts/measure_training_cost.py --baseline 30       # longer window

Read the three numbers it reports like this:

  capture_fps  the camera loop. The verdict lives here -- if training
               starves the pipeline, this is what the demo loses.
  detect_ms    how long one inference takes. Rises under CPU contention
               even when the frame rate holds, so it shows the squeeze
               before it becomes visible.
  detect_fps   how often YOLO ran. Gated on motion, so it tracks what is
               happening in front of the camera, not the training. Compare
               it only if the scene is steady.

So keep the scene the same throughout -- an empty driveway for the whole
run, or someone walking back and forth for the whole run, not one then the
other. Needs the node already running and at least `min_samples` labelled.

Measure against the camera, not a video file. A file source is played back
at the clip's own frame rate, so capture_fps sits at that rate whatever the
CPU is doing and only moves once the pipeline is starved badly enough to
miss the schedule. It validates the method; it cannot show the headroom.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FIELDS = ("capture_fps", "detect_fps", "detect_ms")

# Local training on a few thousand samples finishes in well under a second,
# which can leave one sample inside the training window -- and one sample
# sits anywhere in the baseline's own spread. Below this, report the window
# as too short instead of printing a number that looks like a measurement.
MIN_TRAINING_SAMPLES = 5


def _get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, body: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Sampler(threading.Thread):
    """Polls /api/detection/status until told to stop, tagging each sample."""

    def __init__(self, base: str, interval: float):
        super().__init__(daemon=True)
        self.base, self.interval = base, interval
        self.phase = "baseline"
        self.samples: list = []
        self._stop = threading.Event()
        self.errors = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                worker = _get(f"{self.base}/api/detection/status").get("worker")
                if worker:
                    self.samples.append(
                        {"phase": self.phase,
                         **{f: worker.get(f) for f in FIELDS}})
            except (urllib.error.URLError, OSError, ValueError):
                self.errors += 1
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def of(self, phase: str, field: str) -> list:
        return [s[field] for s in self.samples
                if s["phase"] == phase and isinstance(s.get(field), (int, float))]


def _row(label: str, values: list) -> str:
    if not values:
        return f"  {label:12s} no samples"
    return (f"  {label:12s} median {statistics.median(values):7.2f}   "
            f"min {min(values):7.2f}   max {max(values):7.2f}   "
            f"n={len(values)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=("local", "round"), default="local",
                    help="local = train the head in-process; "
                         "round = full federated round in its own process")
    ap.add_argument("--baseline", type=float, default=20.0,
                    help="seconds of before/after sampling")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between samples")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--server", default="", help="aggregator URL, round mode")
    ap.add_argument("--json", default="", help="also write the raw samples here")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    try:
        status = _get(f"{base}/api/detection/status")
    except Exception as exc:
        print(f"cannot reach {base}: {exc}")
        print("Start the node first.")
        return 2
    if not status.get("worker"):
        print("The node is up but no detection worker is running, so there is")
        print("no frame rate to measure. Start the camera, then re-run.")
        return 2
    print(f"backend : {status.get('backend')}")
    print(f"mode    : {args.mode}")

    sampler = Sampler(base, args.interval)
    sampler.start()

    print(f"\nbaseline: sampling {args.baseline:.0f}s -- keep the scene steady")
    time.sleep(args.baseline)

    sampler.phase = "training"
    started = time.time()
    outcome: dict = {}
    try:
        if args.mode == "local":
            outcome = _post(f"{base}/api/federated/head/train",
                            {"epochs": args.epochs,
                             "min_samples": args.min_samples})
        else:
            body = {"epochs": args.epochs, "min_samples": args.min_samples}
            if args.server:
                body["server"] = args.server
            kickoff = _post(f"{base}/api/federated/round", body)
            print(f"  round started in {kickoff.get('run_dir')}")
            # The round returns at once and runs as a subprocess; the
            # training phase is only honest if we wait for that to exit.
            while True:
                time.sleep(args.interval)
                st = _get(f"{base}/api/federated/round")
                if not st.get("running"):
                    outcome = st.get("result") or {"reason": "no result"}
                    break
    except urllib.error.HTTPError as exc:
        sampler.stop()
        detail = exc.read().decode("utf-8", "replace")[:300]
        print(f"\ntraining request failed: HTTP {exc.code} {detail}")
        if exc.code == 409:
            print("Not enough labelled samples, or a round is already running.")
        return 1
    train_secs = time.time() - started

    sampler.phase = "recovery"
    print(f"training finished in {train_secs:.1f}s; "
          f"sampling {args.baseline:.0f}s of recovery")
    time.sleep(args.baseline)
    sampler.stop()
    sampler.join(timeout=5)

    print(f"\ntraining ran {train_secs:.1f}s "
          f"(accepted={outcome.get('accepted')})")
    if outcome.get("reason"):
        print(f"gate: {outcome['reason']}")

    print()
    for field in FIELDS:
        print(f"{field}")
        for phase in ("baseline", "training", "recovery"):
            print(_row(phase, sampler.of(phase, field)))

        bl, tr = sampler.of("baseline", field), sampler.of("training", field)
        if len(tr) < MIN_TRAINING_SAMPLES:
            print(f"  {'change':12s} not measurable -- only {len(tr)} sample(s) "
                  f"inside a {train_secs:.1f}s training window")
        elif bl:
            b, t = statistics.median(bl), statistics.median(tr)
            if b:
                change = (t - b) / b * 100.0
                worse = change < 0 if field.endswith("fps") else change > 0
                # Noise guard: the baseline's own spread is the floor on what
                # a single run can resolve.
                spread = (max(bl) - min(bl)) / b * 100.0
                note = "(worse)" if worse else "(no cost)"
                if abs(change) <= spread:
                    note = f"(inside the baseline's own {spread:.1f}% spread)"
                print(f"  {'change':12s} {change:+.1f}% {note}")
        print()

    if train_secs < args.interval * MIN_TRAINING_SAMPLES:
        print(f"Training finished in {train_secs:.1f}s -- too fast to sample at "
              f"{args.interval}s intervals.")
        print(f"Raise --epochs until it runs past "
              f"{args.interval * MIN_TRAINING_SAMPLES:.0f}s, lower --interval, "
              f"or use --mode round, which spawns a process and takes longer.")
        print()

    if sampler.errors:
        print(f"note: {sampler.errors} status polls failed; "
              f"a starved API skews this too, so treat it as part of the cost")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "train_seconds": train_secs,
                       "outcome": outcome, "samples": sampler.samples}, f, indent=2)
        print(f"raw samples -> {args.json}")

    print("\nQuote capture_fps for the verdict and detect_ms for the squeeze.")
    print("detect_fps moves with the scene, not the training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
