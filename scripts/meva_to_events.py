"""
Convert one MEVA camera's KPF tracks into an ARGUS SampleStore.

    python scripts/meva_to_events.py \
        --annotations data/datasets/meva/repo/annotation/DIVA-phase-2/MEVA \
        --camera G336

Writes data/datasets/meva/events/<camera>/samples.jsonl (19-feature events
built with src/federated/features.py, labelled routine_* or anomaly) and
holdout.json (which whole clips are held out). The label rule, class mapping
and event-building rules are documented in src/federated/meva.py and in
docs/research/datasets.md. Works for any camera ID with no code changes.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated import meva  # noqa: E402

DEFAULT_OUT_ROOT = Path("data/datasets/meva/events")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--annotations", required=True, type=Path,
                   help="directory holding the MEVA *.geom.yml/.types.yml/.activities.yml files")
    p.add_argument("--camera", required=True, help="camera ID, e.g. G336")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output dir (default {DEFAULT_OUT_ROOT}/<camera>)")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-clips", type=int, default=None)
    p.add_argument("--det-fps", type=float, default=meva.DEFAULT_DET_FPS,
                   help="thin the 30 fps tracks to this detector-like rate (default %(default)s)")
    a = p.parse_args(argv)

    out = a.out or DEFAULT_OUT_ROOT / a.camera
    t0 = time.perf_counter()
    try:
        r = meva.convert_camera(a.annotations, a.camera, out, a.holdout_frac,
                                a.seed, a.limit_clips, a.det_fps)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0

    print(f"camera {r.camera}: {r.clips} clips, {r.tracks} tracks -> {r.events} events "
          f"in {dt:.1f}s")
    print(f"  dropped: short={r.stats.dropped_short} untyped={r.stats.untyped} "
          f"unmapped_class={r.stats.dropped_class}")
    print("  class mix: " + "  ".join(
        f"{k}={v} ({v / max(r.events, 1):.1%})" for k, v in sorted(r.class_mix.items())))
    print(f"  anomaly rate: {r.anomaly_count}/{r.events} = {r.anomaly_rate:.1%} "
          f"by reason {r.reason_counts}")
    if not 0.02 <= r.anomaly_rate <= 0.10:
        print("  WARNING: anomaly rate outside the 2-10% target band")
    print(f"  active hours: {r.active_hours}")
    print(f"  split: train {r.train_clips} clips / {r.train_events} events, "
          f"holdout {r.holdout_clips} clips / {r.holdout_events} events (seed {a.seed})")
    print(f"  wrote {r.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
