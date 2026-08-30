#!/usr/bin/env python3
"""
Fetch the threat-detection weights into models/.

The weights are not committed to git. They come from the upstream thesis
release, and this script downloads them, verifies them against the
checksums that release publishes, and puts them where the config expects.

Verification is the point. A truncated download produces a .pt that
either fails to load at startup or -- worse -- loads and behaves oddly,
and neither failure names its own cause.

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py --variant yolo11s
    python scripts/fetch_models.py --list

Exit codes:
    0  weights present and verified
    1  download or verification failed
    2  bad usage (unknown variant)
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "Nambekai/dangerous-person-detection-yolo11"
RELEASE = "v1.0.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE}"

# SHA-256 of each release zip, copied from the release's own SHA256SUMS.txt.
# Recorded here so verification does not depend on fetching the checksum
# file over the same connection that could have corrupted the download.
VARIANTS = {
    "yolo11n": {
        "asset": "weights-YOLO11n.zip",
        "sha256": "69b1ae121e457b935cd6c3cd4c9bbfd7d8b93d790e6caf923c216a62dedb64c0",
        "note": "F1 0.74, mAP50 0.78, 2.6M params — the Pi default",
    },
    "yolo11s": {
        "asset": "weights-YOLO11s.zip",
        "sha256": "b89b090d8d26a1024486b81f3b10acb832c23bee699e1304c0315713ce118983",
        "note": "F1 0.77, mAP50 0.80 — best accuracy, ~3x the compute",
    },
    "yolo11s_preprocessing": {
        "asset": "weights-YOLO11s-preprocessing.zip",
        "sha256": "324c314bae3bfa3ba57ff46e506861d939caa25b27c7c9f92d3e1e7ab2578545",
        "note": "F1 0.74 — augmentation variant",
    },
    "yolo11m": {
        "asset": "weights-YOLO11m.zip",
        "sha256": "a6d7cf272d13d6332048e7e573cfa0273283323b8076a512fb94865c0decb09f",
        "note": "F1 0.73 — larger, scored worse than s",
    },
    "yolo11l": {
        "asset": "weights-YOLO11l.zip",
        "sha256": "0984d7cffc48fef8bc826a75aa5e3120b8fb3d556f4176a3e3f708855bd6e414",
        "note": "F1 0.76, best recall 0.72 — largest",
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _say(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> bool:
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            read = 0
            with dest.open("wb") as fh:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
                    if total:
                        pct = read * 100 // total
                        print(f"\r  {pct:3d}%  {read // 1024:,} KB", end="", flush=True)
            print()
        return True
    except Exception as exc:
        print()
        _say("FAIL", f"Download failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch threat-detection weights.")
    parser.add_argument("--variant", default="yolo11n", help="Which trained variant to fetch.")
    parser.add_argument("--list", action="store_true", help="List variants and exit.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    if args.list:
        print(f"Variants available from {REPO} {RELEASE}:\n")
        for name, meta in VARIANTS.items():
            print(f"  {name:24s} {meta['note']}")
        return 0

    if args.variant not in VARIANTS:
        _say("FAIL", f"Unknown variant {args.variant!r}. Use --list to see options.")
        return 2

    meta = VARIANTS[args.variant]
    target = MODELS_DIR / f"threat-{args.variant}.pt"

    if target.exists() and not args.force:
        _say("PASS", f"Already present: {target}")
        print("       Re-fetch with --force.")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / meta["asset"]

        if not _download(f"{BASE_URL}/{meta['asset']}", archive):
            return 1

        actual = _sha256(archive)
        if actual != meta["sha256"]:
            _say("FAIL", "Checksum mismatch — discarding download.")
            print(f"       expected {meta['sha256']}")
            print(f"       actual   {actual}")
            return 1
        _say("PASS", f"SHA-256 verified ({actual[:16]}...)")

        try:
            with zipfile.ZipFile(archive) as zf:
                # The archives contain weights/best.pt and weights/last.pt.
                # best.pt is the early-stopped checkpoint we want.
                member = next(
                    (n for n in zf.namelist() if n.endswith("best.pt")), None
                )
                if member is None:
                    _say("FAIL", f"No best.pt inside {meta['asset']}")
                    return 1
                zf.extract(member, tmpdir)
                shutil.move(str(tmpdir / member), str(target))
        except Exception as exc:
            _say("FAIL", f"Extraction failed: {exc}")
            return 1

    size_mb = target.stat().st_size / (1024 * 1024)
    _say("PASS", f"Wrote {target} ({size_mb:.1f} MB)")

    default = MODELS_DIR / "threat-yolo11n.pt"
    if target != default:
        print(f"\nNote: config defaults to {default.name}.")
        print(f"      Point THREAT_MODEL_PATH at {target.name} to use this variant.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
