#!/usr/bin/env python3
"""
Verification accuracy for the face recognition stage.

Reports the number an evaluator actually asks for -- "how often is it
right?" -- rather than a raw similarity score, and reports it the way the
face-recognition literature does: as verification accuracy over genuine
and impostor pairs.

    genuine pair   two images of the same person   -> should MATCH
    impostor pair  two images of different people  -> should REJECT

Accuracy is the share of pairs decided correctly at a given threshold.
Also reported: TAR@FAR, the true-accept rate at a fixed false-accept
rate, which is the metric that matters when a false accept is the
expensive error -- and for a security system it is.

Faces are sourced from a labelled dataset organised one directory per
person, or from a detection dataset where each image yields faces.

Usage:
    python scripts/benchmark_face_accuracy.py --faces <dir-of-person-dirs>
    python scripts/benchmark_face_accuracy.py --images <dir> --max-images 200
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import settings  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def load_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def embeddings_by_person(app, root: Path, max_per_person: int, max_identities: int):
    """
    {person: [embedding, ...]} from one directory per person.

    Only identities with two or more usable images are kept -- one image
    cannot form a genuine pair, and genuine pairs are what make
    verification accuracy meaningful.
    """
    people = {}
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    # Prefer identities with more images: they yield more genuine pairs
    # per unit of inference, which matters on a large set like LFW.
    candidates.sort(key=lambda d: -len(list(d.glob("*.jpg"))))
    for person_dir in candidates:
        if max_identities and len(people) >= max_identities:
            break
        embs = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            embs.append(largest.normed_embedding)
            if len(embs) >= max_per_person:
                break
        if len(embs) >= 2:
            people[person_dir.name] = embs
    return people


def embeddings_from_images(app, root: Path, max_images: int):
    """
    Treat every detected face as its own identity.

    Only impostor pairs can be formed this way, since we have no labels
    saying two faces are the same person. That still measures the thing
    that matters most here -- the false-accept rate -- and it is what an
    unlabelled detection dataset can honestly support.
    """
    embs = []
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    for i, img_path in enumerate(files[:max_images], 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for f in app.get(img):
            embs.append(f.normed_embedding)
        if i % 25 == 0:
            print(f"\r  {i}/{min(len(files), max_images)}", end="", flush=True)
    print(f"\r  {min(len(files), max_images)} images scanned")
    return embs


def main() -> int:
    ap = argparse.ArgumentParser(description="Face verification accuracy.")
    ap.add_argument("--faces", type=Path,
                    help="Directory of per-person directories (genuine + impostor).")
    ap.add_argument("--images", type=Path,
                    help="Flat image directory (impostor / false-accept only).")
    ap.add_argument("--max-per-person", type=int, default=6)
    ap.add_argument("--max-identities", type=int, default=100,
                    help="Cap identities. LFW has 1680 usable; all of them "
                         "would take hours of CPU inference.")
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--max-pairs", type=int, default=200000)
    args = ap.parse_args()

    if not args.faces and not args.images:
        ap.error("pass --faces or --images")

    app = load_app()
    genuine, impostor = [], []

    if args.faces:
        people = embeddings_by_person(app, args.faces, args.max_per_person,
                                      args.max_identities)
        print(f"identities with 2+ usable images: {len(people)}")
        if not people:
            print("[FAIL] no identity had two usable face images")
            return 1

        for embs in people.values():
            for a, b in itertools.combinations(embs, 2):
                genuine.append(float(np.dot(a, b)))

        names = list(people)
        for n1, n2 in itertools.combinations(names, 2):
            for a in people[n1]:
                for b in people[n2]:
                    impostor.append(float(np.dot(a, b)))
                    if len(impostor) >= args.max_pairs:
                        break

    if args.images:
        embs = embeddings_from_images(app, args.images, args.max_images)
        print(f"faces found: {len(embs)}")
        for a, b in itertools.combinations(embs, 2):
            impostor.append(float(np.dot(a, b)))
            if len(impostor) >= args.max_pairs:
                break

    print(f"\ngenuine pairs : {len(genuine)}")
    print(f"impostor pairs: {len(impostor)}")

    if genuine:
        print(f"\ngenuine  mean {np.mean(genuine):.4f}  min {np.min(genuine):.4f}")
    if impostor:
        print(f"impostor mean {np.mean(impostor):.4f}  max {np.max(impostor):.4f}")
    if genuine and impostor:
        print(f"separation (means): {np.mean(genuine) - np.mean(impostor):.4f}")

    hdr = f"{'thresh':>7}{'accuracy':>10}{'TAR':>9}{'FAR':>9}{'FRR':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))

    def _fmt(value):
        return f"{value:>9.4f}" if value == value else f"{'n/a':>9}"

    best = None
    for t in THRESHOLDS:
        ta = sum(1 for s in genuine if s >= t)          # true accepts
        fr = len(genuine) - ta                          # false rejects
        fa = sum(1 for s in impostor if s >= t)         # false accepts
        tr = len(impostor) - fa                         # true rejects
        total = len(genuine) + len(impostor) or 1
        acc = (ta + tr) / total
        tar = ta / len(genuine) if genuine else float("nan")
        far = fa / len(impostor) if impostor else float("nan")
        frr = fr / len(genuine) if genuine else float("nan")

        if genuine and impostor and (best is None or acc > best[1]):
            best = (t, acc, tar, far)

        acc_s = f"{acc:>10.4f}" if (genuine and impostor) else f"{'n/a':>10}"
        print(f"{t:>7.2f}{acc_s}{_fmt(tar)}{_fmt(far)}{_fmt(frr)}")

    print("\n  TAR = true accept rate (same person correctly matched)")
    print("  FAR = false accept rate (different person wrongly matched)")
    print("  FRR = false reject rate (same person wrongly rejected)")

    configured = settings.FACE_SIMILARITY_THRESHOLD
    if impostor:
        fa_at_cfg = sum(1 for s in impostor if s >= configured) / len(impostor)
        print(f"\nAt the configured threshold {configured}:")
        print(f"  false accept rate: {fa_at_cfg:.4f}"
              f"  ({sum(1 for s in impostor if s >= configured)} of {len(impostor)})")
        if genuine:
            tar_cfg = sum(1 for s in genuine if s >= configured) / len(genuine)
            print(f"  true accept rate : {tar_cfg:.4f}")

    if best:
        print(f"\nBest accuracy {best[1]:.4f} at threshold {best[0]:.2f} "
              f"(TAR {best[2]:.4f}, FAR {best[3]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
