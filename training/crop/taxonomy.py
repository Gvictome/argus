"""
Dataset class names -> ARGUS fine-taxonomy slots.

The slot tuples mirror sim/workload_fine.py (CARRIERS, MAKES,
PACKAGE_SIZES); tests/test_crop_training.py fails if they drift.
"""

from typing import Optional, Sequence, Tuple

ARGUS_CARRIERS = ("none", "ups", "usps", "fedex", "amazon")
ARGUS_MAKES = ("nissan", "toyota", "honda", "ford", "chevrolet", "tesla", "other")
ARGUS_SIZES = ("mail", "small", "large")

# Stanford Cars makes whose name is more than one word.
_MULTIWORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def make_of(class_name: str) -> str:
    """'Aston Martin V8 Vantage Coupe 2012' -> 'Aston Martin'."""
    for make in _MULTIWORD_MAKES:
        if class_name.startswith(make + " "):
            return make
    return class_name.split()[0]


def argus_make(make: str) -> str:
    """Full make name -> one of ARGUS_MAKES ('other' when not tracked)."""
    key = make.lower()
    return key if key in ARGUS_MAKES else "other"


def argus_carrier(name: str) -> Optional[str]:
    """Roboflow vehicle class -> carrier slot; None for non-vehicles (logos)."""
    key = name.lower()
    if "logo" in key:
        return None
    if key.startswith("other"):
        return "none"
    # usps before ups: 'usps' contains 'ups'.
    for carrier in ("usps", "ups", "fedex", "amazon"):
        if carrier in key:
            return carrier
    return None


def argus_size(name: str) -> Optional[str]:
    """Package class -> size slot; None for anything that is not a package."""
    key = name.lower()
    if any(w in key for w in ("envelope", "mailer", "letter", "mail")):
        return "mail"
    if "box" in key or "parcel" in key or "package" in key:
        return "large" if "large" in key or "big" in key else "small"
    return None


def crop_box(box: Tuple[float, float, float, float], img_w: int, img_h: int,
             pad: float = 0.1, min_side: int = 16) -> Optional[Tuple[int, int, int, int]]:
    """Pad an (x1, y1, x2, y2) box by `pad` of its size, clamp to the image.

    Returns None for boxes too small to classify. YOLO hands the crop
    classifier the same padded box at runtime, so training crops match.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < min_side or h < min_side:
        return None
    px, py = w * pad, h * pad
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(img_w, int(round(x2 + px))),
        min(img_h, int(round(y2 + py))),
    )


def make_accuracies(true: Sequence[int], pred: Sequence[int],
                    classes: Sequence[str]) -> Tuple[float, float]:
    """(accuracy over full makes, accuracy after collapsing to ARGUS slots)."""
    if len(true) != len(pred) or not true:
        raise ValueError("true and pred must be the same non-zero length")
    slot = [argus_make(c) for c in classes]
    n = len(true)
    exact = sum(t == p for t, p in zip(true, pred)) / n
    collapsed = sum(slot[t] == slot[p] for t, p in zip(true, pred)) / n
    return exact, collapsed
