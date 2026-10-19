"""
Poisoned-peer attacks for the hybrid demo.

A malicious or broken node returns weights that are not an honest training
step. The robust aggregator (src/federated/robust.py) is supposed to refuse
them; these modes exist so the demo can show it doing so live.

    none      honest update, passed through
    signflip  walk the opposite way from the honest update, boosted x5
              (rejected by the norm and direction filters)
    nan       non-finite weights (rejected by the structural check)

Pure functions: inputs are never mutated.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

MODES = ("none", "signflip", "nan")
SIGNFLIP_BOOST = 5.0


def poison_weights(mode: str, global_w: Sequence[np.ndarray],
                   trained_w: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return the weights a peer in `mode` would send back.

    `global_w` is the model the round started from, `trained_w` what an
    honest local step produced from it.
    """
    if mode not in MODES:
        raise ValueError(f"poison mode must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return [np.array(w, dtype=np.float32) for w in trained_w]
    if mode == "signflip":
        return [
            (g - SIGNFLIP_BOOST * (t - g)).astype(np.float32)
            for g, t in zip(global_w, trained_w)
        ]
    out = [np.array(w, dtype=np.float32) for w in trained_w]
    out[0] = out[0].copy()
    out[0].flat[0] = np.nan
    return out
