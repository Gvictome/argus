"""
Byzantine-robust aggregation for ARGUS federated learning.

Plain FedAvg averages every update it receives. One node that is broken,
diverged, or hostile is therefore enough to move the global model, and
the result is pushed back out to every other node -- so a single bad
contributor degrades the whole fleet, silently, and the next round starts
from the damaged model.

This screens updates before averaging. Four filters, cheapest first, each
defending a different failure:

    1. Structural    wrong shapes, NaN, Inf        broken or mismatched node
    2. Norm          update far larger than peers  scaled / exploding update
    3. Direction     points away from consensus    sign-flip, label-flip,
                                                   targeted poisoning
    4. Trimmed mean  drop coordinate extremes      residual outliers

Then a validation gate (see `validate_candidate`) refuses to promote a
candidate that is worse than the model it would replace.

    HONEST LIMIT: consensus filtering needs a majority to be honest, so
    it needs at least 3 contributors to mean anything. With 2 nodes, if
    the two disagree there is no way to tell which one is wrong. Below
    `MIN_CLIENTS_FOR_CONSENSUS` this module says so, skips filters 3-4,
    and leans on the structural, norm, and validation checks -- which are
    absolute rather than relative and still work with one contributor.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Consensus is majority-based, so it is meaningless without a majority.
MIN_CLIENTS_FOR_CONSENSUS = 3

# Reject an update whose L2 norm exceeds this multiple of the median
# norm. Honest nodes training on similar data produce similar-magnitude
# updates; a large multiple is the signature of a scaled ("model
# replacement") attack or a diverged optimizer.
DEFAULT_NORM_FACTOR = 3.0

# Reject an update whose direction disagrees with the consensus
# direction. 0.0 admits anything not actively opposed; negative values
# mean actively pulling the model the other way.
DEFAULT_MIN_COSINE = 0.0

# Fraction trimmed from each end, per coordinate, before averaging.
DEFAULT_TRIM_RATIO = 0.1


@dataclass
class ClientUpdate:
    """One node's contribution to a round."""
    node_id: str
    weights: List[np.ndarray]
    num_samples: int = 1
    metrics: Dict = field(default_factory=dict)


@dataclass
class Screening:
    """Why one update was kept or dropped."""
    node_id: str
    accepted: bool
    reason: str
    norm: float = 0.0
    cosine: float = float("nan")

    def __str__(self) -> str:
        verdict = "accept" if self.accepted else "REJECT"
        return f"{self.node_id}: {verdict} ({self.reason})"


@dataclass
class AggregationReport:
    """What the aggregator did, for the audit log and the dashboard."""
    accepted: List[str]
    rejected: List[str]
    screenings: List[Screening]
    consensus_applied: bool
    median_norm: float
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "consensus_applied": self.consensus_applied,
            "median_norm": round(self.median_norm, 6),
            "note": self.note,
            "detail": [
                {
                    "node_id": s.node_id,
                    "accepted": s.accepted,
                    "reason": s.reason,
                    "norm": round(s.norm, 6),
                    "cosine": None if np.isnan(s.cosine) else round(s.cosine, 4),
                }
                for s in self.screenings
            ],
        }


def _flatten(arrays: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def _shapes(arrays: Sequence[np.ndarray]) -> List[tuple]:
    return [np.asarray(a).shape for a in arrays]


def _is_finite(arrays: Sequence[np.ndarray]) -> bool:
    return all(np.all(np.isfinite(np.asarray(a))) for a in arrays)


class RobustAggregator:
    """
    Screen client updates, then average the survivors.

    Operates on *deltas* (client weights minus the current global
    weights), not raw weights. A delta is what the client actually
    learned; raw weights are dominated by the shared starting point, and
    two updates that disagree completely still look ~identical when
    compared as raw weights. Comparing deltas is what makes the direction
    filter able to see disagreement at all.
    """

    def __init__(
        self,
        norm_factor: float = DEFAULT_NORM_FACTOR,
        min_cosine: float = DEFAULT_MIN_COSINE,
        trim_ratio: float = DEFAULT_TRIM_RATIO,
        min_clients_for_consensus: int = MIN_CLIENTS_FOR_CONSENSUS,
    ):
        self.norm_factor = norm_factor
        self.min_cosine = min_cosine
        self.trim_ratio = trim_ratio
        self.min_clients_for_consensus = min_clients_for_consensus

    # -- filter 1: structural -------------------------------------------

    def _structural_check(
        self, update: ClientUpdate, reference_shapes: List[tuple]
    ) -> Optional[str]:
        """Reject reasons that need no comparison with peers."""
        if not update.weights:
            return "empty update"
        if _shapes(update.weights) != reference_shapes:
            return "shape mismatch (different model architecture)"
        if not _is_finite(update.weights):
            # A diverged client sends NaN/Inf. Averaged in, it makes every
            # coordinate NaN and destroys the global model irrecoverably.
            return "non-finite values (NaN or Inf)"
        if update.num_samples <= 0:
            return "claims zero training samples"
        return None

    # -- aggregation -----------------------------------------------------

    def aggregate(
        self,
        global_weights: List[np.ndarray],
        updates: List[ClientUpdate],
    ) -> Tuple[Optional[List[np.ndarray]], AggregationReport]:
        """
        Produce a new global model from screened updates.

        Returns:
            (new_weights, report). new_weights is None when nothing
            survived screening -- in which case the caller must keep the
            current global model rather than promoting anything.
        """
        screenings: List[Screening] = []

        if not updates:
            return None, AggregationReport([], [], [], False, 0.0, "no updates received")

        reference_shapes = _shapes(global_weights)

        # --- filter 1: structural ---
        survivors: List[ClientUpdate] = []
        for u in updates:
            problem = self._structural_check(u, reference_shapes)
            if problem:
                screenings.append(Screening(u.node_id, False, problem))
                logger.warning("Rejecting %s: %s", u.node_id, problem)
            else:
                survivors.append(u)

        if not survivors:
            return None, AggregationReport(
                [], [u.node_id for u in updates], screenings, False, 0.0,
                "every update failed structural checks",
            )

        flat_global = _flatten(global_weights)
        deltas = {u.node_id: _flatten(u.weights) - flat_global for u in survivors}
        norms = {nid: float(np.linalg.norm(d)) for nid, d in deltas.items()}
        median_norm = float(np.median(list(norms.values())))

        # --- filter 2: norm ---
        # Absolute-ish: works even with one contributor, since it compares
        # against the median of what was actually received.
        kept: List[ClientUpdate] = []
        for u in survivors:
            n = norms[u.node_id]
            if median_norm > 0 and n > self.norm_factor * median_norm:
                reason = (f"update norm {n:.4g} exceeds {self.norm_factor}x "
                          f"median {median_norm:.4g}")
                screenings.append(Screening(u.node_id, False, reason, n))
                logger.warning("Rejecting %s: %s", u.node_id, reason)
            else:
                kept.append(u)

        if not kept:
            return None, AggregationReport(
                [], [u.node_id for u in updates], screenings, False, median_norm,
                "every update failed the norm check",
            )

        # --- filters 3 and 4 need a majority to be meaningful ---
        consensus = len(kept) >= self.min_clients_for_consensus
        note = ""
        if not consensus:
            note = (
                f"only {len(kept)} contributor(s); consensus filtering needs "
                f"{self.min_clients_for_consensus}. Direction and trimming "
                f"skipped -- with a minority you cannot tell which side is wrong."
            )
            logger.info("Robust aggregation: %s", note)
            for u in kept:
                screenings.append(Screening(
                    u.node_id, True, "accepted (no consensus check available)",
                    norms[u.node_id],
                ))
            new_weights = self._weighted_average(global_weights, kept, deltas)
            return new_weights, AggregationReport(
                [u.node_id for u in kept],
                [s.node_id for s in screenings if not s.accepted],
                screenings, False, median_norm, note,
            )

        # --- filter 3: direction ---
        # The coordinate-wise median delta is the consensus direction. The
        # median is used rather than the mean precisely because the mean
        # is what an attacker would be trying to drag.
        stacked = np.stack([deltas[u.node_id] for u in kept])
        reference = np.median(stacked, axis=0)
        ref_norm = float(np.linalg.norm(reference))

        aligned: List[ClientUpdate] = []
        for u in kept:
            d = deltas[u.node_id]
            dn = float(np.linalg.norm(d))
            cos = float(d @ reference / (dn * ref_norm)) if dn > 0 and ref_norm > 0 else 0.0

            if cos < self.min_cosine:
                reason = f"direction disagrees with consensus (cos={cos:.3f})"
                screenings.append(Screening(u.node_id, False, reason, dn, cos))
                logger.warning("Rejecting %s: %s", u.node_id, reason)
            else:
                aligned.append(u)
                screenings.append(Screening(u.node_id, True, "accepted", dn, cos))

        if not aligned:
            return None, AggregationReport(
                [], [u.node_id for u in updates], screenings, True, median_norm,
                "every update disagreed with the consensus direction",
            )

        # --- filter 4: coordinate-wise trimmed mean ---
        new_weights = self._trimmed_mean(global_weights, aligned, deltas)

        return new_weights, AggregationReport(
            [u.node_id for u in aligned],
            [s.node_id for s in screenings if not s.accepted],
            screenings, True, median_norm,
            f"trimmed mean over {len(aligned)} updates",
        )

    # -- combiners -------------------------------------------------------

    def _weighted_average(
        self,
        global_weights: List[np.ndarray],
        updates: List[ClientUpdate],
        deltas: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """Sample-weighted FedAvg over deltas. Used when consensus is unavailable."""
        total = sum(u.num_samples for u in updates) or 1
        flat = sum(deltas[u.node_id] * (u.num_samples / total) for u in updates)
        return self._unflatten(global_weights, _flatten(global_weights) + flat)

    def _trimmed_mean(
        self,
        global_weights: List[np.ndarray],
        updates: List[ClientUpdate],
        deltas: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """
        Average each coordinate after dropping its extremes.

        Per-coordinate rather than per-update: a poisoned update need not
        be an outlier overall, only on the coordinates it is targeting.
        """
        stacked = np.stack([deltas[u.node_id] for u in updates])
        n = stacked.shape[0]
        k = int(np.floor(self.trim_ratio * n))

        if k == 0 or n - 2 * k < 1:
            combined = stacked.mean(axis=0)
        else:
            ordered = np.sort(stacked, axis=0)
            combined = ordered[k:n - k].mean(axis=0)

        return self._unflatten(global_weights, _flatten(global_weights) + combined)

    @staticmethod
    def _unflatten(template: List[np.ndarray], flat: np.ndarray) -> List[np.ndarray]:
        out, i = [], 0
        for a in template:
            arr = np.asarray(a)
            size = arr.size
            out.append(flat[i:i + size].reshape(arr.shape).astype(arr.dtype))
            i += size
        return out


def validate_candidate(
    candidate_metric: float,
    current_metric: float,
    tolerance: float = 0.02,
    higher_is_better: bool = True,
) -> Tuple[bool, str]:
    """
    Should this candidate replace the current global model?

    The last line of defence, and the only one that measures the thing we
    actually care about. Every filter above reasons about the *shape* of
    an update; this asks whether the resulting model is any good.

    A small tolerance is allowed because federated rounds are noisy and
    demanding monotonic improvement would stall the model permanently.

    Args:
        candidate_metric: The candidate's score on the held-out set.
        current_metric: The incumbent's score on the same set.
        tolerance: How much regression to forgive.
        higher_is_better: True for accuracy/mAP, False for loss.

    Returns:
        (promote, reason)
    """
    if not np.isfinite(candidate_metric):
        return False, f"candidate metric is not finite ({candidate_metric})"

    if higher_is_better:
        delta = candidate_metric - current_metric
        if delta >= -tolerance:
            return True, f"promoted ({candidate_metric:.4f} vs {current_metric:.4f}, Δ{delta:+.4f})"
        return False, (f"rejected: {candidate_metric:.4f} is worse than "
                       f"{current_metric:.4f} by {-delta:.4f} (tolerance {tolerance})")

    delta = current_metric - candidate_metric
    if delta >= -tolerance:
        return True, f"promoted (loss {candidate_metric:.4f} vs {current_metric:.4f}, Δ{-delta:+.4f})"
    return False, (f"rejected: loss {candidate_metric:.4f} is worse than "
                   f"{current_metric:.4f} by {-delta:.4f} (tolerance {tolerance})")
