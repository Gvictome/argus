"""
Byzantine-robust aggregation.

These are adversarial by design: each test plays an attacker and asserts
the defence holds. A screening layer nobody tried to break is not a
defence, it is a comment.

The baseline being defended against is plain FedAvg, which averages
whatever it is given -- so every "attack" here would succeed outright
without this module.
"""

import numpy as np
import pytest

from src.federated.robust import (
    ClientUpdate,
    RobustAggregator,
    validate_candidate,
)


def _global(n=50):
    rng = np.random.default_rng(0)
    return [rng.normal(0, 0.1, (n,)), rng.normal(0, 0.1, (n, 2))]


def _honest(global_weights, node_id, scale=0.01, seed=1, samples=100):
    """A node that learned a small amount in a consistent direction."""
    rng = np.random.default_rng(seed)
    # A shared signal plus per-node noise: honest nodes broadly agree.
    return ClientUpdate(
        node_id=node_id,
        weights=[w + scale * (np.ones_like(w) + 0.1 * rng.normal(size=w.shape))
                 for w in global_weights],
        num_samples=samples,
    )


def _fleet(global_weights, n=4, scale=0.01):
    return [_honest(global_weights, f"node{i}", scale=scale, seed=i) for i in range(n)]


class TestStructuralDefences:
    """Failures that need no comparison with peers."""

    def test_nan_update_is_rejected(self):
        """
        The most destructive single failure: averaged in, one NaN makes
        every coordinate NaN and the global model is unrecoverable.
        """
        g = _global()
        updates = _fleet(g, 3)
        bad = _honest(g, "diverged")
        bad.weights[0][0] = np.nan
        updates.append(bad)

        new, report = RobustAggregator().aggregate(g, updates)

        assert "diverged" in report.rejected
        assert all(np.all(np.isfinite(w)) for w in new)

    def test_inf_update_is_rejected(self):
        g = _global()
        updates = _fleet(g, 3)
        bad = _honest(g, "exploded")
        bad.weights[0][0] = np.inf
        updates.append(bad)

        new, report = RobustAggregator().aggregate(g, updates)

        assert "exploded" in report.rejected
        assert all(np.all(np.isfinite(w)) for w in new)

    def test_wrong_architecture_is_rejected(self):
        """A node running a different model would otherwise crash aggregation."""
        g = _global()
        updates = _fleet(g, 3)
        updates.append(ClientUpdate("wrong-arch", [np.zeros((7,)), np.zeros((7, 2))], 100))

        _, report = RobustAggregator().aggregate(g, updates)

        assert "wrong-arch" in report.rejected

    def test_zero_sample_claim_is_rejected(self):
        g = _global()
        updates = _fleet(g, 3)
        liar = _honest(g, "no-data")
        liar.num_samples = 0
        updates.append(liar)

        _, report = RobustAggregator().aggregate(g, updates)

        assert "no-data" in report.rejected


class TestNormDefence:
    def test_scaled_update_is_rejected(self):
        """
        Model-replacement attack: scale an update enormously so that
        averaging it in effectively replaces the global model.
        """
        g = _global()
        updates = _fleet(g, 4)
        attacker = ClientUpdate(
            "attacker",
            [w + 1000.0 * np.ones_like(w) for w in g],
            num_samples=100,
        )
        updates.append(attacker)

        new, report = RobustAggregator().aggregate(g, updates)

        assert "attacker" in report.rejected
        # And the model barely moved, as an honest round should.
        assert np.abs(new[0] - g[0]).max() < 1.0

    def test_honest_variation_is_not_rejected(self):
        """The filter must not eat legitimate disagreement."""
        g = _global()
        updates = _fleet(g, 5)
        # One node simply trained a bit more than its peers.
        updates.append(_honest(g, "eager", scale=0.02, seed=99))

        _, report = RobustAggregator().aggregate(g, updates)

        assert "eager" in report.accepted


class TestDirectionDefence:
    def test_sign_flip_is_rejected(self):
        """
        A node returning the negative of what it learned drags the model
        backwards every round while looking perfectly normal by size.
        """
        g = _global()
        updates = _fleet(g, 4)
        flipped = ClientUpdate(
            "sign-flipper",
            [w - 0.01 * np.ones_like(w) for w in g],   # opposite direction
            num_samples=100,
        )
        updates.append(flipped)

        _, report = RobustAggregator().aggregate(g, updates)

        assert "sign-flipper" in report.rejected
        assert report.consensus_applied

    def test_sign_flip_survives_plain_averaging(self):
        """
        The counterfactual: show the attack works without screening, so
        the test above is proving something.
        """
        g = _global()
        honest = _fleet(g, 4)
        flipped = [w - 0.01 * np.ones_like(w) for w in g]

        naive = [
            np.mean([u.weights[i] for u in honest] + [flipped[i]], axis=0)
            for i in range(len(g))
        ]
        screened, _ = RobustAggregator().aggregate(g, honest + [
            ClientUpdate("sign-flipper", flipped, 100)
        ])

        # The naive average is dragged further from the honest consensus.
        honest_only, _ = RobustAggregator().aggregate(g, honest)
        assert np.abs(naive[0] - honest_only[0]).mean() > \
               np.abs(screened[0] - honest_only[0]).mean()


class TestConsensusFloor:
    """
    Consensus filtering needs a majority. Claiming otherwise would be the
    dishonest part of a security story.
    """

    def test_two_nodes_skip_consensus_and_say_so(self):
        g = _global()
        updates = _fleet(g, 2)

        new, report = RobustAggregator().aggregate(g, updates)

        assert new is not None
        assert report.consensus_applied is False
        assert "consensus" in report.note.lower()

    def test_three_nodes_enable_consensus(self):
        g = _global()

        _, report = RobustAggregator().aggregate(g, _fleet(g, 3))

        assert report.consensus_applied is True

    def test_structural_checks_still_apply_below_the_floor(self):
        """NaN rejection is absolute and must work even with 2 nodes."""
        g = _global()
        bad = _honest(g, "nan-node")
        bad.weights[0][0] = np.nan

        new, report = RobustAggregator().aggregate(g, [_honest(g, "good"), bad])

        assert "nan-node" in report.rejected
        assert all(np.all(np.isfinite(w)) for w in new)


class TestAggregationOutcome:
    def test_no_updates_returns_none(self):
        """Caller must keep the current model, not promote nothing."""
        new, report = RobustAggregator().aggregate(_global(), [])

        assert new is None
        assert "no updates" in report.note

    def test_all_rejected_returns_none(self):
        g = _global()
        bad = [ClientUpdate(f"bad{i}", [np.zeros((7,)), np.zeros((7, 2))], 10)
               for i in range(3)]

        new, _ = RobustAggregator().aggregate(g, bad)

        assert new is None, "aggregator promoted a model built from nothing"

    def test_output_shapes_match_the_global_model(self):
        g = _global()

        new, _ = RobustAggregator().aggregate(g, _fleet(g, 4))

        assert [w.shape for w in new] == [w.shape for w in g]

    def test_honest_round_moves_the_model(self):
        """Screening must not be so strict that learning stops."""
        g = _global()

        new, report = RobustAggregator().aggregate(g, _fleet(g, 4))

        assert report.rejected == []
        assert not np.allclose(new[0], g[0]), "no learning was applied"

    def test_report_is_serializable(self):
        """It goes to the audit log and the dashboard."""
        g = _global()
        _, report = RobustAggregator().aggregate(g, _fleet(g, 3))
        d = report.as_dict()

        import json
        json.loads(json.dumps(d))
        assert d["accepted_count"] == 3


class TestValidationGate:
    """The last defence, and the only one that measures model quality."""

    def test_improvement_is_promoted(self):
        promote, _ = validate_candidate(0.85, 0.80)
        assert promote

    def test_large_regression_is_rejected(self):
        promote, reason = validate_candidate(0.60, 0.80)

        assert not promote
        assert "worse" in reason

    def test_small_regression_is_tolerated(self):
        """Federated rounds are noisy; demanding monotonic gains stalls."""
        promote, _ = validate_candidate(0.79, 0.80, tolerance=0.02)
        assert promote

    def test_nan_metric_is_rejected(self):
        promote, reason = validate_candidate(float("nan"), 0.80)

        assert not promote
        assert "finite" in reason

    def test_loss_direction_is_inverted(self):
        assert validate_candidate(0.20, 0.30, higher_is_better=False)[0]
        assert not validate_candidate(0.90, 0.30, higher_is_better=False)[0]
