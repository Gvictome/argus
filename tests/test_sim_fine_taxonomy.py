"""
Fine-grained taxonomy for the node simulator: carrier, vehicle make and
package size on top of the coarse person / vehicle / animal / package class.

The coarse generator is what src/federated/features.py mirrors, so the first
thing pinned here is that adding the fine generator leaves it byte-identical.
"""

import hashlib
import os
import sys

import numpy as np
import pytest

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim")
sys.path.insert(0, SIM)

import workload  # noqa: E402
import workload_fine as wf  # noqa: E402
from head import FederatedHead  # noqa: E402

ANOMALY = 4
DWELL = 13  # coarse layout: 10-13 dynamics, dwell last

# sha256(X || y)[:16] of workload.generate(site, 3000, seed=7), recorded on
# numpy 2.2.1 before the fine taxonomy existed. A change here means the
# coarse generator drifted from what src/federated/features.py mirrors.
COARSE_FINGERPRINTS = {
    "driveway": "d43cb6b9054ce395",
    "backdoor": "4343d9aa88d288a8",
    "yard": "72563c5cab930c63",
}


@pytest.mark.parametrize("key", sorted(COARSE_FINGERPRINTS))
def test_coarse_generator_is_unchanged(key):
    X, y = workload.generate(workload.SITES[key], 3000, seed=7)
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:16]
    assert digest == COARSE_FINGERPRINTS[key]


def test_fine_vector_extends_the_coarse_layout():
    assert wf.N_FEATURES == workload.N_FEATURES + len(wf.CARRIERS) + len(wf.MAKES) \
        + len(wf.PACKAGE_SIZES) + 1
    X, y, meta = wf.generate(wf.SITES["driveway"], 2000, seed=3)
    assert X.shape == (2000, wf.N_FEATURES)
    assert X.dtype == np.float32
    assert y.shape == (2000,)
    assert set(np.unique(y)) <= set(range(workload.N_CLASSES))
    assert set(meta) >= {"cls", "hour", "carrier", "make", "size",
                         "carrier_pred", "make_pred", "size_pred"}


def test_generation_is_deterministic_per_seed():
    a = wf.generate(wf.SITES["curbside"], 500, seed=11)
    b = wf.generate(wf.SITES["curbside"], 500, seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_subtype_slots_only_fill_for_the_classes_they_describe():
    X, _, meta = wf.generate(wf.SITES["driveway"], 4000, seed=5)
    cls = meta["cls"]
    carrier = X[:, wf.CARRIER_SLICE]
    make = X[:, wf.MAKE_SLICE]
    size = X[:, wf.SIZE_SLICE]

    people_animals = (cls == 0) | (cls == 2)
    assert not carrier[people_animals].any()
    assert not make[cls != 1].any()
    assert not size[cls != 3].any()

    # Vehicles and packages carry exactly one carrier slot ("none" included).
    vp = (cls == 1) | (cls == 3)
    assert np.all(carrier[vp].sum(axis=1) == 1)
    assert np.all(size[cls == 3].sum(axis=1) == 1)


def test_regular_mail_only_comes_from_usps():
    _, _, meta = wf.generate(wf.SITES["curbside"], 6000, seed=2, classifier_acc=1.0)
    mail = (meta["cls"] == 3) & (meta["size"] == wf.PACKAGE_SIZES.index("mail"))
    assert mail.any()
    assert np.all(meta["carrier"][mail] == wf.CARRIERS.index("usps"))


def test_perfect_classifier_reports_the_true_subtype():
    _, _, meta = wf.generate(wf.SITES["driveway"], 3000, seed=4, classifier_acc=1.0)
    for k in ("carrier", "make", "size"):
        assert np.array_equal(meta[k], meta[f"{k}_pred"])


def test_classifier_error_rate_matches_the_requested_accuracy():
    _, _, meta = wf.generate(wf.SITES["driveway"], 20000, seed=8, classifier_acc=0.8)
    veh = meta["cls"] == 1
    hit = (meta["make"][veh] == meta["make_pred"][veh]).mean()
    assert 0.77 < hit < 0.83


def _routine_rate(y, mask, label):
    return float((y[mask] == label).mean())


def _makes(names):
    return [wf.MAKES.index(m) for m in names]


def test_household_vehicle_is_routine_at_night():
    site = wf.SITES["driveway"]
    _, y, meta = wf.generate(site, 40000, seed=9, classifier_acc=1.0, label_noise=0.0)
    start, end = site.base.active_hours
    night = (meta["hour"] < start) | (meta["hour"] >= end)
    car = (meta["cls"] == 1) & (meta["carrier"] == wf.CARRIERS.index("none"))
    household = np.isin(meta["make"], _makes(site.household_makes))
    own, stranger = car & household, car & ~household
    assert (own & night).sum() > 20 and (stranger & night).sum() > 20
    assert _routine_rate(y, own & night, 1) == 1.0
    assert _routine_rate(y, stranger & night, 1) == 0.0


def test_carrier_is_routine_inside_its_window_only():
    site = wf.SITES["driveway"]
    X, y, meta = wf.generate(site, 40000, seed=10, classifier_acc=1.0, label_noise=0.0)
    ups = (meta["cls"] == 1) & (meta["carrier"] == wf.CARRIERS.index("ups"))
    start, end = wf.CARRIER_WINDOWS["ups"]
    # A truck idling past the dwell limit is loitering, delivery window or not.
    moving_on = X[:, DWELL] <= site.base.dwell_limit
    inside = ups & moving_on & (meta["hour"] >= start) & (meta["hour"] < end)
    outside = ups & ((meta["hour"] < start) | (meta["hour"] >= end))
    assert inside.sum() > 20 and outside.sum() > 5
    assert _routine_rate(y, inside, 1) == 1.0
    assert _routine_rate(y, outside, 1) == 0.0


def test_forbidden_class_stays_forbidden_whatever_the_subtype():
    _, y, meta = wf.generate(wf.SITES["backdoor"], 20000, seed=12, label_noise=0.0)
    vehicles = meta["cls"] == 1
    assert vehicles.any()
    assert np.all(y[vehicles] == ANOMALY)


def test_coarse_view_drops_the_subtype_columns():
    X, _, _ = wf.generate(wf.SITES["yard"], 300, seed=1)
    assert wf.coarse_view(X).shape == (300, workload.N_FEATURES)


def test_head_sizes_its_input_to_the_taxonomy():
    coarse = FederatedHead(seed=0)
    fine = FederatedHead(seed=0, n_features=wf.N_FEATURES)
    assert coarse.W[0].shape[0] == workload.N_FEATURES
    assert fine.W[0].shape[0] == wf.N_FEATURES
    # CON-3 bounds the head at roughly 200K parameters.
    assert fine.n_params < 220_000

    X, y, _ = wf.generate(wf.SITES["driveway"], 400, seed=1)
    fine.fit(X, y, epochs=1)
    assert 0.0 <= fine.balanced_accuracy(X, y) <= 1.0
