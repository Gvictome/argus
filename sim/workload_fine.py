"""
Fine-grained taxonomy for the ARGUS federated head: who the vehicle or
package belongs to, not just that it is one.

The coarse generator (workload.py) gives the head four classes. That is
enough to say "a vehicle at 03:00 is unusual", but not "the resident's
Nissan at 03:00 is fine, a stranger's car is not", or "a UPS truck at
14:00 is a delivery, the same truck at 23:00 is not". Those judgements need
the subtype.

Where the subtype comes from
----------------------------
Not from the head. A second-stage crop classifier on the Hailo looks at
each vehicle or package box YOLO emits and returns carrier, make and
package size. That model does not exist yet; here it is simulated as the
true subtype passed through a classifier with a chosen accuracy
(`classifier_acc`). Labels are decided from the TRUE subtype, features
carry the PREDICTED one, so a weak classifier costs the head exactly what
it would cost on a real node.

Feature vector (35 dims) -- coarse 19 first, unchanged, then:
 19-23  carrier one-hot: none, ups, usps, fedex, amazon (vehicles, packages)
 24-30  make one-hot: nissan, toyota, honda, ford, chevrolet, tesla, other (vehicles)
 31-33  package size one-hot: mail, small, large (packages)
 34     crop classifier confidence (0 for people and animals)

Labels are the same five as the coarse head, so the two can be compared
on identical events.

Simulation assumptions, not sourced figures: the carrier delivery windows,
the per-site subtype mixes, and the size multipliers below.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

import workload
from workload import Site

CARRIERS = ("none", "ups", "usps", "fedex", "amazon")
MAKES = ("nissan", "toyota", "honda", "ford", "chevrolet", "tesla", "other")
PACKAGE_SIZES = ("mail", "small", "large")

N_COARSE = workload.N_FEATURES
CARRIER_SLICE = slice(N_COARSE, N_COARSE + len(CARRIERS))
MAKE_SLICE = slice(CARRIER_SLICE.stop, CARRIER_SLICE.stop + len(MAKES))
SIZE_SLICE = slice(MAKE_SLICE.stop, MAKE_SLICE.stop + len(PACKAGE_SIZES))
CONF_INDEX = SIZE_SLICE.stop
N_FEATURES = CONF_INDEX + 1
N_CLASSES = workload.N_CLASSES
ANOMALY = 4
NOT_APPLICABLE = -1

# Hours a carrier is expected on a residential street, [start, end).
CARRIER_WINDOWS: Dict[str, Tuple[int, int]] = {
    "ups": (9, 21),
    "usps": (9, 18),
    "fedex": (8, 20),
    "amazon": (8, 22),
}

# Carrier fleets are not consumer makes; Amazon's vans are mostly Ford Transits.
_CARRIER_FLEET_MAKE = {"ups": "other", "usps": "other", "fedex": "other", "amazon": "ford"}

_VEHICLE, _PACKAGE = 1, 3


@dataclass(frozen=True)
class FineSite:
    """A coarse site plus who normally shows up there."""

    base: Site
    household_makes: Tuple[str, ...]        # residents' cars: routine at any hour
    vehicle_carrier_prior: Tuple[float, ...]  # over CARRIERS
    make_prior: Tuple[float, ...]           # over MAKES, non-carrier vehicles
    package_carrier_prior: Tuple[float, ...]  # over CARRIERS
    usps_mail_share: float                  # USPS items that are letters/flats
    large_share: float                      # non-mail parcels that are large

    @property
    def key(self) -> str:
        return self.base.key


_CURBSIDE = Site(
    key="curbside",
    label="Street curb and mailbox",
    class_prior=(0.30, 0.45, 0.05, 0.20),
    active_hours=(7, 21),
    anomalous_classes=(),
    dwell_limit=0.60,
    note="Mail truck daily, residents park here. Strangers idling are the signal.",
)

SITES: Dict[str, FineSite] = {
    "driveway": FineSite(
        base=workload.SITES["driveway"],
        household_makes=("nissan", "toyota"),
        vehicle_carrier_prior=(0.70, 0.08, 0.05, 0.05, 0.12),
        make_prior=(0.25, 0.20, 0.15, 0.12, 0.10, 0.06, 0.12),
        package_carrier_prior=(0.10, 0.25, 0.25, 0.15, 0.25),
        usps_mail_share=0.50,
        large_share=0.35,
    ),
    "backdoor": FineSite(
        base=workload.SITES["backdoor"],
        household_makes=(),
        vehicle_carrier_prior=(0.40, 0.20, 0.05, 0.20, 0.15),
        make_prior=(0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.15),
        package_carrier_prior=(0.05, 0.35, 0.10, 0.30, 0.20),
        usps_mail_share=0.10,
        large_share=0.60,
    ),
    "yard": FineSite(
        base=workload.SITES["yard"],
        household_makes=(),
        vehicle_carrier_prior=(0.80, 0.05, 0.05, 0.05, 0.05),
        make_prior=(0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.15),
        package_carrier_prior=(0.40, 0.15, 0.15, 0.15, 0.15),
        usps_mail_share=0.20,
        large_share=0.30,
    ),
    "curbside": FineSite(
        base=_CURBSIDE,
        household_makes=("honda", "ford"),
        vehicle_carrier_prior=(0.62, 0.06, 0.20, 0.04, 0.08),
        make_prior=(0.12, 0.15, 0.22, 0.20, 0.12, 0.07, 0.12),
        package_carrier_prior=(0.05, 0.15, 0.55, 0.10, 0.15),
        usps_mail_share=0.80,
        large_share=0.25,
    ),
}


def _sample_subtypes(site: FineSite, cls: np.ndarray,
                     rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """True carrier / make / size per event; NOT_APPLICABLE where meaningless."""
    n = len(cls)
    carrier = np.full(n, NOT_APPLICABLE, dtype=np.int64)
    make = np.full(n, NOT_APPLICABLE, dtype=np.int64)
    size = np.full(n, NOT_APPLICABLE, dtype=np.int64)

    veh = cls == _VEHICLE
    carrier[veh] = rng.choice(len(CARRIERS), size=int(veh.sum()), p=site.vehicle_carrier_prior)
    make[veh] = rng.choice(len(MAKES), size=int(veh.sum()), p=site.make_prior)
    for name, fleet_make in _CARRIER_FLEET_MAKE.items():
        make[veh & (carrier == CARRIERS.index(name))] = MAKES.index(fleet_make)

    pkg = cls == _PACKAGE
    carrier[pkg] = rng.choice(len(CARRIERS), size=int(pkg.sum()), p=site.package_carrier_prior)
    large = rng.random(n) < site.large_share
    size[pkg] = np.where(large[pkg], PACKAGE_SIZES.index("large"), PACKAGE_SIZES.index("small"))
    # Letters and flats only arrive by USPS.
    usps_pkg = pkg & (carrier == CARRIERS.index("usps"))
    mail = usps_pkg & (rng.random(n) < site.usps_mail_share)
    size[mail] = PACKAGE_SIZES.index("mail")

    return {"carrier": carrier, "make": make, "size": size}


def _scale_geometry(ev: Dict[str, np.ndarray], sub: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Box trucks stand taller than cars; mail is small, large parcels big."""
    w, h = ev["w"].copy(), ev["h"].copy()
    truck = (ev["cls"] == _VEHICLE) & (sub["carrier"] > 0)
    w[truck] *= 1.1
    h[truck] *= 1.6
    for name, factor in (("mail", 0.45), ("large", 1.7)):
        m = sub["size"] == PACKAGE_SIZES.index(name)
        w[m] *= factor
        h[m] *= factor
    return {**ev, "w": np.clip(w, 0.01, 1.0), "h": np.clip(h, 0.01, 1.0)}


def _misclassify(true: np.ndarray, n_values: int, acc: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Crop classifier: right with probability acc, else a uniform wrong answer."""
    pred = true.copy()
    applicable = true != NOT_APPLICABLE
    wrong = applicable & (rng.random(len(true)) >= acc)
    shift = rng.integers(1, n_values, size=len(true))
    pred[wrong] = (true[wrong] + shift[wrong]) % n_values
    return pred


def _anomaly(site: FineSite, ev: Dict[str, np.ndarray],
             sub: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalcy decided from the TRUE subtype."""
    cls, hour = ev["cls"], ev["hour"]
    base = workload.anomaly_mask(site.base, ev)

    forbidden = np.zeros(len(cls), dtype=bool)
    for name in site.base.anomalous_classes:
        forbidden |= cls == workload.CLASS_NAMES.index(name)
    loitering = (ev["dwell"] > site.base.dwell_limit) & (cls != _PACKAGE)

    in_window = np.zeros(len(cls), dtype=bool)
    for name, (start, end) in CARRIER_WINDOWS.items():
        is_c = sub["carrier"] == CARRIERS.index(name)
        in_window |= is_c & (hour >= start) & (hour < end)
    carried = ((cls == _VEHICLE) | (cls == _PACKAGE)) & (sub["carrier"] > 0)

    household_idx = [MAKES.index(m) for m in site.household_makes]
    household = (cls == _VEHICLE) & (sub["carrier"] == 0) & np.isin(sub["make"], household_idx)

    rule = np.where(carried, ~in_window | loitering, np.where(household, False, base))
    return forbidden | rule


def _one_hot(idx: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(idx), width), dtype=np.float32)
    ok = idx != NOT_APPLICABLE
    out[np.flatnonzero(ok), idx[ok]] = 1.0
    return out


def generate(
    site: FineSite,
    n: int,
    seed: int = 0,
    classifier_acc: float = 0.88,
    night_rate: float = 0.12,
    label_noise: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return (X[n, 35] float32, y[n] int64, meta) for one site.

    meta holds the true and predicted subtypes per event, for analysis only;
    the head never sees it.
    """
    if not 0.0 <= classifier_acc <= 1.0:
        raise ValueError(f"classifier_acc must be in [0, 1], got {classifier_acc}")
    rng = np.random.default_rng(seed)

    ev = workload.sample_events(site.base, n, rng, night_rate)
    sub = _sample_subtypes(site, ev["cls"], rng)
    ev = _scale_geometry(ev, sub)

    pred = {
        "carrier": _misclassify(sub["carrier"], len(CARRIERS), classifier_acc, rng),
        "make": _misclassify(sub["make"], len(MAKES), classifier_acc, rng),
        "size": _misclassify(sub["size"], len(PACKAGE_SIZES), classifier_acc, rng),
    }
    right = np.all([pred[k] == sub[k] for k in pred], axis=0)
    crop = (ev["cls"] == _VEHICLE) | (ev["cls"] == _PACKAGE)
    conf = np.where(right, rng.beta(8.0, 2.0, n), rng.beta(3.0, 3.0, n))
    conf = np.where(crop, conf, 0.0).astype(np.float32)

    y = np.where(_anomaly(site, ev, sub), ANOMALY, ev["cls"]).astype(np.int64)
    y = workload.add_label_noise(y, rng, label_noise)

    X = np.column_stack([
        workload.assemble(ev),
        _one_hot(pred["carrier"], len(CARRIERS)),
        _one_hot(pred["make"], len(MAKES)),
        _one_hot(pred["size"], len(PACKAGE_SIZES)),
        conf,
    ]).astype(np.float32)

    meta = {"cls": ev["cls"], "hour": ev["hour"], **sub,
            **{f"{k}_pred": v for k, v in pred.items()}}
    return X, y, meta


def site_keys(taxonomy: str) -> list:
    return list(SITES if taxonomy == "fine" else workload.SITES)


def load_site(site_key: str, samples: int, seed: int, taxonomy: str = "coarse",
              classifier_acc: float = 0.88) -> Tuple[np.ndarray, np.ndarray, object, int]:
    """(X, y, site, n_features) for one site under either taxonomy."""
    if taxonomy == "fine":
        site = SITES[site_key]
        X, y, _ = generate(site, samples, seed=seed, classifier_acc=classifier_acc)
        return X, y, site, N_FEATURES
    site = workload.SITES[site_key]
    X, y = workload.generate(site, samples, seed=seed)
    return X, y, site, workload.N_FEATURES


def coarse_view(X: np.ndarray) -> np.ndarray:
    """The same events as the coarse head would see them."""
    return X[:, :N_COARSE]


def describe(site: FineSite, y: np.ndarray, meta: Dict[str, np.ndarray]) -> str:
    counts = np.bincount(y, minlength=N_CLASSES)
    veh = meta["cls"] == _VEHICLE
    pkg = meta["cls"] == _PACKAGE
    carried = (veh | pkg) & (meta["carrier"] > 0)
    return (f"{site.key:9} n={len(y):5}  anomaly={counts[ANOMALY] / len(y):.1%}  "
            f"vehicles={int(veh.sum())} packages={int(pkg.sum())} "
            f"carrier-borne={int(carried.sum())}")
