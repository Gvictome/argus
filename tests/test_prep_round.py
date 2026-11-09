"""
Round-prep rules: parked repeats collapse to one card per object per hour,
and the queue spreads through time instead of bunching.
"""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prep_round", Path(__file__).resolve().parents[1] / "scripts" / "prep_round.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _row(n, t, cls=1, cx=0.3, speed=0.01):
    x = [0.0] * 19
    x[cls] = 1.0
    x[4], x[5], x[8], x[9], x[12], x[18] = 0.2, 0.2, cx, 0.5, speed, 0.8
    return {"n": n, "t": t, "x": x}


def test_parked_car_is_one_representative_per_hour():
    rows = [_row(i, 1000 + i * 300) for i in range(25)]   # every 5 min for 2 h
    groups = prep.group_parked(rows)
    reps = [n for n, (_, rep) in groups.items() if rep]
    assert len({g for g, _ in groups.values()}) == 1
    assert len(reps) == 3                                  # t=0, ~1 h, ~2 h


def test_moving_vehicle_and_people_are_not_grouped():
    rows = [_row(1, 1000, speed=0.3), _row(2, 1300, cls=0)]
    assert prep.group_parked(rows) == {}


def test_two_spots_are_two_objects():
    rows = [_row(1, 1000, cx=0.2), _row(2, 1001, cx=0.7), _row(3, 1300, cx=0.2)]
    groups = prep.group_parked(rows)
    assert groups[1][0] == groups[3][0] != groups[2][0]


def test_spread_is_even_through_time():
    items = [{"t": float(i)} for i in range(100)]
    picked = prep._spread(items, 10)
    assert [p["t"] for p in picked] == [float(i * 10) for i in range(10)]
