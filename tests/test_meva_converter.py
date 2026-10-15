"""
MEVA tracks-to-events converter.

Synthetic KPF snippets in tmp_path; no MEVA download needed. The key
property is that the 19-feature vector is the one the node produces, so
the expected vectors here are built with features.extract on the same
Track the collector would have built.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.federated import meva
from src.federated.features import Track, extract
from src.federated.store import SampleStore

CLIP = "2018-03-05.13-10-00.13-15-00.school.G336"


def _geom_line(track, frame, box, det=1):
    x1, y1, x2, y2 = box
    return (f"- {{'geom': {{'g0': '{x1} {y1} {x2} {y2}', 'id0': {det}, "
            f"'id1': {track}, 'keyframe': False, 'ts0': {frame}}}}}\n")


def _write_clip(root: Path, clip: str, tracks: dict, acts=None, flow=False):
    """tracks: {id: (cls_name, [(frame, box), ...])}"""
    d = root / "kitware-meva-training" / clip[:10] / clip[11:13]
    d.mkdir(parents=True, exist_ok=True)
    geom, types = [], []
    for tid, (cname, pts) in tracks.items():
        for i, (fr, box) in enumerate(pts):
            if flow:
                geom.append(f"- {{ geom: {{id1: {tid}, id0: {i}, ts0: {fr}, "
                            f"ts1: {fr / 30:.2f}, g0: {' '.join(map(str, box))}, "
                            f"keyframe: true }} }}\n")
            else:
                geom.append(_geom_line(tid, fr, box, i))
        if flow:
            types.append(f"- {{ types: {{ id1: {tid}, cset3: {{ {cname}: 1.0 }} }} }}\n")
        else:
            types.append(f"- {{'types': {{'cset3': {{'{cname}': 1.0}}, 'id1': {tid}}}}}\n")
    (d / f"{clip}.geom.yml").write_text("".join(geom))
    (d / f"{clip}.types.yml").write_text("".join(types))
    lines = []
    for name, actors in (acts or []):
        a = ", ".join(f"{{'id1': {i}, 'timespan': [{{'tsr0': [0, 9]}}]}}" for i in actors)
        lines.append(f"- {{'act': {{'act2': {{'{name}': 1.0}}, 'actors': [{a}], "
                     f"'id2': 1, 'src_status': 'good', 'timespan': [{{'tsr0': [0, 9]}}]}}}}\n")
    (d / f"{clip}.activities.yml").write_text("".join(lines))
    return d / f"{clip}.geom.yml"


def _walk(n=10, x0=100, y=500, step=3, w=60, h=120, f0=0, gap=1):
    return [(f0 + i * gap, (x0 + i * step, y, x0 + i * step + w, y + h)) for i in range(n)]


# ---- line parsing -------------------------------------------------------
def test_parse_geom_line_quoted_style():
    rec = meva.parse_geom_line(_geom_line(15, 891, (150, 954, 264, 1079)))
    assert rec == (15, 891, (150.0, 954.0, 264.0, 1079.0))


def test_parse_geom_line_flow_style():
    line = ("- { geom: {id1: 5573913994148029640, id0: 1, ts0: 1040, ts1: 34.67, "
            "g0: 490 259 545 290, keyframe: true } }\n")
    assert meva.parse_geom_line(line) == (5573913994148029640, 1040, (490.0, 259.0, 545.0, 290.0))


@pytest.mark.parametrize("line", ["", "- { meta: \"x\" }\n", "- {'types': {'id1': 3}}\n", "garbage"])
def test_parse_geom_line_ignores_non_geom(line):
    assert meva.parse_geom_line(line) is None


def test_parse_types_line_both_styles():
    assert meva.parse_types_line("- {'types': {'cset3': {'person': 1.0}, 'id1': 15}}\n") == (15, "person")
    assert meva.parse_types_line("- { types: { id1: 77, cset3: { vehicle: 1.0 } } }\n") == (77, "vehicle")
    assert meva.parse_types_line("- { meta: \"x\" }\n") is None


def test_parse_activity_line_maps_actors():
    line = ("- {'act': {'act2': {'person_exits_vehicle': 1.0}, 'actors': "
            "[{'id1': 5, 'timespan': [{'tsr0': [1, 2]}]}, {'id1': 4, 'timespan': [{'tsr0': [1, 2]}]}], "
            "'id2': 2, 'src_status': 'good', 'timespan': [{'tsr0': [1, 2]}]}}\n")
    assert meva.parse_activity_line(line) == ("person_exits_vehicle", (5, 4))
    assert meva.parse_activity_line("- { meta: \"vehicle_stops 6 instances\" }\n") is None


# ---- class mapping ------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("person", 0), ("vehicle", 1), ("bag", 3),
    ("bicycle", None), ("other", None), ("receptacle", None), ("unknown_thing", None),
])
def test_map_class(name, expected):
    assert meva.map_class(name) == expected


# ---- timestamps ---------------------------------------------------------
def test_parse_clip_name():
    info = meva.parse_clip_name(CLIP + ".geom.yml")
    assert info.start == datetime(2018, 3, 5, 13, 10, 0)
    assert (info.site, info.camera, info.name) == ("school", "G336", CLIP)


def test_parse_clip_name_with_seconds_offset():
    info = meva.parse_clip_name("2018-03-07.16-55-06.17-00-06.school.G336")
    assert info.start == datetime(2018, 3, 7, 16, 55, 6)


def test_parse_clip_name_rejects_garbage():
    with pytest.raises(ValueError):
        meva.parse_clip_name("not-a-clip")


# ---- one event per track, features match the node ----------------------
def test_one_event_per_track_with_node_features(tmp_path):
    g = _write_clip(tmp_path, CLIP, {
        1: ("person", _walk(n=20, f0=300)),
        2: ("vehicle", _walk(n=20, x0=800, y=300, step=10, w=300, h=150, f0=0)),
    })
    events = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))
    assert len(events) == 2
    ev = {e.track_id: e for e in events}

    # Rebuild the Track exactly as the collector does: normalised box,
    # update() per detection, extract() at close time.
    pts = _walk(n=20, f0=300)
    t = Track(track_id=1, cls=0, first_seen=300 / 30.0, last_seen=300 / 30.0)
    for fr, (x1, y1, x2, y2) in pts:
        t.update((x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080), 1.0, fr / 30.0)
    when = datetime(2018, 3, 5, 13, 10, 0) + timedelta(seconds=pts[-1][0] / 30.0)
    np.testing.assert_allclose(ev[1].x, extract(t, when), atol=1e-6)
    assert ev[1].x.shape == (19,)
    assert ev[1].x[0] == 1.0 and ev[2].x[1] == 1.0
    assert ev[1].x[18] == 1.0  # confidence
    assert ev[1].when == when


def test_short_tracks_dropped_like_collector(tmp_path):
    g = _write_clip(tmp_path, CLIP, {1: ("person", _walk(n=2)), 2: ("person", _walk(n=5))})
    events = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))
    assert [e.track_id for e in events] == [2]


def test_unmapped_class_dropped_and_counted(tmp_path):
    g = _write_clip(tmp_path, CLIP, {1: ("bicycle", _walk()), 2: ("person", _walk())})
    stats = meva.ConvertStats()
    events = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080), stats=stats)
    assert [e.track_id for e in events] == [2]
    assert stats.dropped_class == {"bicycle": 1}


def test_boxes_beyond_frame_are_clipped(tmp_path):
    g = _write_clip(tmp_path, CLIP, {1: ("person", [(i, (1900, 1000, 1925, 1085)) for i in range(5)])})
    (e,) = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))
    assert np.all(np.abs(e.x) <= 1.0 + 1e-6)
    assert e.x[8] <= 1.0 and e.x[9] <= 1.0


def test_flow_style_clip_converts(tmp_path):
    g = _write_clip(tmp_path, "2018-03-07.16-55-06.17-00-06.school.G336",
                    {123456789012345678: ("vehicle", _walk(n=6))}, flow=True)
    (e,) = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))
    assert e.cls == 1 and e.when.hour == 16


def test_det_fps_subsamples_updates(tmp_path):
    g = _write_clip(tmp_path, CLIP, {1: ("person", _walk(n=31, step=1))})
    full = meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))[0]
    sub = meva.clip_to_events(g, det_fps=10.0, frame_size=(1920, 1080))[0]
    # Same total displacement, three times the per-update step at 10 fps.
    assert sub.x[10] > full.x[10]
    assert sub.frames < full.frames


def test_activities_attached_to_track(tmp_path):
    g = _write_clip(tmp_path, CLIP, {1: ("person", _walk()), 2: ("vehicle", _walk())},
                    acts=[("person_exits_vehicle", [1, 2]), ("vehicle_stops", [2])])
    ev = {e.track_id: e for e in meva.clip_to_events(g, det_fps=30.0, frame_size=(1920, 1080))}
    assert ev[1].activities == ("person_exits_vehicle",)
    assert set(ev[2].activities) == {"person_exits_vehicle", "vehicle_stops"}


# ---- holdout ------------------------------------------------------------
def test_holdout_is_clip_level_and_deterministic():
    clips = [f"2018-03-05.1{i % 10}-{i:02d}-00.x.school.G336" for i in range(40)]
    a = meva.split_clips(clips, 0.2, seed=0)
    b = meva.split_clips(list(reversed(clips)), 0.2, seed=0)
    assert a == b
    train, hold = a
    assert not set(train) & set(hold)
    assert set(train) | set(hold) == set(clips)
    assert len(hold) == 8
    assert meva.split_clips(clips, 0.2, seed=1)[1] != hold


def test_holdout_small_inputs():
    train, hold = meva.split_clips(["a", "b"], 0.2, seed=0)
    assert len(hold) == 1 and len(train) == 1
    train, hold = meva.split_clips(["a"], 0.2, seed=0)
    assert hold == [] and train == ["a"]


# ---- anomaly rule -------------------------------------------------------
def _ev(cls=0, hour=12, acts=(), clip="c"):
    x = np.zeros(19, dtype=np.float32)
    x[cls] = 1.0
    return meva.Event(clip=clip, camera="G336", track_id=1, cls=cls,
                      when=datetime(2018, 3, 5, hour, 0, 0), x=x,
                      activities=tuple(acts), frames=5)


def test_rule_rare_class_flagged():
    events = [_ev(0)] * 60 + [_ev(1)] * 38 + [_ev(3)] * 1
    prof = meva.build_profile(events)
    assert meva.anomaly_reasons(_ev(3), prof) == ["rare_class"]
    assert meva.anomaly_reasons(_ev(0), prof) == []


def test_rule_rare_activity_flagged():
    events = [_ev(0, acts=("person_talks_to_person",))] * 200 + [_ev(0, acts=("person_sits_down",))]
    prof = meva.build_profile(events)
    assert meva.anomaly_reasons(_ev(0, acts=("person_sits_down",)), prof) == ["rare_activity"]
    assert meva.anomaly_reasons(_ev(0, acts=("person_talks_to_person",)), prof) == []


def test_rule_off_hours_flagged():
    events = [_ev(0, hour=h) for h in (10, 11, 12, 13) for _ in range(100)] + [_ev(0, hour=3)]
    prof = meva.build_profile(events)
    assert 3 not in prof.active_hours and {10, 11, 12, 13} <= prof.active_hours
    assert meva.anomaly_reasons(_ev(0, hour=3), prof) == ["off_hours"]
    assert meva.anomaly_reasons(_ev(0, hour=11), prof) == []


def test_rule_deterministic_and_label_mapping():
    events = [_ev(0)] * 60 + [_ev(1)] * 38 + [_ev(3)]
    p1, p2 = meva.build_profile(events), meva.build_profile(list(reversed(events)))
    assert p1 == p2
    assert [meva.label_for(e, p1) for e in events[:1] + events[-1:]] == [0, 4]
    assert meva.label_for(_ev(1), p1) == 1


# ---- end to end ---------------------------------------------------------
def test_convert_camera_writes_store_and_manifest(tmp_path):
    ann = tmp_path / "ann"
    for i in range(10):
        clip = f"2018-03-05.1{i}-00-00.1{i}-05-00.school.G336"
        _write_clip(ann, clip, {1: ("person", _walk()), 2: ("vehicle", _walk(x0=700))})
    _write_clip(ann, "2018-03-05.10-00-00.10-05-00.admin.G326", {1: ("person", _walk())})
    out = tmp_path / "out"
    res = meva.convert_camera(ann, "G336", out, holdout_frac=0.2, seed=0)
    assert res.clips == 10 and res.events == 20
    store = SampleStore(out / "samples.jsonl")
    assert len(store._rows) == 20 and store.n_labelled == 20
    manifest = json.loads((out / "holdout.json").read_text())
    hold = set(manifest["holdout_clips"])
    assert len(hold) == 2 and not hold & set(manifest["train_clips"])
    for r in store._rows:
        assert (r["meta"]["clip"] in hold) == (r["meta"]["split"] == "holdout")
    # Re-running replaces rather than appends.
    meva.convert_camera(ann, "G336", out, holdout_frac=0.2, seed=0)
    assert len(SampleStore(out / "samples.jsonl")._rows) == 20


def test_convert_camera_no_clips_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        meva.convert_camera(tmp_path, "G999", tmp_path / "o")


# ---- sim/fl.py --store --------------------------------------------------
def test_fl_client_store_split_uses_holdout_clips(tmp_path):
    import importlib.util

    ann = tmp_path / "ann"
    for i in range(10):
        clip = f"2018-03-05.1{i}-00-00.1{i}-05-00.school.G336"
        _write_clip(ann, clip, {1: ("person", _walk()), 2: ("vehicle", _walk(x0=700))})
    out = tmp_path / "out" / "G336"
    meva.convert_camera(ann, "G336", out, holdout_frac=0.2, seed=0)

    spec = importlib.util.spec_from_file_location(
        "sim_fl", Path(__file__).resolve().parents[1] / "sim" / "fl.py")
    fl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl_mod)

    Xt, yt, Xv, yv = fl_mod.load_store_split(out)
    assert Xt.shape == (16, 19) and Xv.shape == (4, 19)
    assert fl_mod.load_store_split(out / "samples.jsonl")[0].shape == (16, 19)
    with pytest.raises(FileNotFoundError):
        fl_mod.load_store_split(tmp_path / "nope")
