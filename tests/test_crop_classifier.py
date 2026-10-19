"""
Runtime crop classifier: which tasks run on which detection, how boxes are
cropped, and that every failure degrades to "no subtype".
"""

import numpy as np
import pytest

from src.detection import crop_classifier as cc


def _fixed(task, classes, winner):
    """A backend that always picks `winner` with ~0.9 probability."""
    def run(batch):
        logits = np.zeros((len(batch), len(classes)), dtype=np.float32)
        logits[:, classes.index(winner)] = 4.0
        return logits
    return cc.TaskModel(task=task, classes=tuple(classes), run=run)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_vehicle_gets_carrier_and_make():
    clf = cc.CropClassifier([
        _fixed("carrier", cc.CARRIERS, "ups"),
        _fixed("make", ["Ford", "Nissan", "Bugatti"], "Nissan"),
        _fixed("size", cc.SIZES, "large"),
    ])
    [r] = clf.classify(FRAME, "vehicle", [(100, 100, 200, 120)])
    assert r["carrier"] == "ups"
    assert r["make"] == "nissan"
    assert "size" not in r
    assert 0.0 < r["confidence"] <= 1.0


def test_package_gets_carrier_and_size_not_make():
    clf = cc.CropClassifier([
        _fixed("carrier", cc.CARRIERS, "usps"),
        _fixed("make", ["Nissan"], "Nissan"),
        _fixed("size", cc.SIZES, "mail"),
    ])
    [r] = clf.classify(FRAME, "package", [(10, 10, 60, 40)])
    assert r == {"carrier": "usps", "size": "mail", "confidence": r["confidence"]}


def test_untracked_make_collapses_to_other():
    clf = cc.CropClassifier([_fixed("make", ["Bugatti", "Nissan"], "Bugatti")])
    [r] = clf.classify(FRAME, "vehicle", [(0, 0, 100, 100)])
    assert r["make"] == "other"


def test_people_and_tiny_boxes_get_no_subtype():
    clf = cc.CropClassifier([_fixed("make", ["Nissan"], "Nissan")])
    assert clf.classify(FRAME, "human", [(0, 0, 100, 200)]) == [None]
    assert clf.classify(FRAME, "vehicle", [(0, 0, 5, 5), (0, 0, 80, 60)])[0] is None


def test_missing_task_model_is_simply_skipped():
    clf = cc.CropClassifier([_fixed("size", cc.SIZES, "small")])
    assert clf.classify(FRAME, "vehicle", [(0, 0, 100, 100)]) == [None]


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        cc.CropClassifier([_fixed("colour", ["red"], "red")])


def test_padded_crop_is_clamped_to_the_frame():
    crop = cc.padded_crop(FRAME, (600, 440, 100, 100))
    assert crop is not None
    assert crop.shape[0] <= 480 and crop.shape[1] <= 640


def test_preprocess_matches_training_layout():
    x = cc.preprocess(np.full((50, 80, 3), 128, dtype=np.uint8))
    assert x.shape == (3, cc.IMG, cc.IMG)
    assert x.dtype == np.float32


# ---- DetectionService integration ------------------------------------------

from src.detection import Detection, DetectionService, DetectionType  # noqa: E402


class _Boom:
    def classify(self, *a, **k):
        raise RuntimeError("model fell over")


def test_service_fills_subtype_on_vehicles_and_packages_only():
    svc = DetectionService()
    svc.attach_crop_classifier(cc.CropClassifier([
        _fixed("carrier", cc.CARRIERS, "amazon"),
        _fixed("make", ["Ford"], "Ford"),
        _fixed("size", cc.SIZES, "large"),
    ]))
    dets = [
        Detection(DetectionType.VEHICLE, 0.9, (10, 10, 120, 80)),
        Detection(DetectionType.PACKAGE, 0.8, (300, 300, 60, 50)),
        Detection(DetectionType.HUMAN, 0.9, (200, 50, 60, 160)),
    ]
    svc.classify_crops(FRAME, dets)
    assert dets[0].subtype["carrier"] == "amazon" and dets[0].subtype["make"] == "ford"
    assert dets[1].subtype["size"] == "large" and "make" not in dets[1].subtype
    assert dets[2].subtype is None


def test_service_survives_a_crop_classifier_failure():
    svc = DetectionService()
    svc.attach_crop_classifier(_Boom())
    dets = [Detection(DetectionType.VEHICLE, 0.9, (10, 10, 120, 80))]
    svc.classify_crops(FRAME, dets)
    assert dets[0].subtype is None


def test_no_crop_classifier_means_no_change():
    svc = DetectionService()
    dets = [Detection(DetectionType.VEHICLE, 0.9, (10, 10, 120, 80))]
    svc.classify_crops(FRAME, dets)
    assert dets[0].subtype is None
