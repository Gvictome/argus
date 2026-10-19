"""
Label plumbing for the second-stage crop classifier: turning dataset class
names into the carrier / make / package-size slots the fine taxonomy uses.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training", "crop"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

import taxonomy as tx  # noqa: E402
import workload_fine as wf  # noqa: E402


@pytest.mark.parametrize("name,make", [
    ("Nissan Leaf Hatchback 2012", "Nissan"),
    ("AM General Hummer SUV 2000", "AM General"),
    ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin"),
    ("Land Rover Range Rover SUV 2012", "Land Rover"),
    ("Mercedes-Benz S-Class Sedan 2012", "Mercedes-Benz"),
    ("smart fortwo Convertible 2012", "smart"),
])
def test_make_is_parsed_from_a_stanford_cars_class_name(name, make):
    assert tx.make_of(name) == make


def test_argus_make_slot_matches_the_sim_taxonomy():
    assert tx.ARGUS_MAKES == wf.MAKES
    assert tx.argus_make("Nissan") == "nissan"
    assert tx.argus_make("Chevrolet") == "chevrolet"
    assert tx.argus_make("Tesla") == "tesla"
    assert tx.argus_make("Bugatti") == "other"


def test_carrier_names_from_roboflow_map_onto_argus_carriers():
    assert tx.ARGUS_CARRIERS == wf.CARRIERS
    assert tx.argus_carrier("UPS") == "ups"
    assert tx.argus_carrier("UPS-van") == "ups"
    assert tx.argus_carrier("USPS-Truck") == "usps"
    assert tx.argus_carrier("Fedex-truck") == "fedex"
    assert tx.argus_carrier("Amazon-van") == "amazon"
    assert tx.argus_carrier("Other-Vehicles") == "none"
    # Logos are not vehicles; they are dropped, not guessed.
    assert tx.argus_carrier("UPS-logo") is None


def test_package_names_map_onto_argus_sizes():
    assert tx.ARGUS_SIZES == wf.PACKAGE_SIZES
    assert tx.argus_size("envelope") == "mail"
    assert tx.argus_size("mailer") == "mail"
    assert tx.argus_size("large box") == "large"
    assert tx.argus_size("box") == "small"
    assert tx.argus_size("person") is None


def test_crop_box_is_padded_and_clamped_to_the_image():
    # 10% padding each side, clamped at the image edge.
    assert tx.crop_box((10, 10, 50, 30), img_w=100, img_h=100, pad=0.1) == (6, 8, 54, 32)
    assert tx.crop_box((0, 0, 100, 100), img_w=100, img_h=100, pad=0.1) == (0, 0, 100, 100)


def test_crop_box_rejects_degenerate_boxes():
    assert tx.crop_box((10, 10, 10, 40), img_w=100, img_h=100) is None
    assert tx.crop_box((10, 10, 12, 12), img_w=100, img_h=100, min_side=8) is None


def test_collapsing_predictions_to_argus_slots_scores_correctly():
    makes = ["Nissan", "Toyota", "Bugatti", "Ferrari"]
    # Bugatti predicted as Ferrari is wrong at make level but right at ARGUS
    # level: both are "other".
    acc_make, acc_slot = tx.make_accuracies(true=[0, 1, 2], pred=[0, 0, 3], classes=makes)
    assert acc_make == pytest.approx(1 / 3)
    assert acc_slot == pytest.approx(2 / 3)
