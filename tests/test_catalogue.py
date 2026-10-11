import os

import numpy as np
import pytest

from ahr_rl.catalogue import STATIC_NAMES, index_catalogues, load_catalogue, static_features

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
I = {n: i for i, n in enumerate(STATIC_NAMES)}


def test_flat_catalogue_features():
    cat = load_catalogue(os.path.join(FIX, "catalogue_flat.json"))
    s = static_features(cat, [6913761, 102098426, 999])
    a, b, missing = s
    assert a[I["form_last"]] == pytest.approx(0.1)  # '311x': digits 3,1,1 -> last finish 1st
    assert a[I["first_up"]] == 1.0  # ends with 'x'
    assert a[I["form_wins5"]] == pytest.approx(2 / 5)
    assert a[I["draw_rel"]] == pytest.approx(3 / 4)
    assert a[I["is_flat"]] == 1.0 and a[I["is_harness"]] == 0.0
    assert a[I["distance_km"]] == pytest.approx(1.2)
    mean_w = np.mean([59.5, 58, 58, 55])
    assert a[I["weight_rel"]] == pytest.approx((59.5 - mean_w) / 5)
    assert b[I["female"]] == 1.0 and b[I["claim"]] == pytest.approx(0.5)
    assert not missing.any()  # runner not in the catalogue -> zeros


def test_harness_catalogue_features():
    cat = load_catalogue(os.path.join(FIX, "catalogue_harness.json"))
    s = static_features(cat, [95263970])
    assert s[0, I["is_harness"]] == 1.0 and s[0, I["weight_rel"]] == 0.0
    assert s[0, I["distance_km"]] == pytest.approx(2.24)
    assert s[0, I["form_last"]] == pytest.approx(0.4)  # '3x124' -> last 4


def test_index_catalogues():
    idx = index_catalogues(FIX)
    assert idx == {}  # fixture names don't follow the recorder's naming
