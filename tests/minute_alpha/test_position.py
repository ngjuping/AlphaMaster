"""Position map stays inside [-1, 1] and applies the production dead zone."""

from __future__ import annotations

import numpy as np

from minute_alpha.position import target_positions
from minute_alpha.target import target_log_open


def test_position_bounds_and_deadzone():
    factor = np.array([[-10.0, -0.02, 0.0, 0.02, 0.2, 5.0]])
    pos = target_positions(factor)
    assert pos.min() >= -1.0 and pos.max() <= 1.0
    assert pos[0, 1] == 0.0
    assert pos[0, 2] == 0.0
    assert pos[0, 3] == 0.0
    assert pos[0, 4] > 0.05
    assert pos[0, 0] < -0.99
    assert pos[0, 5] > 0.99


def test_target_matches_open_lag():
    open_ = np.array([[10.0, 11.0, 12.1, 12.1]])
    target = target_log_open(open_)
    np.testing.assert_allclose(target[0, 0], np.log(12.1 / 11.0))
    np.testing.assert_allclose(target[0, 1], np.log(12.1 / 12.1))
    assert target[0, 2] == 0.0 and target[0, 3] == 0.0
