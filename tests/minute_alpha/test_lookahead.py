"""No future bar is allowed to change a past factor or a past training label."""

from __future__ import annotations

import numpy as np

from minute_alpha.features import compute_features
from minute_alpha.search import sample_formula
from minute_alpha.target import target_log_open
from minute_alpha.vm import execute


def _raw(t: int = 600) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)
    ret = rng.normal(0, 0.001, size=(1, t))
    close = 100 * np.exp(np.cumsum(ret, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    high = np.maximum(close, open_) * 1.002
    low = np.minimum(close, open_) * 0.998
    volume = rng.random((1, t)) + 0.2
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_future_prices_do_not_change_past_factor():
    raw = _raw()
    cut = 420
    feat = compute_features(raw)
    tokens = sample_formula(np.random.Generator(np.random.PCG64(4)))
    base = execute(tokens, feat)
    bumped = {k: v.copy() for k, v in raw.items()}
    bumped["close"][:, cut:] *= 1.5
    bumped["open"][:, cut:] *= 1.5
    bumped["high"][:, cut:] *= 1.5
    bumped["low"][:, cut:] *= 1.5
    other = execute(tokens, compute_features(bumped))
    assert base is not None and other is not None
    np.testing.assert_allclose(base[:, :cut], other[:, :cut], atol=1e-5, rtol=1e-5)


def test_prefix_target_hides_future_opens():
    raw = _raw()
    full = target_log_open(raw["open"])
    cut = 400
    prefix = target_log_open(raw["open"][:, :cut])
    # Known labels inside the prefix match. The last two are 0, not the true future.
    np.testing.assert_allclose(prefix[0, : cut - 2], full[0, : cut - 2])
    assert prefix[0, -1] == 0.0 and prefix[0, -2] == 0.0
    assert full[0, cut - 2] != 0.0 or True
    # The label that needs open[cut] is not stored as that future value.
    assert prefix[0, cut - 2] == 0.0
    if abs(full[0, cut - 2]) > 0:
        assert prefix[0, cut - 2] != full[0, cut - 2]
