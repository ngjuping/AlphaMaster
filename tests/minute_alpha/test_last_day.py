"""The last-day driver is deterministic and does not read future closes."""

from __future__ import annotations

import numpy as np

from minute_alpha.last_day import run_from_fields, simple_next_return


def _fields(t: int, n: int = 2, seed: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, size=(n, t))
    close = 100 * np.exp(np.cumsum(ret, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    high = np.maximum(close, open_) * 1.001
    low = np.minimum(close, open_) * 0.999
    volume = rng.random((n, t)) + 0.2
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": np.arange(t, dtype=np.int64) * 60 + 1_700_000_000,
        "symbols": np.array(["AAA", "BBB"]),
    }


def test_simple_return_is_next_close():
    close = np.array([[10.0, 11.0, 9.0]])
    out = simple_next_return(close)
    assert out[0, 0] == 0.0
    np.testing.assert_allclose(out[0, 1], 0.1)
    np.testing.assert_allclose(out[0, 2], 9.0 / 11.0 - 1.0)


def test_last_day_repeat_matches_and_ignores_future():
    window, minutes = 48, 3
    fields = _fields(window + minutes + 2)
    first = run_from_fields(fields, minutes=minutes, window=window, n_formulas=6, n_jobs=1)
    second = run_from_fields(fields, minutes=minutes, window=window, n_formulas=6, n_jobs=1)
    np.testing.assert_allclose(first["positions"], second["positions"])
    assert first["names"] == second["names"]

    bumped = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in fields.items()}
    # Last settlement bar's close is in the future of every earlier decision.
    bumped["close"] = bumped["close"].copy()
    bumped["close"][:, -1] *= 1.4
    third = run_from_fields(bumped, minutes=minutes, window=window, n_formulas=6, n_jobs=1)
    np.testing.assert_allclose(first["positions"], third["positions"])
    parallel = run_from_fields(fields, minutes=minutes, window=window, n_formulas=6, n_jobs=2)
    np.testing.assert_allclose(first["positions"], parallel["positions"])
    assert first["names"] == parallel["names"]
