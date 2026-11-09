"""Week backtest marks missing minutes instead of filling them with zero."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minute_alpha.last_week import _tables, run_from_packs


def _pack(times, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=np.int64)
    ret = rng.normal(0, 0.001, size=times.size)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "time": times,
        "open": open_,
        "high": np.maximum(close, open_) * 1.001,
        "low": np.minimum(close, open_) * 0.999,
        "close": close,
        "volume": rng.random(times.size) + 0.2,
    }


def test_gap_is_not_zero_and_repeat_matches():
    window, minutes = 40, 6
    latest = 1_700_000_000 + (window + minutes - 1) * 60
    clock = latest - np.arange(window + minutes)[::-1] * 60
    # Drop one settlement minute from the second symbol only.
    hole = int(clock[window + 2])
    b_times = clock[clock != hole]
    packs = {"AAA": _pack(clock, 1), "BBB": _pack(b_times, 2)}
    first = run_from_packs(packs, latest_open=latest, minutes=minutes, window=window, n_formulas=4, n_jobs=1)
    second = run_from_packs(packs, latest_open=latest, minutes=minutes, window=window, n_formulas=4, n_jobs=1)
    b_rows = first["per_symbol"][1]
    hole_rows = [r for r in b_rows if r["decision"] == hole - 60 or r["decision"] == hole]
    assert hole_rows, "expected a gap row next to the missing print"
    assert any(r["gap"] and r["reason"] in {"missing_print", "missing_next_close"} for r in b_rows)
    assert all(not (r["gap"] and r["simple_return"] == 0) for r in b_rows)
    for left, right in zip(first["per_symbol"], second["per_symbol"]):
        for a, b in zip(left, right):
            assert a["gap"] == b["gap"]
            assert a["reason"] == b["reason"]
            if not a["gap"]:
                assert a["position"] == b["position"]
                assert a["name"] == b["name"]
    carried = dict(first)
    carried["carry_position"] = True
    tables = _tables(carried)
    pos = tables["position_target_cumsum"]
    bbb = pos[pos.symbol == "BBB"].reset_index(drop=True)
    gap_at = bbb.index[bbb["target"].isna()]
    assert len(gap_at)
    for idx in gap_at:
        assert pd.isna(bbb.loc[idx, "target"])
        assert pd.isna(bbb.loc[idx, "product"])
        if idx > 0 and pd.notna(bbb.loc[idx - 1, "position"]):
            assert bbb.loc[idx, "position"] == bbb.loc[idx - 1, "position"]
            assert bbb.loc[idx, "position"] != 0 or bbb.loc[idx - 1, "position"] != 0
