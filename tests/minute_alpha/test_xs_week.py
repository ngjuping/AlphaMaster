"""Cross-sectional week retrain: clock, costs, rank gross, and no look-ahead."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from minute_alpha.position import target_positions
from minute_alpha.xs_week import (
    ANCHOR_1M_OPEN,
    aggregate_symbol_ics,
    bar_seconds,
    base_window,
    cross_section_weights,
    decision_times,
    latest_closed_open,
    packs_from_frames,
    periods_per_year_bar,
    pick_best,
    run_variant_from_packs,
    settle_with_hold,
    usable_mask,
    variant_specs,
    week_bars,
    window_forward_returns,
    _commit_dir,
    _rank_scores,
    build_tables,
)


def test_clock_matches_the_one_minute_anchor():
    assert ANCHOR_1M_OPEN == 1794198240
    assert datetime.fromtimestamp(ANCHOR_1M_OPEN, timezone.utc) == datetime(2026, 11, 9, 4, 24, tzinfo=timezone.utc)
    latest_5 = latest_closed_open(300)
    latest_15 = latest_closed_open(900)
    assert datetime.fromtimestamp(latest_5, timezone.utc) == datetime(2026, 11, 9, 4, 20, tzinfo=timezone.utc)
    assert datetime.fromtimestamp(latest_15, timezone.utc) == datetime(2026, 11, 9, 4, 0, tzinfo=timezone.utc)
    d5 = decision_times(latest_5, 300, week_bars(300))
    d15 = decision_times(latest_15, 900, week_bars(900))
    assert len(d5) == 2016 and len(d15) == 672
    assert int(d5[-1]) + 300 == latest_5
    assert int(d15[-1]) + 900 == latest_15
    assert datetime.fromtimestamp(int(d5[0]), timezone.utc) == datetime(2026, 11, 2, 4, 20, tzinfo=timezone.utc)
    assert datetime.fromtimestamp(int(d15[0]), timezone.utc) == datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)
    assert periods_per_year_bar(300) == 105192
    assert periods_per_year_bar(900) == 35064
    assert base_window("5m") == 864 and base_window("15m") == 288
    assert bar_seconds("5m") == 300 and bar_seconds("15m") == 900


def test_variant_directories():
    assert [spec["dirname"] for spec in variant_specs()] == [
        "portal_out_7d_5m_xs_ts",
        "portal_out_7d_5m_xs_ts_min",
        "portal_out_7d_5m_xs_rank",
        "portal_out_7d_5m_xs_rank_long",
        "portal_out_7d_15m_xs_ts",
        "portal_out_7d_15m_xs_ts_min",
        "portal_out_7d_15m_xs_rank",
        "portal_out_7d_15m_xs_rank_long",
    ]
    windows = {(spec["timeframe"], spec["variant"]): spec["window"] for spec in variant_specs()}
    assert windows[("5m", "ts")] == 864
    assert windows[("15m", "ts_min")] == 288
    assert windows[("5m", "rank_long")] == 2592
    assert windows[("15m", "rank_long")] == 864


def test_objectives_and_rank_weights():
    stacked = np.array([[0.9, 0.3], [0.1, 0.3]])
    assert pick_best(aggregate_symbol_ics(stacked, "mean")) == 0
    assert pick_best(aggregate_symbol_ics(stacked, "min")) == 1
    assert pick_best(np.array([0.2, 0.2, 0.1])) == 0
    weights = cross_section_weights(np.array([1.0, 1.0, 3.0]), np.array([0.4, -0.2, 0.1]))
    assert abs(weights.sum()) < 1e-12
    assert abs(np.abs(weights).sum() - 0.7) < 1e-12
    assert weights[2] > 0 and weights[0] < 0
    flat = cross_section_weights(np.array([2.0, 2.0, 2.0]), np.array([0.3, 0.3, 0.3]))
    assert np.allclose(flat, 0.0)
    mask = usable_mask(np.ones(600), 600)
    assert not mask[:499].any()
    assert mask[499]


def test_forward_return_ignores_the_future_and_blank_is_not_zero():
    times = np.array([0, 300, 600], dtype=np.int64)
    close = np.array([100.0, 110.0, 121.0])
    known = window_forward_returns(times, close, times, close, 300, decision=600)
    assert abs(known[0] - 0.1) < 1e-12
    assert abs(known[1] - (121.0 / 110.0 - 1.0)) < 1e-12
    assert np.isnan(known[2])
    later_time = np.array([0, 300, 600, 900], dtype=np.int64)
    later_close = np.array([100.0, 110.0, 121.0, 5.0])
    still = window_forward_returns(times, close, later_time, later_close, 300, decision=600)
    np.testing.assert_allclose(known, still, equal_nan=True)

    pnl, turnover, held = settle_with_hold(
        np.array([0.5, np.nan, -0.25]),
        np.array([0.01, np.nan, 0.02]),
        0.0001,
    )
    assert np.isnan(pnl[1]) and np.isnan(turnover[1])
    assert abs(held[1] - 0.5) < 1e-12
    assert abs(pnl[0] - (0.5 * 0.01 - 0.5 * 0.0001)) < 1e-12
    assert abs(turnover[2] - 0.75) < 1e-12
    assert abs(pnl[2] - (-0.25 * 0.02 - 0.75 * 0.0001)) < 1e-12
    # A trade on the blank bar is still charged, on the next real return.
    pnl2, turnover2, _held2 = settle_with_hold(
        np.array([0.5, 0.2, 0.2]),
        np.array([0.01, np.nan, 0.02]),
        0.0001,
    )
    assert abs(turnover2[0] - 0.5) < 1e-12
    assert np.isnan(pnl2[1])
    assert abs(turnover2[2] - 0.3) < 1e-12


def test_rank_score_prefers_the_cross_sectional_formula():
    decision = 600
    times = np.array([0, 300, 600], dtype=np.int64)
    # Two interior points where formula 0 lines up with the next return and formula 1 does not.
    prepared = []
    returns = np.array([0.01, 0.02, -0.01])
    for sign in (1.0, 2.0, 3.0):
        facs = np.vstack([
            np.array([sign * 0.01, sign * 0.02, 0.0]),
            np.array([-sign * 0.01, -sign * 0.02, 0.0]),
        ])
        ret = np.array([
            returns[0] * np.sign(sign),
            returns[1] * np.sign(sign),
            np.nan,
        ])
        prepared.append(("S", {
            "times": times,
            "facs": facs,
            "ret": ret,
            "ok": np.array([True, True]),
        }))
    # Rebuild per symbol so the cross section is [1, 2, 3] times the return pattern.
    prepared = []
    for scale in (1.0, 2.0, 3.0):
        facs = np.zeros((2, 3))
        facs[0] = np.array([scale, scale, 0.0])
        facs[1] = np.array([-scale, -scale, 0.0])
        ret = np.array([scale * 0.01, scale * 0.02, np.nan])
        prepared.append((f"S{scale}", {
            "times": times,
            "facs": facs,
            "ret": ret,
            "ok": np.array([True, True]),
        }))
    scores = _rank_scores(prepared, decision, min_names=3)
    assert pick_best(scores) == 0
    assert scores[0] > 0.9
    assert scores[1] < -0.9


def _frames(times: np.ndarray, seed: int) -> dict[str, pd.DataFrame]:
    frames = {}
    for i, symbol in enumerate(("AAA", "BBB", "CCC")):
        rng = np.random.default_rng(seed + i)
        steps = rng.normal(0.001 * (i - 1), 0.01, size=times.size)
        close = 100 * np.exp(np.cumsum(steps))
        open_ = np.concatenate([[close[0]], close[:-1]])
        frames[symbol] = pd.DataFrame({
            "time": times.astype(np.int64),
            "open": open_,
            "high": np.maximum(close, open_) * 1.002,
            "low": np.minimum(close, open_) * 0.998,
            "close": close,
            "volume": rng.random(times.size) + 0.2,
        })
    return frames


def test_retrain_does_not_read_the_future_and_gaps_hold():
    window, n_decisions, bar = 80, 2, 300
    start = 1_700_000_000 - (1_700_000_000 % bar)
    times = start + np.arange(window + n_decisions, dtype=np.int64) * bar
    frames = _frames(times, seed=7)
    packs = packs_from_frames(frames, latest_open=int(times[-1]), bar_sec=bar)
    decisions = times[window - 1: window - 1 + n_decisions]
    common = dict(
        decisions=decisions, timeframe="5m", window=window, bar_sec=bar,
        n_formulas=8, progress_every=0, min_ts_obs=20,
    )
    tanh = run_variant_from_packs(packs, variant="ts", objective="mean", position_mode="tanh", label="A", **common)
    again = run_variant_from_packs(packs, variant="ts", objective="mean", position_mode="tanh", label="A", **common)
    np.testing.assert_allclose(tanh["position"], again["position"])
    assert tanh["names"] == again["names"]
    cut = int(decisions[0])
    trimmed_frames = {s: frame.loc[frame["time"] <= cut].copy() for s, frame in frames.items()}
    trimmed = packs_from_frames(trimmed_frames, latest_open=cut, bar_sec=bar)
    prefix = run_variant_from_packs(
        trimmed,
        decisions=decisions[:1],
        timeframe="5m",
        variant="ts",
        objective="mean",
        position_mode="tanh",
        label="A",
        window=window,
        bar_sec=bar,
        n_formulas=8,
        progress_every=0,
        min_ts_obs=20,
    )
    np.testing.assert_allclose(prefix["position"][:, 0], tanh["position"][:, 0])
    assert prefix["names"][0][0] == tanh["names"][0][0]
    assert np.isnan(prefix["returns"][:, 0]).all()
    assert np.isfinite(tanh["returns"][:, 0]).all()

    rank = run_variant_from_packs(packs, variant="rank", objective="rank", position_mode="rank", label="B", **common)
    updated = rank["updated"][:, 0]
    assert updated.all()
    assert abs(rank["position"][:, 0].sum()) < 1e-8
    gross_tanh = np.abs(target_positions(rank["factor"][:, 0])).sum()
    assert abs(np.abs(rank["position"][:, 0]).sum() - gross_tanh) < 1e-8
    live = tanh["updated"][:, 0]
    assert live.all()
    np.testing.assert_allclose(tanh["position"][live, 0], target_positions(tanh["factor"][live, 0]))

    hole = int(decisions[1])
    gappy = {s: frame.copy() for s, frame in frames.items()}
    gappy["BBB"] = gappy["BBB"].loc[gappy["BBB"]["time"] != hole].copy()
    gap_packs = packs_from_frames(gappy, latest_open=int(times[-1]), bar_sec=bar)
    held = run_variant_from_packs(gap_packs, variant="ts", objective="mean", position_mode="tanh", label="A", **common)
    b = held["symbols"].index("BBB")
    assert np.isnan(held["returns"][b, 1])
    assert held["position"][b, 1] == held["position"][b, 0]
    assert not held["updated"][b, 1]
    tables = build_tables(held)
    assert list(tables["factor_target_cumsum"].columns) == [
        "time", "time_utc", "symbol", "factor_name", "segment", "factor",
        "target", "product", "cumsum", "oos_cumsum", "product_sum",
    ]
    assert list(tables["position_target_cumsum"].columns) == [
        "time", "time_utc", "symbol", "segment", "position", "target", "product", "cumsum", "oos_cumsum",
    ]
    portal = Path(__file__).resolve().parents[2] / "minute_alpha" / "portal_out_7d"
    for name, frame in (
        ("week_metrics.csv", tables["metrics"]),
        ("daily_portfolio_returns.csv", tables["daily"]),
        ("gaps.csv", tables["gaps"]),
        ("retrain_timing_history.csv", tables["timing_rows"]),
        ("factor_target_cumsum.csv", tables["factor_target_cumsum"]),
        ("position_target_cumsum.csv", tables["position_target_cumsum"]),
    ):
        expected = pd.read_csv(portal / name, nrows=0).columns.tolist()
        assert list(frame.columns) == expected, name
    b_rows = tables["gaps"]
    assert ((b_rows["symbol"] == "BBB") & (b_rows["reason"] == "missing_print")).any()


def test_commit_dir_refuses_existing_output(tmp_path: Path):
    dest = tmp_path / "portal_out_7d_5m_xs_ts"
    dest.mkdir()
    (dest / "week_metrics.csv").write_text("x", encoding="utf-8")
    src = tmp_path / "partial"
    src.mkdir()
    try:
        _commit_dir(src, dest)
    except FileExistsError:
        return
    raise AssertionError("existing directory was overwritten")
