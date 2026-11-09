"""Cross-sectional retrain on closed 5m and 15m Bitfinex candles.

The settlement week is the same clock the 1-minute study used
(``portal_out_7d``, last 1-minute open 2026-11-09 04:24:00 UTC). Candles are
the exchange ``trade:5m`` / ``trade:15m`` prints, not a resample of 1-minute
bars. Every closed bar is a retrain. The forward return of that bar is not
inside the training score.

Eight jobs run at once: 5m/15m crossed with

- ``ts``: one formula, score = mean of per-coin time-series Pearson IC,
  position = per-coin tanh
- ``ts_min``: same position, score = the worst coin's Pearson IC
- ``rank``: score = mean cross-sectional Spearman IC, position = demeaned
  rank scaled to the per-coin tanh gross
- ``rank_long``: the rank job with a training window three times longer

Missing settlement prices stay blank and do not flatten the book. A position
change on a blank bar is charged on the next bar that does have a return.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from minute_alpha.config import N_FORMULAS, PAIRS, SECONDS_PER_YEAR, SEED
from minute_alpha.backtest import max_drawdown, sharpe
from minute_alpha.features import compute_features
from minute_alpha.position import target_positions
from minute_alpha.search import make_candidates
from minute_alpha.vm import decode, execute

# portal_out_7d coverage: the last 1-minute open that study settled through.
ANCHOR_1M_OPEN = 1794198240
ROOT = Path(__file__).resolve().parent

# normalize_output zeros the first 499 bars when the window is at least 500.
WARMUP_ZERO = 499
MIN_TS_OBS = 30
MIN_RANK_NAMES = 3
MIN_DAILY_OBS = 8
MIN_WEEK_OBS = 20


def periods_per_year_bar(bar_sec: int) -> int:
    """Sharpe scale: ``mean / std(ddof=1) * sqrt(periods)``.

    ``periods`` is the number of bars in 365.25 days. 5m → 105192, 15m → 35064.
    """
    return int(round(SECONDS_PER_YEAR / float(bar_sec)))


def latest_closed_open(bar_sec: int, anchor: int = ANCHOR_1M_OPEN) -> int:
    """Open time of the last candle of this size that was already closed.

    The anchored 1-minute bar opens at ``anchor`` and closes 60 seconds later.
    A candle of ``bar_sec`` that opens at ``t`` closes at ``t + bar_sec``.
    """
    closed_by = int(anchor) + 60
    return int(((closed_by - int(bar_sec)) // int(bar_sec)) * int(bar_sec))


def week_bars(bar_sec: int) -> int:
    return 7 * 24 * 3600 // int(bar_sec)


def base_window(timeframe: str) -> int:
    return {"5m": 864, "15m": 288}[timeframe]


def bar_seconds(timeframe: str) -> int:
    return {"5m": 300, "15m": 900}[timeframe]


def decision_times(latest_open: int, bar_sec: int, n_decisions: int) -> np.ndarray:
    """Decision opens whose next-bar close is ``latest_open`` on the last step.

    ``n_decisions`` bars tile exactly ``n_decisions * bar_sec`` seconds, which
    is 7 days when ``n_decisions`` is ``week_bars``.
    """
    last = int(latest_open) - int(bar_sec)
    first = last - (int(n_decisions) - 1) * int(bar_sec)
    out = np.arange(first, last + int(bar_sec), int(bar_sec), dtype=np.int64)
    if out.size != int(n_decisions):
        raise RuntimeError(f"decision clock {out.size} != {n_decisions}")
    if int(out[-1]) + int(bar_sec) != int(latest_open):
        raise RuntimeError("last decision does not settle on latest_open")
    return out


def variant_specs() -> list[dict]:
    """The eight jobs. Directory names are the ones the study asked for."""
    order = (
        ("ts", "mean", "tanh", 1, "xs_ts", "A 时序 IC 均值"),
        ("ts_min", "min", "tanh", 1, "xs_ts_min", "A_min 最差币 IC"),
        ("rank", "rank", "rank", 1, "xs_rank", "B 截面 Rank IC"),
        ("rank_long", "rank", "rank", 3, "xs_rank_long", "B_long 截面 Rank IC，窗口 ×3"),
    )
    specs = []
    for timeframe in ("5m", "15m"):
        bar = bar_seconds(timeframe)
        for key, objective, mode, mult, suffix, label in order:
            specs.append({
                "key": f"{timeframe}_{key}",
                "timeframe": timeframe,
                "variant": key,
                "objective": objective,
                "position_mode": mode,
                "window_mult": mult,
                "window": base_window(timeframe) * mult,
                "bar_seconds": bar,
                "label": label,
                "dirname": f"portal_out_7d_{timeframe}_{suffix}",
            })
    return specs


def pick_best(scores: np.ndarray) -> int | None:
    """Highest finite score. An exact tie keeps the earlier formula."""
    best_i = None
    best_s = -np.inf
    for i, raw in enumerate(np.asarray(scores, dtype=np.float64).ravel()):
        if np.isfinite(raw) and float(raw) > best_s:
            best_s = float(raw)
            best_i = i
    return best_i


def aggregate_symbol_ics(stacked: np.ndarray, objective: str) -> np.ndarray:
    """``stacked`` is ``[n_symbols, n_formulas]`` of Pearson ICs, NaN if missing."""
    stacked = np.asarray(stacked, dtype=np.float64)
    n_form = stacked.shape[1]
    if objective == "mean":
        count = np.isfinite(stacked).sum(axis=0)
        total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
        out = np.full(n_form, np.nan)
        ok = count > 0
        out[ok] = total[ok] / count[ok]
        return out
    if objective == "min":
        masked = np.where(np.isfinite(stacked), stacked, np.inf)
        out = masked.min(axis=0).astype(np.float64)
        out[~np.isfinite(out)] = np.nan
        return out
    raise ValueError(f"unknown time-series objective {objective!r}")


def usable_mask(forward_ret: np.ndarray, window: int) -> np.ndarray:
    """Training pairs whose forward return is already known and not a warm-up zero."""
    mask = np.isfinite(np.asarray(forward_ret, dtype=np.float64))
    if int(window) >= 500:
        mask[:WARMUP_ZERO] = False
    return mask


def window_forward_returns(
    times: np.ndarray,
    close: np.ndarray,
    full_time: np.ndarray,
    full_close: np.ndarray,
    bar_sec: int,
    decision: int,
) -> np.ndarray:
    """``close[t + bar] / close[t] - 1`` only when that next open is already <= decision."""
    times = np.asarray(times, dtype=np.int64)
    close = np.asarray(close, dtype=np.float64)
    full_time = np.asarray(full_time, dtype=np.int64)
    full_close = np.asarray(full_close, dtype=np.float64)
    ret = np.full(times.shape, np.nan, dtype=np.float64)
    if full_time.size == 0 or times.size == 0:
        return ret
    nxt = times + int(bar_sec)
    pos = np.searchsorted(full_time, nxt)
    pos_clip = np.clip(pos, 0, full_time.size - 1)
    matched = (pos < full_time.size) & (full_time[pos_clip] == nxt) & (nxt <= int(decision))
    c1 = full_close[pos_clip]
    good = matched & np.isfinite(close) & np.isfinite(c1) & (close != 0.0)
    ret[good] = c1[good] / close[good] - 1.0
    return ret


def _pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson of each row of ``x`` with the shared ``y``. Both finite."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - float(y.mean())
    sa = np.sqrt((xc * xc).mean(axis=1))
    sb = float(np.sqrt((yc * yc).mean()))
    ic = (xc * yc).mean(axis=1) / (sa * sb + 1e-12)
    ic[(sa < 1e-8) | (sb < 1e-8)] = np.nan
    return ic


def _pearson_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    sx = np.sqrt((xc * xc).sum(axis=0))
    sy = np.sqrt((yc * yc).sum(axis=0))
    ic = (xc * yc).sum(axis=0) / (sx * sy + 1e-12)
    ic[(sx < 1e-8) | (sy < 1e-8)] = np.nan
    return ic


def _row_pearson(factors: np.ndarray, y: np.ndarray, mask: np.ndarray, min_obs: int) -> np.ndarray:
    factors = np.asarray(factors, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.full(factors.shape[0], np.nan)
    idx = np.flatnonzero(mask & np.isfinite(y))
    if idx.size < min_obs:
        return out
    x = factors[:, idx]
    yy = y[idx]
    good = np.isfinite(x).all(axis=1)
    if good.any():
        out[good] = _pearson_rows(x[good], yy)
    for i in np.flatnonzero(~good):
        m = np.isfinite(x[i])
        if int(m.sum()) < min_obs:
            continue
        out[i] = _pearson_rows(x[i : i + 1, m], yy[m])[0]
    return out


def _spearman_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = rankdata(x, axis=0, method="average")
    ry = rankdata(y, axis=0, method="average")
    return _pearson_cols(np.asarray(rx, dtype=np.float64), np.asarray(ry, dtype=np.float64))


def _spearman_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    value = _spearman_matrix(np.asarray(x, dtype=np.float64)[:, None], np.asarray(y, dtype=np.float64)[:, None])
    return float(value[0])


def cross_section_weights(alpha: np.ndarray, tanh_position: np.ndarray) -> np.ndarray:
    """Demeaned average-rank, scaled so the book is dollar-neutral and matches tanh gross.

    ``sum(w) = 0`` and ``sum(|w|) = sum(|tanh_position|)``. Names are not
    clipped back to 1; clipping would shrink the gross below the per-coin book.
    Fewer than two finite names, a flat rank, or a zero tanh gross returns zeros.
    """
    alpha = np.asarray(alpha, dtype=np.float64).ravel()
    tanh_position = np.asarray(tanh_position, dtype=np.float64).ravel()
    if alpha.shape != tanh_position.shape:
        raise ValueError("alpha and tanh position length differ")
    n = int(alpha.size)
    if n < 2 or not np.isfinite(alpha).all() or not np.isfinite(tanh_position).all():
        return np.zeros(n, dtype=np.float64)
    ranks = np.asarray(rankdata(alpha, method="average"), dtype=np.float64)
    demeaned = ranks - float(ranks.mean())
    l1 = float(np.abs(demeaned).sum())
    gross = float(np.abs(tanh_position).sum())
    if l1 < 1e-12 or gross == 0.0:
        return np.zeros(n, dtype=np.float64)
    return demeaned / l1 * gross


def settle_with_hold(position: np.ndarray, ret: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Book ``position * return - cost * turnover`` on bars that have a return.

    A NaN position carries the previous holding (starting from flat). A NaN
    return stays a NaN PnL: it is not a zero return. Turnover that happened
    on blank bars is added to the next booked bar, so the cost is still the
    actual traded amount and the blank return is not filled in.
    """
    pos_in = np.asarray(position, dtype=np.float64).ravel()
    ret = np.asarray(ret, dtype=np.float64).ravel()
    held = np.empty(pos_in.shape, dtype=np.float64)
    last = 0.0
    for i, value in enumerate(pos_in):
        if np.isfinite(value):
            last = float(value)
        held[i] = last
    pnl = np.full(held.shape, np.nan)
    turnover = np.full(held.shape, np.nan)
    prev = 0.0
    pending = 0.0
    for i in range(held.size):
        pending += abs(held[i] - prev)
        prev = held[i]
        if np.isfinite(ret[i]):
            pnl[i] = held[i] * ret[i] - pending * float(cost)
            turnover[i] = pending
            pending = 0.0
    return pnl, turnover, held


def _icir(values: list[float] | np.ndarray) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    sd = float(arr.std(ddof=1))
    if sd < 1e-12:
        return None
    return float(arr.mean() / sd)


def _pearson_pair(x: np.ndarray, y: np.ndarray, min_obs: int) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < min_obs:
        return None
    ic = _pearson_rows(x[m][None, :], y[m])[0]
    if not np.isfinite(ic):
        return None
    return float(ic)


def packs_from_frames(frames: dict[str, pd.DataFrame], latest_open: int, bar_sec: int) -> dict[str, dict]:
    """Keep closed prints at or before ``latest_open``. Off-grid stamps are dropped."""
    packs = {}
    for symbol, frame in frames.items():
        sub = frame.loc[frame["time"] <= int(latest_open)].copy()
        on_grid = (sub["time"].to_numpy(dtype=np.int64) % int(bar_sec)) == 0
        if not on_grid.all():
            print(f"{symbol} drop {int((~on_grid).sum())} off-grid {bar_sec}s bars", flush=True)
            sub = sub.loc[on_grid]
        sub = sub.sort_values("time").drop_duplicates("time")
        times = sub["time"].to_numpy(dtype=np.int64)
        pack = {
            "time": times,
            "index": {int(t): i for i, t in enumerate(times)},
        }
        for col in ("open", "high", "low", "close", "volume"):
            pack[col] = sub[col].to_numpy(dtype=np.float64)
        packs[symbol] = pack
    return packs


def _prepare_symbol(pack: dict, decision: int, window: int, bar_sec: int, candidates) -> dict | None:
    idx = pack["index"].get(int(decision))
    if idx is None or idx + 1 < window:
        return None
    sl = slice(idx + 1 - window, idx + 1)
    raw = {key: pack[key][sl][None, :] for key in ("open", "high", "low", "close", "volume")}
    t0 = time.process_time()
    feat = compute_features(raw)
    feature_s = time.process_time() - t0
    t1 = time.process_time()
    n_form = len(candidates)
    facs = np.empty((n_form, window), dtype=np.float64)
    ok = np.ones(n_form, dtype=bool)
    for fi, (_name, tokens) in enumerate(candidates):
        out = execute(tokens, feat)
        if out is None or int(out.shape[-1]) != window:
            ok[fi] = False
            facs[fi] = np.nan
        else:
            facs[fi] = np.asarray(out[0], dtype=np.float64)
    search_s = time.process_time() - t1
    times = pack["time"][sl]
    close = pack["close"][sl]
    ret = window_forward_returns(times, close, pack["time"], pack["close"], bar_sec, decision)
    return {
        "times": times,
        "facs": facs,
        "ok": ok,
        "ret": ret,
        "feature_seconds": feature_s,
        "execute_seconds": search_s,
    }


def _rank_scores(prepared: list[tuple[str, dict]], decision: int, min_names: int) -> np.ndarray:
    """Mean Spearman IC. One pass over time scores every formula together."""
    n_form = prepared[0][1]["facs"].shape[0]
    scores = np.full(n_form, np.nan)
    if len(prepared) < min_names:
        return scores
    times_union = np.unique(np.concatenate([
        np.asarray(item["times"], dtype=np.int64)
        for _symbol, item in prepared
    ]))
    times_union = times_union[times_union < int(decision)]
    if times_union.size == 0:
        return scores
    n_sym = len(prepared)
    n_t = int(times_union.size)
    alpha = np.full((n_sym, n_form, n_t), np.nan)
    ys = np.full((n_sym, n_t), np.nan)
    for si, (_symbol, item) in enumerate(prepared):
        ts = np.asarray(item["times"], dtype=np.int64)
        warm = WARMUP_ZERO if ts.size >= 500 else 0
        pos = np.searchsorted(times_union, ts)
        pos_clip = np.clip(pos, 0, max(n_t - 1, 0))
        keep = (pos < n_t) & (times_union[pos_clip] == ts)
        keep &= np.arange(ts.size) >= warm
        keep &= ts < int(decision)
        if not keep.any():
            continue
        cols = pos[keep]
        ys[si, cols] = item["ret"][keep]
        # Advanced indexing would transpose a (formula, time) block, so assign per bar.
        taken = np.asarray(item["facs"][:, keep], dtype=np.float64)
        for j, col in enumerate(cols):
            alpha[si, :, int(col)] = taken[:, j]
    sum_ic = np.zeros(n_form, dtype=np.float64)
    count = np.zeros(n_form, dtype=np.float64)
    for k in range(n_t):
        yk = ys[:, k]
        base = np.isfinite(yk)
        n_base = int(base.sum())
        if n_base < min_names:
            continue
        block = alpha[:, :, k]
        good = np.isfinite(block[base]).all(axis=0)
        if good.any():
            xb = np.asarray(block[base][:, good], dtype=np.float64)
            yb = np.asarray(yk[base], dtype=np.float64)
            rx = np.asarray(rankdata(xb, axis=0, method="average"), dtype=np.float64)
            ry = np.asarray(rankdata(yb, method="average"), dtype=np.float64)
            ic = _pearson_cols(rx, np.broadcast_to(ry[:, None], rx.shape))
            finite = np.isfinite(ic)
            sum_ic[good] += np.where(finite, ic, 0.0)
            count[good] += finite
        if (~good).any():
            for f in np.flatnonzero(~good):
                mask = base & np.isfinite(block[:, f])
                if int(mask.sum()) < min_names:
                    continue
                ic = _spearman_1d(block[mask, f], yk[mask])
                if np.isfinite(ic):
                    sum_ic[f] += ic
                    count[f] += 1.0
    ok = count > 0
    scores[ok] = sum_ic[ok] / count[ok]
    return scores


def _retrain_decision(
    packs: dict[str, dict],
    symbols: list[str],
    decision: int,
    *,
    window: int,
    bar_sec: int,
    candidates,
    objective: str,
    position_mode: str,
    min_ts_obs: int = MIN_TS_OBS,
    min_rank_names: int = MIN_RANK_NAMES,
) -> dict | None:
    """Fit one shared formula on data with time <= ``decision`` and map positions."""
    prepared: list[tuple[str, dict]] = []
    feature_s = 0.0
    execute_s = 0.0
    for symbol in symbols:
        item = _prepare_symbol(packs[symbol], decision, window, bar_sec, candidates)
        if item is None:
            continue
        feature_s += item["feature_seconds"]
        execute_s += item["execute_seconds"]
        prepared.append((symbol, item))
    if not prepared:
        return None
    t_score = time.process_time()
    if objective == "rank":
        scores = _rank_scores(prepared, decision, min_rank_names)
    else:
        rows = []
        for _symbol, item in prepared:
            mask = usable_mask(item["ret"], item["times"].size)
            ic = _row_pearson(item["facs"], item["ret"], mask, min_ts_obs)
            ic[~item["ok"]] = np.nan
            rows.append(ic)
        scores = aggregate_symbol_ics(np.vstack(rows), objective)
    score_s = time.process_time() - t_score
    best = pick_best(scores)
    if best is None:
        return {
            "name": "",
            "formula": "",
            "train_ic": None,
            "positions": {},
            "factors": {},
            "eligible": [symbol for symbol, _item in prepared],
            "feature_seconds": feature_s,
            "search_seconds": execute_s + score_s,
            "inference_seconds": 0.0,
            "gross": 0.0,
            "n_symbols": len(prepared),
        }
    t_inf = time.process_time()
    fit_symbols = []
    alphas = []
    for symbol, item in prepared:
        if not item["ok"][best]:
            continue
        alpha = float(item["facs"][best, -1])
        if not np.isfinite(alpha):
            continue
        fit_symbols.append(symbol)
        alphas.append(alpha)
    alpha_arr = np.asarray(alphas, dtype=np.float64)
    tanh = target_positions(alpha_arr) if alpha_arr.size else np.zeros(0)
    positions: dict[str, float] = {}
    factors: dict[str, float] = {}
    if position_mode == "tanh":
        for symbol, alpha, pos in zip(fit_symbols, alphas, tanh):
            positions[symbol] = float(pos)
            factors[symbol] = float(alpha)
    elif position_mode == "rank":
        # A one-name cross section has no long-short book. Hold the previous book.
        if len(fit_symbols) >= 2:
            weights = cross_section_weights(alpha_arr, tanh)
            for symbol, alpha, weight in zip(fit_symbols, alphas, weights):
                positions[symbol] = float(weight)
                factors[symbol] = float(alpha)
    else:
        raise ValueError(f"unknown position mode {position_mode!r}")
    inference_s = time.process_time() - t_inf
    gross = float(np.abs(tanh).sum()) if tanh.size else 0.0
    if position_mode == "rank" and positions:
        gross = float(sum(abs(v) for v in positions.values()))
    return {
        "name": candidates[best][0],
        "formula": decode(candidates[best][1]),
        "train_ic": float(scores[best]),
        "positions": positions,
        "factors": factors,
        "eligible": [symbol for symbol, _item in prepared],
        "feature_seconds": feature_s,
        "search_seconds": execute_s + score_s,
        "inference_seconds": inference_s,
        "gross": gross,
        "n_symbols": len(fit_symbols),
    }


def _settlement_returns(packs, symbols, decisions, bar_sec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = len(symbols), len(decisions)
    returns = np.full((n, m), np.nan)
    has_bar = np.zeros((n, m), dtype=bool)
    has_next = np.zeros((n, m), dtype=bool)
    for i, symbol in enumerate(symbols):
        pack = packs[symbol]
        index = pack["index"]
        close = pack["close"]
        for k, decision in enumerate(decisions):
            i0 = index.get(int(decision))
            i1 = index.get(int(decision) + int(bar_sec))
            has_bar[i, k] = i0 is not None
            has_next[i, k] = i1 is not None
            if i0 is None or i1 is None:
                continue
            c0 = close[i0]
            c1 = close[i1]
            if np.isfinite(c0) and np.isfinite(c1) and c0 != 0.0:
                returns[i, k] = float(c1 / c0 - 1.0)
    return returns, has_bar, has_next


def run_variant_from_packs(
    packs: dict[str, dict],
    *,
    decisions: np.ndarray,
    timeframe: str,
    variant: str,
    objective: str,
    position_mode: str,
    window: int,
    bar_sec: int,
    label: str = "",
    n_formulas: int = N_FORMULAS,
    min_ts_obs: int = MIN_TS_OBS,
    min_rank_names: int = MIN_RANK_NAMES,
    progress_every: int = 100,
) -> dict:
    symbols = list(packs)
    decisions = np.asarray(decisions, dtype=np.int64)
    candidates = make_candidates(n_formulas, SEED)
    returns, has_bar, has_next = _settlement_returns(packs, symbols, decisions, bar_sec)
    n, m = len(symbols), len(decisions)
    position = np.zeros((n, m), dtype=np.float64)
    factor = np.full((n, m), np.nan)
    updated = np.zeros((n, m), dtype=bool)
    eligible = np.zeros((n, m), dtype=bool)
    names = [[""] * m for _ in range(n)]
    held_pos = {s: 0.0 for s in symbols}
    held_fac = {s: np.nan for s in symbols}
    held_name = {s: "" for s in symbols}
    timing_rows = []
    selection_rows = []
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    for k, decision in enumerate(decisions):
        step_wall = time.perf_counter()
        rec = _retrain_decision(
            packs, symbols, int(decision),
            window=window, bar_sec=bar_sec, candidates=candidates,
            objective=objective, position_mode=position_mode,
            min_ts_obs=min_ts_obs, min_rank_names=min_rank_names,
        )
        step_wall_s = time.perf_counter() - step_wall
        if rec is not None and rec["positions"]:
            for symbol, pos in rec["positions"].items():
                held_pos[symbol] = pos
                held_fac[symbol] = rec["factors"][symbol]
                held_name[symbol] = rec["name"]
        if rec is not None and (rec["name"] or rec["feature_seconds"] + rec["search_seconds"] > 0):
            total = rec["feature_seconds"] + rec["search_seconds"] + rec["inference_seconds"]
            timing_rows.append({
                "symbol": "ALL",
                "decision_unix": int(decision),
                "decision_utc": _utc(int(decision)),
                "total_seconds": total,
                "feature_seconds": rec["feature_seconds"],
                "search_seconds": rec["search_seconds"],
                "inference_seconds": rec["inference_seconds"],
                "champion": rec["name"],
            })
        if rec is not None and rec["name"]:
            total = rec["feature_seconds"] + rec["search_seconds"] + rec["inference_seconds"]
            selection_rows.append({
                "decision_unix": int(decision),
                "decision_utc": _utc(int(decision)),
                "champion": rec["name"],
                "formula": rec["formula"],
                "train_ic": rec["train_ic"],
                "n_symbols": rec["n_symbols"],
                "gross": rec["gross"],
                "cpu_seconds": total,
                "wall_seconds": step_wall_s,
            })
        for i, symbol in enumerate(symbols):
            position[i, k] = held_pos[symbol]
            factor[i, k] = held_fac[symbol]
            names[i][k] = held_name[symbol]
            updated[i, k] = rec is not None and symbol in rec["positions"]
            eligible[i, k] = rec is not None and symbol in rec["eligible"]
        if progress_every and (k % progress_every == 0 or k + 1 == m):
            last = timing_rows[-1]["total_seconds"] if timing_rows else 0.0
            print(
                f"{timeframe} {variant} bar {k + 1}/{m} last_cpu {last:.3f}s",
                flush=True,
            )
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return {
        "symbols": symbols,
        "decisions": decisions,
        "position": position,
        "factor": factor,
        "returns": returns,
        "names": names,
        "updated": updated,
        "eligible": eligible,
        "has_bar": has_bar,
        "has_next": has_next,
        "timing_rows": timing_rows,
        "selection_rows": selection_rows,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "periods": periods_per_year_bar(bar_sec),
        "bar_seconds": int(bar_sec),
        "window": int(window),
        "objective": objective,
        "position_mode": position_mode,
        "timeframe": timeframe,
        "variant": variant,
        "label": label,
        "n_formulas": int(n_formulas),
        "seed": SEED,
    }


def _utc(stamp: int) -> str:
    from minute_alpha.diagnostics import time_utc
    return time_utc(int(stamp))


def _metrics_of(pnl: np.ndarray, turnover: np.ndarray, periods: int) -> dict:
    pnl = np.asarray(pnl, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)
    if pnl.size == 0:
        equity = np.array([1.0])
    else:
        equity = np.cumprod(1.0 + pnl)
        equity = np.where(np.isfinite(equity), equity, 1.0)
    return {
        "return": float(equity[-1] - 1.0) if pnl.size else 0.0,
        "sharpe": sharpe(pnl, periods) if pnl.size else 0.0,
        "mdd": max_drawdown(equity),
        "turnover_mean": float(turnover.mean()) if turnover.size else 0.0,
    }


def _nanmean_cols(values: np.ndarray) -> np.ndarray:
    count = np.isfinite(values).sum(axis=0)
    total = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=0)
    out = np.full(values.shape[1], np.nan)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out


def _metric_line(symbol, pnl_1, to_1, pnl_0, bh_1, bh_0, periods, gap_count, valid_count) -> dict:
    m1 = _metrics_of(pnl_1, to_1, periods)
    m0 = _metrics_of(pnl_0, np.zeros(pnl_0.size, dtype=np.float64), periods)
    b1 = _metrics_of(bh_1, np.zeros(bh_1.size, dtype=np.float64), periods)
    b0 = _metrics_of(bh_0, np.zeros(bh_0.size, dtype=np.float64), periods)
    flat = _metrics_of(np.zeros(valid_count), np.zeros(valid_count), periods)
    return {
        "symbol": symbol,
        "return_1bp": m1["return"],
        "sharpe_1bp": m1["sharpe"],
        "mdd_1bp": m1["mdd"],
        "turnover_1bp": m1["turnover_mean"],
        "return_0cost": m0["return"],
        "sharpe_0cost": m0["sharpe"],
        "mdd_0cost": m0["mdd"],
        "turnover_0cost": m0["turnover_mean"],
        "buyhold_return_1bp": b1["return"],
        "buyhold_sharpe_1bp": b1["sharpe"],
        "buyhold_mdd_1bp": b1["mdd"],
        "buyhold_return_0cost": b0["return"],
        "buyhold_sharpe_0cost": b0["sharpe"],
        "buyhold_mdd_0cost": b0["mdd"],
        "flat_return": flat["return"],
        "valid_minutes": valid_count,
        "gap_minutes": gap_count,
    }


def _daily(decisions, port_1, port_0, port_bh) -> pd.DataFrame:
    frame = pd.DataFrame({
        "time": np.asarray(decisions, dtype=np.int64),
        "pnl_1bp": port_1,
        "pnl_0cost": port_0,
        "pnl_buyhold_0cost": port_bh,
    })
    frame = frame[np.isfinite(frame["pnl_1bp"])].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "return_1bp", "return_0cost", "buyhold_return_0cost", "minutes"])
    stamps = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["date"] = stamps.dt.strftime("%Y-%m-%d")
    rows = []
    for date, sub in frame.groupby("date", sort=True):
        rows.append({
            "date": date,
            "return_1bp": float(np.prod(1.0 + sub["pnl_1bp"].to_numpy()) - 1.0),
            "return_0cost": float(np.prod(1.0 + sub["pnl_0cost"].to_numpy()) - 1.0),
            "buyhold_return_0cost": float(np.prod(1.0 + sub["pnl_buyhold_0cost"].to_numpy()) - 1.0),
            "minutes": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _bar_rank_ic(factor: np.ndarray, ret: np.ndarray, min_names: int) -> np.ndarray:
    _n, m = factor.shape
    out = np.full(m, np.nan)
    for k in range(m):
        mask = np.isfinite(factor[:, k]) & np.isfinite(ret[:, k])
        if int(mask.sum()) < min_names:
            continue
        out[k] = _spearman_1d(factor[mask, k], ret[mask, k])
    return out


def _oos_ic(result: dict) -> dict:
    symbols = result["symbols"]
    decisions = result["decisions"]
    factor = result["factor"]
    ret = result["returns"]
    dates = pd.to_datetime(decisions, unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()
    per_symbol = []
    daily_means = []
    unique_dates = list(dict.fromkeys(dates.tolist()))
    for i, symbol in enumerate(symbols):
        ic = _pearson_pair(factor[i], ret[i], MIN_WEEK_OBS)
        day_ics = []
        for date in unique_dates:
            m = dates == date
            day_ic = _pearson_pair(factor[i, m], ret[i, m], MIN_DAILY_OBS)
            if day_ic is not None:
                day_ics.append(day_ic)
        per_symbol.append({
            "symbol": symbol,
            "oos_pearson_ic": ic,
            "oos_daily_icir": _icir(day_ics),
            "n_obs": int((np.isfinite(factor[i]) & np.isfinite(ret[i])).sum()),
            "n_days": len(day_ics),
        })
    for date in unique_dates:
        m = dates == date
        coins = []
        for i in range(len(symbols)):
            day_ic = _pearson_pair(factor[i, m], ret[i, m], MIN_DAILY_OBS)
            if day_ic is not None:
                coins.append(day_ic)
        if coins:
            daily_means.append(float(np.mean(coins)))
    defined = [row["oos_pearson_ic"] for row in per_symbol if row["oos_pearson_ic"] is not None]
    rank_ic = _bar_rank_ic(factor, ret, MIN_RANK_NAMES)
    rank_daily = []
    for date in unique_dates:
        chunk = rank_ic[dates == date]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            rank_daily.append(float(chunk.mean()))
    finite_rank = rank_ic[np.isfinite(rank_ic)]
    ts_mean = float(np.mean(defined)) if defined else None
    ts_icir = _icir(daily_means)
    rank_mean = float(finite_rank.mean()) if finite_rank.size else None
    rank_icir_daily = _icir(rank_daily)
    rank_icir_bar = _icir(finite_rank)
    if result["objective"] == "rank":
        headline_ic = rank_mean
        headline_icir = rank_icir_daily
        headline = "截面 Rank IC：逐 bar Spearman 的均值；ICIR 是按 UTC 日平均后再 mean/std(ddof=1)，不年化"
    else:
        headline_ic = ts_mean
        headline_icir = ts_icir
        headline = "时序 Pearson IC：各币整周 IC 的平均；ICIR 是各币当日 IC 再跨币平均后的 mean/std(ddof=1)，不年化"
    return {
        "per_symbol": per_symbol,
        "headline_ic": headline_ic,
        "headline_icir": headline_icir,
        "headline": headline,
        "ts_ic_mean": ts_mean,
        "ts_icir_daily": ts_icir,
        "rank_ic_mean": rank_mean,
        "rank_icir_daily": rank_icir_daily,
        "rank_icir_bar": rank_icir_bar,
        "rank_ic_bars": int(finite_rank.size),
        "daily_ts": daily_means,
        "daily_rank": rank_daily,
    }


def _coverage(packs: dict[str, dict], decisions: np.ndarray, bar_sec: int) -> list[dict]:
    slots = [int(t) for t in decisions] + [int(decisions[-1]) + int(bar_sec)]
    rows = []
    for symbol, pack in packs.items():
        have = set(int(t) for t in pack["time"])
        missing = sum(1 for t in slots if t not in have)
        rows.append({
            "symbol": symbol,
            "first": int(pack["time"][0]) if pack["time"].size else None,
            "last": int(pack["time"][-1]) if pack["time"].size else None,
            "prints": int(pack["time"].size),
            "missing_week_slots": int(missing),
            "week_slots": len(slots),
        })
    return rows


def build_tables(result: dict) -> dict:
    symbols = result["symbols"]
    decisions = result["decisions"]
    periods = int(result["periods"])
    n, m = len(symbols), len(decisions)
    position = result["position"]
    factor = result["factor"]
    ret = result["returns"]
    names = result["names"]
    pnl_1 = np.full((n, m), np.nan)
    pnl_0 = np.full((n, m), np.nan)
    turn = np.full((n, m), np.nan)
    bh_1 = np.full((n, m), np.nan)
    bh_0 = np.full((n, m), np.nan)
    metric_rows = []
    for i, symbol in enumerate(symbols):
        pnl_1[i], turn[i], _held = settle_with_hold(position[i], ret[i], 0.0001)
        pnl_0[i], _to0, _held = settle_with_hold(position[i], ret[i], 0.0)
        ones = np.ones(m, dtype=np.float64)
        bh_1[i], _bto, _held = settle_with_hold(ones, ret[i], 0.0001)
        bh_0[i], _bto0, _held = settle_with_hold(ones, ret[i], 0.0)
        valid = np.isfinite(pnl_1[i])
        metric_rows.append(_metric_line(
            symbol,
            pnl_1[i, valid], turn[i, valid],
            pnl_0[i, valid],
            bh_1[i, np.isfinite(bh_1[i])],
            bh_0[i, np.isfinite(bh_0[i])],
            periods,
            gap_count=int((~np.isfinite(ret[i])).sum()),
            valid_count=int(valid.sum()),
        ))
    port_1 = _nanmean_cols(pnl_1)
    port_0 = _nanmean_cols(pnl_0)
    port_to = _nanmean_cols(turn)
    port_bh1 = _nanmean_cols(bh_1)
    port_bh0 = _nanmean_cols(bh_0)
    port_valid = np.isfinite(port_1)
    metric_rows.append(_metric_line(
        "ALL",
        port_1[port_valid], port_to[port_valid],
        port_0[port_valid],
        port_bh1[np.isfinite(port_bh1)],
        port_bh0[np.isfinite(port_bh0)],
        periods,
        gap_count=int((~port_valid).sum()),
        valid_count=int(port_valid.sum()),
    ))
    fac_rows = []
    pos_rows = []
    fac_csum = np.zeros(n)
    pos_csum = np.zeros(n)
    fac_seen = np.zeros(n, dtype=bool)
    pos_seen = np.zeros(n, dtype=bool)
    all_fac_csum = 0.0
    all_pos_csum = 0.0
    all_fac_seen = False
    all_pos_seen = False
    all_fac_path = np.full(m, np.nan)
    all_pos_path = np.full(m, np.nan)
    gap_rows = []
    for k, decision in enumerate(decisions):
        products_f = []
        products_p = []
        for i, symbol in enumerate(symbols):
            gap = not np.isfinite(ret[i, k])
            if gap:
                product_f = np.nan
                product_p = np.nan
                f_show = np.nan if not fac_seen[i] else fac_csum[i]
                p_show = np.nan if not pos_seen[i] else pos_csum[i]
                if not result["has_bar"][i, k]:
                    reason = "missing_print"
                else:
                    reason = "missing_next_close"
                gap_rows.append({
                    "time": int(decision),
                    "time_utc": _utc(int(decision)),
                    "symbol": symbol,
                    "reason": reason,
                })
            else:
                if np.isfinite(factor[i, k]):
                    product_f = float(factor[i, k] * ret[i, k])
                    fac_csum[i] += product_f
                    fac_seen[i] = True
                    f_show = fac_csum[i]
                    products_f.append(product_f)
                else:
                    product_f = np.nan
                    f_show = np.nan if not fac_seen[i] else fac_csum[i]
                product_p = float(position[i, k] * ret[i, k])
                pos_csum[i] += product_p
                pos_seen[i] = True
                p_show = pos_csum[i]
                products_p.append(product_p)
            stamp = int(decision)
            fac_rows.append({
                "time": stamp,
                "time_utc": _utc(stamp),
                "symbol": symbol,
                "factor_name": names[i][k],
                "segment": "oos",
                "factor": factor[i, k],
                "target": ret[i, k],
                "product": product_f,
                "cumsum": f_show,
                "oos_cumsum": f_show,
                "product_sum": np.nan,
            })
            pos_rows.append({
                "time": stamp,
                "time_utc": _utc(stamp),
                "symbol": symbol,
                "segment": "oos",
                "position": position[i, k],
                "target": ret[i, k],
                "product": product_p,
                "cumsum": p_show,
                "oos_cumsum": p_show,
            })
        if products_f:
            mean_f = float(np.mean(products_f))
            all_fac_csum += mean_f
            all_fac_seen = True
            show_f = all_fac_csum
        else:
            mean_f = np.nan
            show_f = all_fac_csum if all_fac_seen else np.nan
        if products_p:
            mean_p = float(np.mean(products_p))
            all_pos_csum += mean_p
            all_pos_seen = True
            show_p = all_pos_csum
        else:
            mean_p = np.nan
            show_p = all_pos_csum if all_pos_seen else np.nan
        all_fac_path[k] = show_f
        all_pos_path[k] = show_p
        stamp = int(decision)
        fac_rows.append({
            "time": stamp,
            "time_utc": _utc(stamp),
            "symbol": "ALL",
            "factor_name": "equal_weight_mean",
            "segment": "oos",
            "factor": np.nan,
            "target": np.nan,
            "product": mean_f,
            "cumsum": show_f,
            "oos_cumsum": show_f,
            "product_sum": float(np.sum(products_f)) if products_f else np.nan,
        })
        pos_rows.append({
            "time": stamp,
            "time_utc": _utc(stamp),
            "symbol": "ALL",
            "segment": "oos",
            "position": float(np.nanmean(position[:, k])),
            "target": np.nan,
            "product": mean_p,
            "cumsum": show_p,
            "oos_cumsum": show_p,
        })
    ic = _oos_ic(result)
    totals = np.array([row["total_seconds"] for row in result["timing_rows"]], dtype=np.float64)
    walls = np.array([row["wall_seconds"] for row in result["selection_rows"]], dtype=np.float64)
    train_ics = np.array([
        row["train_ic"] for row in result["selection_rows"] if row["train_ic"] is not None
    ], dtype=np.float64)
    insufficient = int((result["has_bar"] & result["has_next"] & ~result["eligible"]).sum())
    undefined = int((result["eligible"] & ~result["updated"] & result["has_next"]).sum())
    timing = {
        "retrain_count": int(totals.size),
        "retrain_seconds_median": float(np.median(totals)) if totals.size else None,
        "retrain_seconds_p95": float(np.percentile(totals, 95)) if totals.size else None,
        "retrain_seconds_mean": float(totals.mean()) if totals.size else None,
        "retrain_cpu_seconds_median": float(np.median(totals)) if totals.size else None,
        "retrain_cpu_seconds_p95": float(np.percentile(totals, 95)) if totals.size else None,
        "retrain_wall_seconds_median": float(np.median(walls)) if walls.size else None,
        "retrain_wall_seconds_p95": float(np.percentile(walls, 95)) if walls.size else None,
        "wall_seconds": float(result["wall_seconds"]),
        "cpu_seconds": float(result["cpu_seconds"]),
        "serial_estimate_seconds": float(totals.sum()) if totals.size else 0.0,
        "gap_rows": int(len(gap_rows)),
        "insufficient_history_rows": insufficient,
        "cross_section_undefined_rows": undefined,
        "train_ic_median": float(np.median(train_ics)) if train_ics.size else None,
        "periods_per_year": periods,
        "bar_seconds": int(result["bar_seconds"]),
        "window": int(result["window"]),
        "objective": result["objective"],
        "position_mode": result["position_mode"],
        "timeframe": result["timeframe"],
        "variant": result["variant"],
        "label": result["label"],
        "n_formulas": int(result["n_formulas"]),
        "seed": int(result["seed"]),
        "anchor_1m_open": ANCHOR_1M_OPEN,
        "first_decision": int(decisions[0]),
        "last_decision": int(decisions[-1]),
        "latest_open": int(decisions[-1]) + int(result["bar_seconds"]),
        "oos_ic_mean": ic["headline_ic"],
        "oos_icir": ic["headline_icir"],
        "oos_ic_note": ic["headline"],
        "ts_ic_mean": ic["ts_ic_mean"],
        "ts_icir_daily": ic["ts_icir_daily"],
        "rank_ic_mean": ic["rank_ic_mean"],
        "rank_icir_daily": ic["rank_icir_daily"],
        "rank_icir_bar": ic["rank_icir_bar"],
        "rank_ic_bars": ic["rank_ic_bars"],
        "data": "bitfinex trade candles, not a 1-minute aggregation",
        "sharpe": (
            f"mean/std(ddof=1)*sqrt({periods}); "
            f"{periods} = round(365.25*86400/{int(result['bar_seconds'])})"
        ),
    }
    return {
        "metrics": pd.DataFrame(metric_rows),
        "factor_target_cumsum": pd.DataFrame(fac_rows),
        "position_target_cumsum": pd.DataFrame(pos_rows),
        "gaps": pd.DataFrame(gap_rows, columns=["time", "time_utc", "symbol", "reason"]),
        "timing_rows": pd.DataFrame(result["timing_rows"], columns=[
            "symbol", "decision_unix", "decision_utc", "total_seconds",
            "feature_seconds", "search_seconds", "inference_seconds", "champion",
        ]),
        "selection": pd.DataFrame(result["selection_rows"], columns=[
            "decision_unix", "decision_utc", "champion", "formula", "train_ic",
            "n_symbols", "gross", "cpu_seconds", "wall_seconds",
        ]),
        "daily": _daily(decisions, port_1, port_0, port_bh0),
        "symbol_ic": pd.DataFrame(ic["per_symbol"]),
        "timing": timing,
        "ic": ic,
        "port_1": port_1,
        "port_0": port_0,
        "port_bh0": port_bh0,
        "decisions": decisions,
        "all_pos_cumsum": all_pos_path,
        "all_fac_cumsum": all_fac_path,
    }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.4%}"


def _num(value, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _secs(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    value = float(value)
    if value >= 60:
        return f"{value:.2f} 秒（{value / 60:.1f} 分钟）"
    return f"{value:.3f} 秒"


def _method_lines() -> list[str]:
    return [
        "K 线直接取 Bitfinex 公开 `trade:5m` / `trade:15m` candle（`https://api-pub.bitfinex.com/v2/candles/trade:{tf}:{pair}/hist`），"
        "字段是 `[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]`，MTS 是 bar 开盘。只保留已经闭合的 bar。这不是 1 分钟线聚合。"
        "portal_out_7d 的 1 分钟序列缺口很多，聚合成 5m/15m 会把没成交的分钟收成假的 OHLC；"
        "长窗口还要大约 16 天历史，超出那份 1 分钟缓存。",
        "",
        "结算锚点是 portal_out_7d 最后一根 1 分钟的开盘 2026-11-09 04:24:00 UTC（unix 1794198240）。"
        "这根 1 分钟在 04:25:00 闭合。因此最后一根可用 5 分钟开盘是 04:20（它的收盘价大约在 04:25），"
        "最后一根可用 15 分钟开盘是 04:00（收盘价大约在 04:15，因为 04:15 开盘的 15 分钟要到 04:30 才闭合）。",
        "5 分钟的 2016 根收益正好铺满 2026-11-02 04:25 到 2026-11-09 04:25，和 1 分钟研究是同一段 7 天。"
        "15 分钟的 672 根铺满 2026-11-02 04:15 到 2026-11-09 04:15，同样是 7 天，整体早 10 分钟。",
        "",
        "每根 bar 收盘重训。训练窗是该币截至决策 bar 的最后 W 根真实 candle：5m 为 864（3 天），15m 为 288（3 天），"
        "B_long 再乘 3（5m 2592，15m 864）。公式池是种子 20261109 的 32 个候选，和分钟版相同，7 个币共用选出的那一条。",
        "训练 IC 只用「下一根收益的收盘时刻 ≤ 决策时刻」的配对。决策 bar 自己的下一根收益不进训练。",
        f"窗口长度 ≥ 500 时，生产 `normalize_output` 会把前 {WARMUP_ZERO} 根置 0。这 {WARMUP_ZERO} 根不进选式 IC，"
        "否则 5m 的 864 根里有一半是人为的 0。15m 的 288 根走扩张 z-score，除最后一根外都参加。仓位仍用完整窗口算出来的最后一根。",
        "",
        "变体 A：选式目标是各币时序 Pearson IC 的平均。A_min 改成这些 IC 里最差的一个（最大化最差币）。"
        "仓位是各币自己的生产映射：`tanh(z)` 后把绝对值小于 0.05 的置 0，再截到 [-1, 1]。"
        "z 是 N=1 的时序标准化，不是截面 z-score。",
        "变体 B：每个历史时点，用当时同时有因子和下一根收益的币做 Spearman，再对时间取平均。有效币少于 3 个的时点不进平均。"
        "仓位是这些币 alpha 的平均秩、去掉均值，再按 L1 缩放到 G = Σ|逐币 tanh 仓位|。"
        "所以 Σw = 0，Σ|w| = G，总 gross 和同一条公式的逐币版一样。单币不截断回 1，否则总 gross 会变小。"
        "7 个币、秩没有并列时，最大单币权重是 0.25×G。有效币少于 2 个就不重配，继续持有上一档仓位。",
        "截面 IC 和截面仓位都只用该时点有效的币。",
        "",
        "成本两档：零成本，以及 0 手续费 + 1bp 滑点。滑点乘的是实际换手 |Δ仓位|。"
        "缺成交的 bar 收益留空，不当成 0，也不把仓位打回 0。空白 bar 上如果已经换过仓，这笔换手记到下一根有收益的 bar 上扣除。",
        "组合收益是当根有结算的币的等权平均，和 portal_out_7d 一样。",
        "Sharpe = 收益均值 / 样本标准差(ddof=1) × sqrt(每年 bar 数)。"
        "5m 的系数是 sqrt(105192)，15m 是 sqrt(35064)，也就是 365.25 天 × 每天 288 或 96 根。MDD 与 portal_out_7d 相同，净值从第一根有结算的 bar 连乘。",
        "样本外 IC 是拼接后的持仓因子（每根 bar 用当时选中的公式）对已经实现的下一根收益。"
        "时序 ICIR 用 UTC 日频 IC 的 mean/std，不另做年化。截面另外给出逐 bar Rank IC 的 ICIR；总表里的 ICIR 用日频，这样 5m 和 15m 可以放在一起看。",
        "重训中位数和 P95 是单次重训的 CPU 时间（`process_time`，8 个进程抢核时不会被抬高）。"
        "墙钟是该变体进程从第一根跑到最后一根的实际流逝时间，含争用。总墙钟是 8 个进程一起跑完的时间。",
    ]


def _metrics_table(metrics: pd.DataFrame) -> list[str]:
    lines = [
        "| 币 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有 | 有效 bar | 缺口 bar |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['symbol']} | {_pct(row['return_1bp'])} | {_num(row['sharpe_1bp'])} | {_pct(row['mdd_1bp'])} | "
            f"{_num(row['turnover_1bp'], 4)} | {_pct(row['return_0cost'])} | {_num(row['sharpe_0cost'])} | "
            f"{_pct(row['mdd_0cost'])} | {_pct(row['buyhold_return_0cost'])} | "
            f"{int(row['valid_minutes'])} | {int(row['gap_minutes'])} |"
        )
    return lines


def _daily_table(daily: pd.DataFrame) -> list[str]:
    lines = [
        "| 日期 | 1bp | 0 成本 | 买入持有 | bar 数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if daily is None or daily.empty:
        lines.append("| — | — | — | — | 0 |")
        return lines
    for _, row in daily.iterrows():
        lines.append(
            f"| {row['date']} | {_pct(row['return_1bp'])} | {_pct(row['return_0cost'])} | "
            f"{_pct(row['buyhold_return_0cost'])} | {int(row['minutes'])} |"
        )
    return lines


def _ic_table(symbol_ic: pd.DataFrame) -> list[str]:
    lines = [
        "| 币 | 样本外 Pearson IC | 日频 ICIR | 有效配对 |",
        "| --- | --- | --- | --- |",
    ]
    if symbol_ic is None or symbol_ic.empty:
        return lines
    for _, row in symbol_ic.iterrows():
        lines.append(
            f"| {row['symbol']} | {_num(row['oos_pearson_ic'], 4)} | {_num(row['oos_daily_icir'], 3)} | {int(row['n_obs'])} |"
        )
    return lines


def _champion_lines(selection: pd.DataFrame) -> list[str]:
    if selection is None or selection.empty:
        return ["没有选出公式。"]
    counts = selection["champion"].value_counts()
    lines = ["选中次数最多的公式：", ""]
    lines.append("| 公式 | 次数 | 占比 |")
    lines.append("| --- | --- | --- |")
    total = int(counts.sum())
    for name, count in counts.head(8).items():
        lines.append(f"| {name} | {int(count)} | {count / total:.1%} |")
    top = counts.index[0]
    sample = selection.loc[selection["champion"] == top, "formula"]
    if not sample.empty:
        lines.append("")
        lines.append(f"`{top}` 的代币序列：`{sample.iloc[0]}`。")
    return lines


def _report_variant(tables: dict) -> str:
    timing = tables["timing"]
    ic = tables["ic"]
    lines = [
        f"# {timing['timeframe']} {timing['label']}",
        "",
        *_method_lines(),
        "",
        f"本目录是 {timing['timeframe']}、窗口 {timing['window']} 根、目标 `{timing['objective']}`、仓位 `{timing['position_mode']}`。",
        f"决策从 {_utc(timing['first_decision'])} 到 {_utc(timing['last_decision'])}，"
        f"最后一根结算开盘是 {_utc(timing['latest_open'])}。",
        f"Sharpe 年化系数 sqrt({timing['periods_per_year']})。{timing['sharpe']}。",
        "",
        f"重训 {timing['retrain_count']} 次。CPU 中位数 {_secs(timing['retrain_cpu_seconds_median'])}，"
        f"P95 {_secs(timing['retrain_cpu_seconds_p95'])}。"
        f"单次墙钟中位数 {_secs(timing['retrain_wall_seconds_median'])}，"
        f"P95 {_secs(timing['retrain_wall_seconds_p95'])}。",
        f"本变体进程墙钟 {_secs(timing['wall_seconds'])}，CPU 合计 {_secs(timing['serial_estimate_seconds'])}。",
        f"收益留空的行数 {timing['gap_rows']}。历史不够而保持原仓位的币-bar 数 {timing['insufficient_history_rows']}。"
        f"有历史但这一根没有新仓位（截面不足两个币，或该币没算出因子）的币-bar 数 {timing['cross_section_undefined_rows']}。",
        "",
        f"样本外 IC 均值 {_num(ic['headline_ic'], 4)}，日频 ICIR {_num(ic['headline_icir'], 3)}。{ic['headline']}。",
        f"作为对照：各币 Pearson IC 均值 {_num(ic['ts_ic_mean'], 4)}（日频 ICIR {_num(ic['ts_icir_daily'], 3)}）；"
        f"逐 bar Rank IC 均值 {_num(ic['rank_ic_mean'], 4)}（日频 ICIR {_num(ic['rank_icir_daily'], 3)}，"
        f"逐 bar ICIR {_num(ic['rank_icir_bar'], 3)}，有效 bar {ic['rank_ic_bars']}）。",
        f"选中公式的训练 IC 中位数 {_num(timing.get('train_ic_median'), 4)}。",
        "",
        "## 各币",
        "",
        *_metrics_table(tables["metrics"]),
        "",
        "## 分天组合收益",
        "",
        *_daily_table(tables["daily"]),
        "",
        "## 选中公式的样本外 IC",
        "",
        *_ic_table(tables["symbol_ic"]),
        "",
        *_champion_lines(tables["selection"]),
        "",
        "日频列名 `minutes` 与 portal_out_7d 一致，这里数的是有结算的 bar，不是时钟分钟。",
        "汇总表在 `minute_alpha/portal_out_7d_xs_summary/REPORT.md`。",
        "",
    ]
    return "\n".join(lines)


def _plot_variant(tables: dict, folder: Path, title_prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from minute_alpha.diagnostics import plot_factor_cumsum, plot_position_cumsum

    decisions = tables["decisions"]
    valid = np.isfinite(tables["port_1"])
    if valid.any():
        eq1 = np.cumprod(1.0 + tables["port_1"][valid])
        eq0 = np.cumprod(1.0 + tables["port_0"][valid])
        ebh = np.cumprod(1.0 + tables["port_bh0"][valid])
        times = decisions[valid]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(times, eq1, label="strategy 0 fee + 1bp", lw=1.3)
        ax.plot(times, eq0, label="strategy 0 cost", lw=1.2)
        ax.plot(times, ebh, label="buy & hold", lw=1.1)
        ax.plot(times, np.ones_like(eq1), label="flat", lw=1.0)
        ax.set_title(f"{title_prefix} equal-weight equity (blank bars omitted)")
        ax.set_xlabel("decision unix seconds")
        ax.set_ylabel("equity (start = 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / "equity.png", dpi=120)
        plt.close(fig)
    boundary = int(decisions[0])
    plot_factor_cumsum(
        tables["factor_target_cumsum"], np.array([boundary]), 0,
        folder / "factor_target_cumsum.png",
        title=f"{title_prefix} factor × next-bar return (blank bars hold the cumsum)",
    )
    plot_position_cumsum(
        tables["position_target_cumsum"], np.array([boundary]), 0,
        folder / "position_target_cumsum.png",
        title=f"{title_prefix} position × next-bar return (no costs; blank bars hold)",
    )


def _write_variant(result: dict, tables: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tables["factor_target_cumsum"].to_csv(folder / "factor_target_cumsum.csv", index=False)
    tables["position_target_cumsum"].to_csv(folder / "position_target_cumsum.csv", index=False)
    tables["metrics"].to_csv(folder / "week_metrics.csv", index=False)
    tables["daily"].to_csv(folder / "daily_portfolio_returns.csv", index=False)
    tables["gaps"].to_csv(folder / "gaps.csv", index=False)
    tables["timing_rows"].to_csv(folder / "retrain_timing_history.csv", index=False)
    tables["selection"].to_csv(folder / "selection.csv", index=False)
    tables["symbol_ic"].to_csv(folder / "symbol_ic.csv", index=False)
    (folder / "week_timing.json").write_text(
        json.dumps(_json_safe(tables["timing"]), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (folder / "REPORT.md").write_text(_report_variant(tables), encoding="utf-8")
    _plot_variant(tables, folder, f"{result['timeframe']} {result['variant']}")


def _commit_dir(tmp: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refuse to overwrite {dest}")
    if dest.exists():
        dest.rmdir()
    tmp.rename(dest)


def _run_spec(payload: dict) -> dict:
    """One variant. Spawned so a multi-threaded parent cannot deadlock on fork."""
    spec = payload["spec"]
    key = spec["key"]
    packs = _load_packs(spec["timeframe"], refresh=False, quiet=True)
    bar = spec["bar_seconds"]
    decisions = decision_times(latest_closed_open(bar), bar, week_bars(bar))
    if payload.get("limit"):
        decisions = decisions[: int(payload["limit"])]
    print(f"start {key} window {spec['window']} decisions {len(decisions)}", flush=True)
    result = run_variant_from_packs(
        packs,
        decisions=decisions,
        timeframe=spec["timeframe"],
        variant=spec["variant"],
        objective=spec["objective"],
        position_mode=spec["position_mode"],
        window=spec["window"],
        bar_sec=spec["bar_seconds"],
        label=spec["label"],
        n_formulas=int(payload.get("n_formulas", N_FORMULAS)),
        progress_every=100 if not payload.get("limit") else 10,
    )
    result["coverage"] = _coverage(packs, decisions, spec["bar_seconds"])
    tables = build_tables(result)
    tables["timing"]["coverage"] = result["coverage"]
    out_root = Path(payload["out_root"])
    dest = out_root / spec["dirname"]
    tmp = out_root / f".{spec['dirname']}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    _write_variant(result, tables, tmp)
    if not payload.get("limit"):
        _commit_dir(tmp, dest)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    print(f"done {key} wall {result['wall_seconds']:.1f}s", flush=True)
    slim = {
        "key": key,
        "dirname": spec["dirname"],
        "timeframe": spec["timeframe"],
        "variant": spec["variant"],
        "label": spec["label"],
        "timing": tables["timing"],
        "metrics": tables["metrics"].to_dict(orient="records"),
        "daily": tables["daily"].to_dict(orient="records"),
        "symbol_ic": tables["symbol_ic"].to_dict(orient="records"),
        "ic": {
            "headline_ic": tables["ic"]["headline_ic"],
            "headline_icir": tables["ic"]["headline_icir"],
            "headline": tables["ic"]["headline"],
            "ts_ic_mean": tables["ic"]["ts_ic_mean"],
            "ts_icir_daily": tables["ic"]["ts_icir_daily"],
            "rank_ic_mean": tables["ic"]["rank_ic_mean"],
            "rank_icir_daily": tables["ic"]["rank_icir_daily"],
            "rank_icir_bar": tables["ic"]["rank_icir_bar"],
            "rank_ic_bars": tables["ic"]["rank_ic_bars"],
        },
        "decisions": [int(t) for t in tables["decisions"]],
        "all_pos_cumsum": [None if not np.isfinite(v) else float(v) for v in tables["all_pos_cumsum"]],
        "port_1": [None if not np.isfinite(v) else float(v) for v in tables["port_1"]],
        "champions": [],
    }
    if not tables["selection"].empty:
        counts = tables["selection"]["champion"].value_counts()
        slim["champions"] = [{"name": str(n), "count": int(c)} for n, c in counts.head(5).items()]
    return slim


def _plot_grid(slims: list[dict], folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["ts", "ts_min", "rank", "rank_long"]
    frames = ["5m", "15m"]
    titles = {
        "ts": "A mean time-series IC",
        "ts_min": "A_min worst-coin IC",
        "rank": "B rank IC",
        "rank_long": "B_long rank IC, window x3",
    }
    by_key = {item["key"]: item for item in slims}
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=False)
    for r, variant in enumerate(order):
        for c, timeframe in enumerate(frames):
            ax = axes[r][c]
            item = by_key.get(f"{timeframe}_{variant}")
            if item is None:
                ax.set_visible(False)
                continue
            y = np.asarray([np.nan if v is None else v for v in item["all_pos_cumsum"]], dtype=np.float64)
            x = np.asarray(item["decisions"], dtype=np.int64)
            ax.plot(x, y, lw=1.2, color="C0")
            ax.axhline(0.0, color="black", lw=0.6)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{timeframe} {titles[variant]}", fontsize=10)
            if r == 3:
                ax.set_xlabel("decision unix")
            if c == 0:
                ax.set_ylabel("cumsum position × return")
    fig.suptitle("4×2 position × next-bar return (no costs; blank bars hold the line)", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "position_cumsum_4x2.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    for r, variant in enumerate(order):
        for c, timeframe in enumerate(frames):
            ax = axes[r][c]
            item = by_key.get(f"{timeframe}_{variant}")
            if item is None:
                ax.set_visible(False)
                continue
            pnl = np.asarray([np.nan if v is None else v for v in item["port_1"]], dtype=np.float64)
            ok = np.isfinite(pnl)
            if not ok.any():
                continue
            equity = np.cumprod(1.0 + pnl[ok])
            ax.plot(np.asarray(item["decisions"])[ok], equity, lw=1.2, color="C1")
            ax.axhline(1.0, color="black", lw=0.6)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{timeframe} {titles[variant]}", fontsize=10)
            if r == 3:
                ax.set_xlabel("decision unix")
            if c == 0:
                ax.set_ylabel("equity, 0 fee + 1bp")
    fig.suptitle("4×2 equal-weight equity after 0 fee + 1bp slippage", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "equity_4x2.png", dpi=120)
    plt.close(fig)


def _summary_report(slims: list[dict], pool_wall: float, coverage: dict) -> str:
    lines = [
        "# 5m / 15m 截面训练回测",
        "",
        *_method_lines(),
        "",
        f"8 个变体同时跑，进程池墙钟 {_secs(pool_wall)}。",
        "",
        "## 数据覆盖",
        "",
        "周内槽位含决策 bar 和最后一根结算 bar。缺的槽位不会被填 0。",
        "",
    ]
    for timeframe, rows in coverage.items():
        lines.append(f"### {timeframe}")
        lines.append("")
        lines.append("| 币 | 第一根 | 最后一根 | 根数 | 周内缺失 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['symbol']} | {_utc(row['first']) if row['first'] else '—'} | "
                f"{_utc(row['last']) if row['last'] else '—'} | {row['prints']} | "
                f"{row['missing_week_slots']} / {row['week_slots']} |"
            )
        lines.append("")
    lines += [
        "## 汇总",
        "",
        "| 周期 | 变体 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有 | OOS IC | 日频 ICIR | CPU 中位 | CPU P95 | 进程墙钟 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = [spec["key"] for spec in variant_specs()]
    by_key = {item["key"]: item for item in slims}
    for key in order:
        item = by_key.get(key)
        if item is None:
            continue
        all_row = next(row for row in item["metrics"] if row["symbol"] == "ALL")
        timing = item["timing"]
        lines.append(
            f"| {item['timeframe']} | {item['label']} | {_pct(all_row['return_1bp'])} | "
            f"{_num(all_row['sharpe_1bp'])} | {_pct(all_row['mdd_1bp'])} | {_num(all_row['turnover_1bp'], 4)} | "
            f"{_pct(all_row['return_0cost'])} | {_num(all_row['sharpe_0cost'])} | {_pct(all_row['mdd_0cost'])} | "
            f"{_pct(all_row['buyhold_return_0cost'])} | {_num(item['ic']['headline_ic'], 4)} | "
            f"{_num(item['ic']['headline_icir'], 3)} | {_secs(timing['retrain_cpu_seconds_median'])} | "
            f"{_secs(timing['retrain_cpu_seconds_p95'])} | {_secs(timing['wall_seconds'])} |"
        )
    lines += [
        "",
        "换手是有结算的 bar 上的平均实际换手（空白 bar 的换手滚到下一根结算 bar）。",
        "OOS IC / 日频 ICIR 与该变体的选式目标一致：A 与 A_min 用各币 Pearson，B 与 B_long 用截面 Rank IC。",
        "",
        "图：`position_cumsum_4x2.png` 是无成本的仓位×下一根收益累加，`equity_4x2.png` 是扣 1bp 滑点后的等权净值。",
        "",
    ]
    for key in order:
        item = by_key.get(key)
        if item is None:
            continue
        timing = item["timing"]
        lines += [
            f"## {item['timeframe']} {item['label']}",
            "",
            f"目录 `minute_alpha/{item['dirname']}/`。窗口 {timing['window']} 根。",
            f"样本外 IC {_num(item['ic']['headline_ic'], 4)}，日频 ICIR {_num(item['ic']['headline_icir'], 3)}。"
            f"逐 bar Rank IC {_num(item['ic']['rank_ic_mean'], 4)}，逐 bar ICIR {_num(item['ic']['rank_icir_bar'], 3)}。",
            f"重训 {timing['retrain_count']} 次，CPU 中位数 {_secs(timing['retrain_cpu_seconds_median'])}，"
            f"P95 {_secs(timing['retrain_cpu_seconds_p95'])}，单次墙钟中位数 {_secs(timing['retrain_wall_seconds_median'])}，"
            f"进程墙钟 {_secs(timing['wall_seconds'])}。",
            f"训练 IC 中位数 {_num(timing.get('train_ic_median'), 4)}。"
            f"缺口行 {timing['gap_rows']}，历史不足 {timing['insufficient_history_rows']}，"
            f"截面不足两个币 {timing['cross_section_undefined_rows']}。",
            "",
            "### 各币",
            "",
            *_metrics_table(pd.DataFrame(item["metrics"])),
            "",
            "### 分天",
            "",
            *_daily_table(pd.DataFrame(item["daily"])),
            "",
            "### 样本外 IC",
            "",
            *_ic_table(pd.DataFrame(item["symbol_ic"])),
            "",
        ]
        if item["champions"]:
            lines.append("选中次数： " + "，".join(
                f"{row['name']} {row['count']}" for row in item["champions"]
            ) + "。")
            lines.append("")
    return "\n".join(lines)


def _load_packs(timeframe: str, refresh: bool = False, quiet: bool = False) -> dict[str, dict]:
    from minute_alpha.data import load_timeframe_frames

    bar = bar_seconds(timeframe)
    latest = latest_closed_open(bar)
    frames = load_timeframe_frames(timeframe, bars=10000, cache=True, refresh=refresh)
    packs = packs_from_frames(frames, latest, bar)
    need_from = int(decision_times(latest, bar, week_bars(bar))[0]) - (base_window(timeframe) * 3 - 1) * bar
    for symbol, pack in packs.items():
        first = int(pack["time"][0]) if pack["time"].size else None
        last = int(pack["time"][-1]) if pack["time"].size else None
        if not quiet:
            print(
                f"{timeframe} {symbol} prints {pack['time'].size} "
                f"{_utc(first) if first else None} → {_utc(last) if last else None} "
                f"need_from {_utc(need_from)}",
                flush=True,
            )
        if first is None or first > need_from:
            raise RuntimeError(f"{timeframe} {symbol} does not cover the long window start {need_from}")
        if last is None or last < latest and not quiet:
            print(f"{timeframe} {symbol} last print is before {_utc(latest)}", flush=True)
    return packs


def main(limit_decisions: int | None = None, only: list[str] | None = None, refresh: bool = False) -> dict:
    specs = variant_specs()
    if only:
        wanted = set(only)
        specs = [spec for spec in specs if spec["key"] in wanted or spec["variant"] in wanted or spec["timeframe"] in wanted]
        if not specs:
            raise SystemExit(f"no variants matched {only}")
    out_root = ROOT if not limit_decisions else Path("/tmp/xs_week_limit")
    if limit_decisions:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        for spec in specs:
            dest = out_root / spec["dirname"]
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(f"refuse to overwrite {dest}")
    timeframes = sorted({spec["timeframe"] for spec in specs})
    packs = {tf: _load_packs(tf, refresh=refresh) for tf in timeframes}
    decisions = {}
    coverage = {}
    for tf in timeframes:
        bar = bar_seconds(tf)
        latest = latest_closed_open(bar)
        clock = decision_times(latest, bar, week_bars(bar))
        decisions[tf] = clock
        coverage[tf] = _coverage(packs[tf], clock if not limit_decisions else clock[:limit_decisions], bar)
        print(
            f"{tf} latest {_utc(latest)} decisions {len(clock)} "
            f"{_utc(int(clock[0]))} → {_utc(int(clock[-1]))}",
            flush=True,
        )
    payloads = [{
        "spec": spec,
        "limit": limit_decisions,
        "out_root": str(out_root),
        "n_formulas": N_FORMULAS,
    } for spec in specs]
    wall0 = time.perf_counter()
    if len(payloads) == 1:
        slims = [_run_spec(payloads[0])]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            slims = pool.map(_run_spec, payloads)
    pool_wall = time.perf_counter() - wall0
    if limit_decisions:
        print(f"limit run done in {pool_wall:.1f}s under {out_root}", flush=True)
        return {"pool_wall": pool_wall, "slims": slims}
    summary = out_root / "portal_out_7d_xs_summary"
    if summary.exists():
        shutil.rmtree(summary)
    summary.mkdir(parents=True)
    text = _summary_report(slims, pool_wall, coverage)
    (summary / "REPORT.md").write_text(text, encoding="utf-8")
    (summary / "summary.json").write_text(
        json.dumps(_json_safe({"pool_wall_seconds": pool_wall, "runs": slims}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_grid(slims, summary)
    artifact = Path("/opt/cursor/artifacts/minute_alpha_xs")
    artifact.mkdir(parents=True, exist_ok=True)
    shutil.copy(summary / "REPORT.md", artifact / "REPORT.md")
    for name in ("position_cumsum_4x2.png", "equity_4x2.png"):
        src = summary / name
        if src.exists():
            shutil.copy(src, artifact / name)
    print(f"pool wall {pool_wall:.1f}s summary {summary}", flush=True)
    return {"pool_wall": pool_wall, "summary": str(summary)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5m/15m cross-sectional week retrain")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N decisions, write under /tmp")
    parser.add_argument("--only", default="", help="comma-separated variant keys, e.g. 15m_ts,5m_rank")
    parser.add_argument("--refresh", action="store_true", help="redownload Bitfinex candles")
    args = parser.parse_args()
    only = [part for part in args.only.split(",") if part]
    main(limit_decisions=args.limit or None, only=only or None, refresh=args.refresh)
