"""FTMO single-symbol score used to rank formulas inside a training window.

The arithmetic follows ``MT5Backtest._multi_objective`` with
``REWARD_MODE='ftmo'`` and ``evaluate_fold``. Selection never sees labels past
the array it is given: callers pass a prefix whose last two targets are the
boundary zeros from ``target_log_open``.
"""

from __future__ import annotations

import math

import numpy as np

from minute_alpha.config import (
    IC_GATE_MULT,
    IC_GATE_THRESH,
    IC_NEG_MULT,
    N_FOLDS,
    REWARD_ALPHA,
    TRAIN_COST_RATE,
    WF_GAP,
)
from minute_alpha.position import target_positions


def repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


def walk_forward_folds(t: int, n_folds: int = N_FOLDS, gap: int = WF_GAP) -> list[dict]:
    fold_size = t // n_folds
    if fold_size < 2:
        return [{"train_start": 0, "train_end": t, "val_start": 0, "val_end": t, "gap": 0}]
    total_required = fold_size * n_folds + gap * (n_folds - 1)
    if total_required > t:
        gap = max(0, (t - fold_size * n_folds) // n_folds)
    folds = []
    for k in range(1, n_folds):
        train_start = (k - 1) * fold_size
        train_end = k * fold_size
        val_start = train_end + gap
        val_end = min(val_start + fold_size, t)
        if val_start >= t or val_end <= val_start:
            break
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "gap": gap,
            }
        )
    if not folds:
        return [{"train_start": 0, "train_end": t, "val_start": 0, "val_end": t, "gap": 0}]
    return folds


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _sortino(pnl: np.ndarray, periods: int) -> float:
    flat = np.asarray(pnl, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    mean = float(flat.mean())
    down = flat[flat < 0]
    raw = float(down.std(ddof=0)) if down.size else 0.0
    full = float(flat.std(ddof=0)) if flat.size else 0.0
    floor = max(full * 0.2, 1e-8)
    downside = max(raw, floor, 1e-8)
    return _clip(mean / downside * math.sqrt(periods), -20.0, 20.0)


def _calmar(pnl: np.ndarray, periods: int) -> float:
    flat = np.asarray(pnl, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    ann = float(flat.mean()) * periods
    cum = np.cumsum(flat)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if cum.size else 0.0
    dd = max(dd, 1e-8)
    return _clip(ann / dd, -10.0, 10.0)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return None
    xm = x - x.mean()
    ym = y - y.mean()
    sx = math.sqrt(float((xm ** 2).mean()))
    sy = math.sqrt(float((ym ** 2).mean()))
    if sx < 1e-6 or sy < 1e-6:
        return None
    return float((xm * ym).mean() / (sx * sy + 1e-8))


def _ic_stability(factor: np.ndarray, target: np.ndarray) -> float:
    """Time-series IC stability. factor[:-1] vs target[1:], as in production."""
    n, t = factor.shape
    if t < 10:
        return 0.0
    ics = []
    for i in range(n):
        ic = _pearson(factor[i, :-1], target[i, 1:])
        if ic is not None:
            ics.append(ic)
    if not ics:
        return 0.0
    mean = sum(ics) / len(ics)
    var = sum((v - mean) ** 2 for v in ics) / len(ics)
    stab = mean / (math.sqrt(var) + 1e-6)
    return _clip(stab, -3.0, 3.0)


def _turnover_quality(position: np.ndarray) -> float:
    # Production casts continuous tanh positions with int(), so every |p|<1
    # counts as flat and the frequency score is the empty-trade value -2.
    # That term is constant under the production position map and does not
    # change which formula wins. Keep the same constant.
    flat = np.abs(np.asarray(position))
    if flat.size == 0 or float(flat.max()) < 1.0:
        return -2.0
    return -2.0


def _beta_penalty(position: np.ndarray) -> float:
    flat = np.asarray(position, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    long_ratio = float((flat > 0.05).mean())
    short_ratio = float((flat < -0.05).mean())
    max_ratio = max(long_ratio, short_ratio)
    if max_ratio > 0.85:
        return -2.0 * (max_ratio - 0.85) / 0.15
    if max_ratio > 0.70:
        return -0.5 * (max_ratio - 0.70) / 0.15
    return 0.0


def _half_consistency(pnl: np.ndarray, periods: int) -> float:
    t = pnl.shape[1]
    if t < 20:
        return 0.0
    half = t // 2
    s1 = _sortino(pnl[:, :half], periods)
    s2 = _sortino(pnl[:, half:], periods)
    if s1 > 0 and s2 > 0:
        return 0.5
    if s1 * s2 < 0:
        return -1.0
    return 0.0


def _exposure_penalty(position: np.ndarray) -> float:
    exposure = float(np.abs(position).mean())
    if exposure < 0.10:
        return (exposure / 0.10 - 1.0) * 2.0
    return 0.0


def _turnover_penalty(turnover: np.ndarray) -> float:
    mean_to = float(np.asarray(turnover).mean())
    return -_clip((mean_to - 0.2) * 3.0, 0.0, 3.0)


def _multi_objective(factor, target, pnl, position, periods: int) -> float:
    ann = float(pnl.mean()) * periods
    sortino = _sortino(pnl, periods)
    calmar = _calmar(pnl, periods)
    ic = _ic_stability(factor, target)
    tq = _turnover_quality(position)
    exp_pen = _exposure_penalty(position)
    beta = _beta_penalty(position)
    consist = _half_consistency(pnl, periods)
    return (
        0.80 * ann
        + 0.05 * sortino
        + 0.10 * calmar
        + 0.03 * ic
        + 0.02 * tq
        + exp_pen
        + beta
        + consist
    )


def _ic_gate(score: float, ic: float | None) -> float:
    if ic is None:
        return score
    if ic > IC_GATE_THRESH:
        return score * IC_GATE_MULT
    if ic < -IC_GATE_THRESH:
        return score * IC_NEG_MULT
    return score


def _mean_ic(factor: np.ndarray, target: np.ndarray) -> float | None:
    ics = []
    for i in range(factor.shape[0]):
        ic = _pearson(factor[i, :-1], target[i, 1:])
        if ic is not None:
            ics.append(ic)
    if not ics:
        return None
    return sum(ics) / len(ics)


def pnl_and_turnover(factor: np.ndarray, target: np.ndarray, cost: float):
    position = target_positions(factor)
    prev = np.roll(position, 1, axis=1)
    prev[:, 0] = 0.0
    turnover = np.abs(position - prev)
    pnl = position * target - turnover * cost
    return position, turnover, pnl


def score_formula(
    factor: np.ndarray,
    target: np.ndarray,
    periods: int,
    cost: float = TRAIN_COST_RATE,
    n_folds: int = N_FOLDS,
    gap: int = WF_GAP,
) -> tuple[float, float]:
    """Return (train_score, val_score) averaged over inner walk-forward folds."""
    _pos, turnover, pnl = pnl_and_turnover(factor, target, cost)
    folds = walk_forward_folds(factor.shape[1], n_folds=n_folds, gap=gap)
    use_wf = not (len(folds) == 1 and folds[0]["train_end"] == factor.shape[1] and folds[0]["gap"] == 0)
    train_scores = []
    val_scores = []
    if not use_wf:
        folds = walk_forward_folds(factor.shape[1], n_folds=n_folds, gap=gap)
    for fold in folds:
        ts, te = fold["train_start"], fold["train_end"]
        vs, ve = fold["val_start"], fold["val_end"]
        train_score = _multi_objective(
            factor[:, ts:te], target[:, ts:te], pnl[:, ts:te], _pos[:, ts:te], periods
        ) + _turnover_penalty(turnover[:, ts:te])
        ic_tr = _mean_ic(factor[:, ts:te], target[:, ts:te])
        train_scores.append(REWARD_ALPHA * _ic_gate(train_score, ic_tr))

        base_val = _multi_objective(
            factor[:, vs:ve], target[:, vs:ve], pnl[:, vs:ve], _pos[:, vs:ve], periods
        )
        oos = _sortino(pnl[:, vs:ve], periods)
        mult = max(0.1, 0.5 + oos * 0.4) if oos <= 0 else min(1.2, 1.0 + oos * 0.1)
        ic_va = _mean_ic(factor[:, vs:ve], target[:, vs:ve])
        val_scores.append(_ic_gate(base_val * mult, ic_va))
    return float(np.mean(train_scores)), float(np.mean(val_scores))
