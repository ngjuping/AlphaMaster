"""Operator library in production registration order.

Names, arities and numerics follow ``model_core/ops.py``. Binary and ternary
operators require identical shapes. The VM, not each operator, maps residual
non-finite values to 0 / +1 / -1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from minute_alpha.kernels import clean, delay, ema_exact, rolling

_F64 = np.float64


class ShapeError(ValueError):
    pass


def _check(name: str, args: tuple[np.ndarray, ...]) -> None:
    base = args[0].shape
    for other in args[1:]:
        if other.shape != base:
            raise ShapeError(f"{name}: {base} vs {other.shape}")


def _delay(x: np.ndarray, d: int) -> np.ndarray:
    return delay(x, d)


def _gate(c: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _check("GATE", (c, x, y))
    mask = (c > 0).astype(_F64)
    return mask * x + (1.0 - mask) * y


def _jump(x: np.ndarray) -> np.ndarray:
    n, t = x.shape
    cnt = np.arange(1, t + 1, dtype=_F64).reshape(1, t)
    csum = np.cumsum(x, axis=1)
    mean = csum / cnt
    var = np.cumsum(x * x, axis=1) / cnt - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12)) + 1e-6
    out = np.tanh((x - mean) / std - 1.5)
    if t > 5:
        out[:, :5] = 0.0
    return out


def _decay(x: np.ndarray) -> np.ndarray:
    return (x + 0.8 * _delay(x, 1) + 0.6 * _delay(x, 2)) / 2.4


def _wma(x: np.ndarray) -> np.ndarray:
    return (3.0 * x + 2.0 * _delay(x, 1) + 1.0 * _delay(x, 2)) / 6.0


def _mean(x: np.ndarray, d: int) -> np.ndarray:
    return rolling(x, d).mean(axis=-1)


def _std(x: np.ndarray, d: int) -> np.ndarray:
    w = rolling(x, d)
    m = w.mean(axis=-1, keepdims=True)
    std = np.sqrt(((w - m) ** 2).mean(axis=-1)) + 1e-6
    return clean(std)


def _rank(x: np.ndarray, d: int) -> np.ndarray:
    w = rolling(x, d)
    cur = w[:, :, -1:]
    return clean((w < cur).mean(axis=-1))


def _corr10(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _check("TS_CORR_10", (x, y))
    wx = rolling(x, 10)
    wy = rolling(y, 10)
    mx = wx.mean(axis=-1, keepdims=True)
    my = wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    sx = np.sqrt(((wx - mx) ** 2).mean(axis=-1))
    sy = np.sqrt(((wy - my) ** 2).mean(axis=-1))
    corr = cov / (sx * sy + 1e-8)
    corr[(sx < 1e-6) | (sy < 1e-6)] = 0.0
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _quantile(x: np.ndarray, d: int) -> np.ndarray:
    w = rolling(x, d)
    cur = w[:, :, -1:]
    return clean((w < cur).mean(axis=-1))


def _skew(x: np.ndarray, d: int) -> np.ndarray:
    w = rolling(x, d)
    m = w.mean(axis=-1, keepdims=True)
    s = np.sqrt(((w - m) ** 2).mean(axis=-1)) + 1e-6
    skew = ((w - m) ** 3).mean(axis=-1) / (s ** 3)
    return np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)


def _delta(x: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(x)
    if d > 0:
        out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _arg_max(x: np.ndarray, d: int) -> np.ndarray:
    idx = rolling(x, d).argmax(axis=-1).astype(_F64)
    return idx / max(d - 1, 1)


def _arg_min(x: np.ndarray, d: int) -> np.ndarray:
    idx = rolling(x, d).argmin(axis=-1).astype(_F64)
    return idx / max(d - 1, 1)


def _decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    w = rolling(x, d)
    weights = np.arange(1, d + 1, dtype=_F64)
    weights = weights / weights.sum()
    return (w * weights).sum(axis=-1)


def _decay_exp(x: np.ndarray, d: int, alpha: float = 0.5) -> np.ndarray:
    # Weight i applies to the i-th oldest bar in the window (production code).
    w = rolling(x, d)
    weights = np.array([alpha * (1 - alpha) ** i for i in range(d)], dtype=_F64)
    weights = weights / weights.sum()
    return (w * weights).sum(axis=-1)


def _scale(x: np.ndarray) -> np.ndarray:
    return x / (np.cumsum(np.abs(x), axis=1) + 1e-6)


def _cov(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    _check("COVARIANCE_10", (x, y))
    wx = rolling(x, d)
    wy = rolling(y, d)
    mx = wx.mean(axis=-1, keepdims=True)
    my = wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    return clean(cov)


def _product(x: np.ndarray, d: int) -> np.ndarray:
    safe = np.maximum(x, -0.999)
    log_x = np.log1p(safe)
    log_sum = np.clip(rolling(log_x, d).sum(axis=-1), -10.0, 10.0)
    return np.nan_to_num(np.expm1(log_sum), nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    out = np.sign(x) * np.abs(x) ** a
    return np.nan_to_num(np.clip(out, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


def _cs_rank(x: np.ndarray) -> np.ndarray:
    n, t = x.shape
    if n == 1:
        return np.nan_to_num(x, nan=0.5, posinf=0.5, neginf=0.5)
    order = np.argsort(x, axis=0)
    ranks = np.empty_like(x)
    vals = np.broadcast_to(np.arange(n, dtype=_F64)[:, None], x.shape).copy()
    np.put_along_axis(ranks, order, vals, axis=0)
    return np.nan_to_num(ranks / (n - 1), nan=0.5, posinf=0.5, neginf=0.5)


def _cs_scale(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.nan_to_num(x, nan=0.5, posinf=0.5, neginf=0.5)
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    span = mx - mn
    zero = np.abs(span) < 1e-9
    span_safe = np.where(zero, 1.0, span)
    out = (x - mn) / span_safe
    out = np.where(zero, 0.5, out)
    return np.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_neutralize(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(x - x.mean(axis=0, keepdims=True), nan=0.0, posinf=0.0, neginf=0.0)


def _ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    return rolling(x, d).sum(axis=-1)


def _zscore(x: np.ndarray, w: int) -> np.ndarray:
    windows = rolling(x, w)
    m = windows.mean(axis=-1)
    s = np.sqrt(((windows - m[..., None]) ** 2).mean(axis=-1)) + 1e-9
    z = (x - m) / s
    z = np.where(s < 2e-9, 0.0, z)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(x: np.ndarray) -> np.ndarray:
    windows = rolling(np.asarray(x, dtype=_F64), 20)
    lower = np.quantile(windows, 0.05, axis=-1)
    upper = np.quantile(windows, 0.95, axis=-1)
    span = upper - lower
    safe_lower = np.where(span < 1e-9, x, lower)
    safe_upper = np.where(span < 1e-9, x, upper)
    out = np.minimum(np.maximum(x, safe_lower), safe_upper)
    return clean(out)


def _add(x, y):
    _check("ADD", (x, y))
    return x + y


def _sub(x, y):
    _check("SUB", (x, y))
    return x - y


def _mul(x, y):
    _check("MUL", (x, y))
    return x * y


def _div(x, y):
    _check("DIV", (x, y))
    return x / (y + 1e-6)


def _min(x, y):
    _check("MIN", (x, y))
    return clean(np.minimum(x, y))


def _max(x, y):
    _check("MAX", (x, y))
    return clean(np.maximum(x, y))


def _if_gt(x, y, z):
    _check("IF_GT", (x, y, z))
    return np.nan_to_num(np.where(x > 0, y, z), nan=0.0, posinf=0.0, neginf=0.0)


# (name, arity, function) in OPERATOR_REGISTRY order.
OPERATORS: list[tuple[str, int, object]] = [
    ("ADD", 2, _add),
    ("SUB", 2, _sub),
    ("MUL", 2, _mul),
    ("DIV", 2, _div),
    ("NEG", 1, lambda x: -x),
    ("ABS", 1, np.abs),
    ("SIGN", 1, np.sign),
    ("GATE", 3, _gate),
    ("JUMP", 1, _jump),
    ("DECAY", 1, _decay),
    ("DELAY1", 1, lambda x: _delay(x, 1)),
    ("MAX3", 1, lambda x: np.maximum(x, np.maximum(_delay(x, 1), _delay(x, 2)))),
    ("TS_MEAN_5", 1, lambda x: _mean(x, 5)),
    ("TS_MEAN_10", 1, lambda x: _mean(x, 10)),
    ("TS_MEAN_20", 1, lambda x: _mean(x, 20)),
    ("TS_STD_5", 1, lambda x: _std(x, 5)),
    ("TS_STD_10", 1, lambda x: _std(x, 10)),
    ("TS_STD_20", 1, lambda x: _std(x, 20)),
    ("TS_RANK_5", 1, lambda x: _rank(x, 5)),
    ("TS_RANK_10", 1, lambda x: _rank(x, 10)),
    ("TS_RANK_20", 1, lambda x: _rank(x, 20)),
    ("TS_CORR_10", 2, _corr10),
    ("MOMENTUM_5", 1, lambda x: _mean(x, 5) - _mean(x, 20)),
    ("MOMENTUM_10", 1, lambda x: _mean(x, 10) - _mean(x, 20)),
    ("TS_MAX_10", 1, lambda x: rolling(x, 10).max(axis=-1)),
    ("TS_MIN_10", 1, lambda x: rolling(x, 10).min(axis=-1)),
    ("WMA", 1, _wma),
    ("DELAY4", 1, lambda x: _delay(x, 4)),
    ("EMA_5", 1, lambda x: ema_exact(x, 5)),
    ("EMA_20", 1, lambda x: ema_exact(x, 20)),
    ("TS_QUANTILE_10", 1, lambda x: _quantile(x, 10)),
    ("TS_SKEW_10", 1, lambda x: _skew(x, 10)),
    ("TS_MIN_20", 1, lambda x: rolling(x, 20).min(axis=-1)),
    ("TS_MAX_20", 1, lambda x: rolling(x, 20).max(axis=-1)),
    ("DELTA", 1, lambda x: _delta(x, 1)),
    ("TS_ARG_MAX_5", 1, lambda x: _arg_max(x, 5)),
    ("TS_ARG_MIN_5", 1, lambda x: _arg_min(x, 5)),
    ("DECAY_LINEAR_5", 1, lambda x: _decay_linear(x, 5)),
    ("SCALE", 1, _scale),
    ("COVARIANCE_10", 2, lambda x, y: _cov(x, y, 10)),
    ("PRODUCT_5", 1, lambda x: _product(x, 5)),
    ("SIGNED_POWER_2", 1, lambda x: _signed_power(x, 2.0)),
    ("TS_DECAY_EXP_5", 1, lambda x: _decay_exp(x, 5, 0.5)),
    ("DELTA_5", 1, lambda x: _delta(x, 5)),
    ("CS_RANK", 1, _cs_rank),
    ("CS_SCALE", 1, _cs_scale),
    ("CS_NEUTRALIZE", 1, _cs_neutralize),
    ("TS_SUM_5", 1, lambda x: clean(_ts_sum(x, 5))),
    ("TS_SUM_10", 1, lambda x: clean(_ts_sum(x, 10))),
    ("TS_SUM_20", 1, lambda x: clean(_ts_sum(x, 20))),
    ("MIN", 2, _min),
    ("MAX", 2, _max),
    ("POWER", 1, lambda x: _signed_power(x, 2.0)),
    ("SIGNED_LOG", 1, lambda x: clean(np.sign(x) * np.log1p(np.abs(x)))),
    ("SQRT", 1, lambda x: clean(np.sign(x) * np.sqrt(np.abs(x)))),
    ("TS_ZSCORE_10", 1, lambda x: _zscore(x, 10)),
    ("TS_ZSCORE_20", 1, lambda x: _zscore(x, 20)),
    ("WINSORIZE", 1, _winsorize),
    ("CLIP", 1, lambda x: np.clip(x, -3.0, 3.0)),
    ("SIGMOID", 1, lambda x: np.nan_to_num(2.0 * expit(x) - 1.0, nan=0.0, posinf=1.0, neginf=-1.0)),
    ("TANH_SQUASH", 1, lambda x: np.nan_to_num(np.tanh(x), nan=0.0, posinf=1.0, neginf=-1.0)),
    ("IF_GT", 3, _if_gt),
]

OPERATOR_NAMES = tuple(name for name, _, _ in OPERATORS)
ARITY = {name: arity for name, arity, _ in OPERATORS}
POSITIVE_ONLY = {"TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS"}
INFECTED = {
    "TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS",
    "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
    "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
    "TS_MAX_10", "TS_MAX_20",
    "CLIP", "SQRT", "POWER", "SIGNED_LOG",
    "SIGMOID", "TANH_SQUASH", "WINSORIZE",
    "WMA", "EMA_5", "EMA_20", "DECAY", "DECAY_LINEAR_5",
    "TS_DECAY_EXP_5",
}
RESTORE = {
    "SUB", "DIV", "NEG", "GATE", "IF_GT",
    "TS_ZSCORE_10", "TS_ZSCORE_20",
    "CS_NEUTRALIZE", "CS_RANK", "CS_SCALE",
    "TS_STD_5", "TS_STD_10", "TS_STD_20",
    "TS_CORR_10", "TS_SKEW_10", "TS_QUANTILE_10",
    "DELTA", "DELTA_5", "MOMENTUM_5", "MOMENTUM_10",
}
