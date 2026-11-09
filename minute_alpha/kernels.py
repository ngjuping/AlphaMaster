"""Causal rolling kernels.

Windows are left-padded with zeros, matching ``model_core`` unfold. Median uses
the lower of the two middle values, matching ``torch.median`` on an even window.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from minute_alpha.config import SECONDS_PER_YEAR

EPS = 1e-9
CLIP = 5.0


def clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    """Causal window view, shape [N, T, w]. Index -1 is the current bar."""
    if w <= 1:
        return x[..., None]
    pad = np.zeros((x.shape[0], w - 1), dtype=x.dtype)
    xp = np.concatenate([pad, x], axis=1)
    return sliding_window_view(xp, w, axis=1)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return rolling(x, w).mean(axis=-1)


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    return rolling(x, w).sum(axis=-1)


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Population std inside the window, plus 1e-9 (features._rolling_std)."""
    wdw = rolling(x, w)
    m = wdw.mean(axis=-1, keepdims=True)
    return np.sqrt(((wdw - m) ** 2).mean(axis=-1)) + 1e-9


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    return rolling(x, w).max(axis=-1)


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    return rolling(x, w).min(axis=-1)


def delay(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return x
    out = np.zeros_like(x)
    out[:, d:] = x[:, :-d]
    return out


def _lower_median(windows: np.ndarray) -> np.ndarray:
    k = (windows.shape[-1] - 1) // 2
    part = np.partition(windows, k, axis=-1)
    return part[..., k]


def robust_norm(x: np.ndarray, w: int = 200) -> np.ndarray:
    """Causal rolling median/MAD norm. Warm-up (t < w-1) is 0.

    Computed in float32 so the lower-median index matches the production
    torch.float32 feature path. A float64 median can pick a different order
    statistic and move the whole robust z-score.
    """
    x32 = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n, t = x32.shape
    if t < w:
        return np.zeros((n, t), dtype=np.float32)
    wdw = rolling(x32, w)
    med = _lower_median(wdw)
    mad = _lower_median(np.abs(wdw - med[..., None])) + np.float32(1e-6)
    out = np.clip((x32 - med) / mad, -CLIP, CLIP).astype(np.float32)
    out = clean(out)
    out[:, : w - 1] = 0.0
    return out


def norm(x: np.ndarray) -> np.ndarray:
    return clean(robust_norm(clean(x), 200))


def ema_exact(x: np.ndarray, span: int) -> np.ndarray:
    """Operator EMA: exact recursion, out[0] = x[0], in the input dtype."""
    src = np.asarray(x)
    if src.shape[1] == 0:
        return src.copy()
    alpha = src.dtype.type(2.0 / (span + 1.0))
    if float(alpha) >= 1.0:
        return src.copy()
    decay = src.dtype.type(1.0) - alpha
    out = np.empty_like(src)
    out[:, 0] = src[:, 0]
    for i in range(1, src.shape[1]):
        out[:, i] = alpha * src[:, i] + decay * out[:, i - 1]
    return out


def ema_feature(x: np.ndarray, span: int) -> np.ndarray:
    """Feature EMA. Exact while T < 2*w_full, else causal convolution.

    Matches ``MT5FeatureEngineer._ema_simple`` (not the operator EMA, which
    was later forced onto the exact path only). Arithmetic stays float32 when
    the caller passes float32, matching the production feature tensor.
    """
    src = np.asarray(x, dtype=np.float32)
    n, t = src.shape
    alpha = np.float32(2.0 / (span + 1.0))
    if t == 0 or float(alpha) >= 1.0:
        return src.copy()
    w_full = max(1, math.ceil(-math.log(1e-6) / (-math.log(1.0 - float(alpha)))))
    if t < 2 * w_full:
        return ema_exact(src, span)
    decay = np.float32(1.0) - alpha
    powers = np.arange(w_full - 1, -1, -1, dtype=np.float32)
    weights = alpha * (decay ** powers)
    first = np.repeat(src[:, :1], w_full - 1, axis=1)
    xp = np.concatenate([first, src], axis=1)
    windows = sliding_window_view(xp, w_full, axis=1)
    return clean((windows * weights).sum(axis=-1))


def linear_slope(x: np.ndarray, w: int) -> np.ndarray:
    src = np.asarray(x, dtype=np.float32)
    wdw = rolling(src, w)
    tidx = np.arange(w, dtype=np.float32)
    tc = tidx - tidx.mean()
    tvar = np.float32((tc ** 2).sum())
    xm = wdw.mean(axis=-1, keepdims=True)
    slope = ((wdw - xm) * tc).sum(axis=-1) / (tvar + np.float32(EPS))
    slope = slope / (xm[..., 0] + np.float32(EPS))
    return clean(slope)


def trend_strength(x: np.ndarray, w: int) -> np.ndarray:
    src = np.asarray(x, dtype=np.float32)
    wdw = rolling(src, w)
    tidx = np.arange(w, dtype=np.float32)
    tc = tidx - tidx.mean()
    tvar = np.float32((tc ** 2).sum())
    xm = wdw.mean(axis=-1, keepdims=True)
    slope = ((wdw - xm) * tc).sum(axis=-1) / (tvar + np.float32(EPS))
    pred = xm + slope[..., None] * tc
    ss_res = ((wdw - pred) ** 2).sum(axis=-1)
    ss_tot = ((wdw - xm) ** 2).sum(axis=-1)
    r2 = np.clip(1.0 - ss_res / (ss_tot + np.float32(EPS)), 0.0, 1.0)
    slope_norm = slope / (xm[..., 0] + np.float32(EPS))
    return clean(slope_norm * r2)


def ts_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    wx = rolling(x, w)
    wy = rolling(y, w)
    mx = wx.mean(axis=-1, keepdims=True)
    my = wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    sx = np.sqrt(((wx - mx) ** 2).mean(axis=-1))
    sy = np.sqrt(((wy - my) ** 2).mean(axis=-1))
    corr = cov / (sx * sy + 1e-8)
    corr[(sx < 1e-6) | (sy < 1e-6)] = 0.0
    return clean(corr)


def log_ret(close: np.ndarray, k: int) -> np.ndarray:
    src = np.asarray(close, dtype=np.float32)
    out = np.zeros(src.shape, dtype=np.float32)
    if src.shape[1] > k:
        out[:, k:] = np.log(src[:, k:] / (src[:, :-k] + np.float32(EPS)))
    return out


def atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 14) -> np.ndarray:
    prev = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return rolling_mean(tr, w)


def periods_per_year(times: np.ndarray) -> int:
    """Same estimator as ``model_core.backtest.estimate_periods_per_year``."""
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    estimates = []
    for row in arr:
        if row.size < 2:
            continue
        span = float(row[-1] - row[0])
        if span <= 0:
            continue
        years = span / SECONDS_PER_YEAR
        if years <= 0:
            continue
        estimates.append(int(max(10, min(600_000, round(row.size / years)))))
    if not estimates:
        return 6240
    estimates.sort()
    return estimates[len(estimates) // 2]
