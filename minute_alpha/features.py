"""Production feature stack, 65 names, same order as ``FEATURE_REGISTRY``.

Each series is a causal function of OHLCV. With N=1, cross-sectional features
fall back to the time-series path used by single-symbol production training.
"""

from __future__ import annotations

import math

import numpy as np

from minute_alpha.kernels import (
    EPS,
    atr,
    clean,
    ema_feature,
    linear_slope,
    log_ret,
    norm,
    rolling,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
    trend_strength,
    ts_corr,
)

FEATURE_NAMES: tuple[str, ...] = (
    "RET", "RET5", "RET20", "MA_DIFF", "SLOPE20",
    "ATR", "RVOL", "HL_RANGE", "VOL_REGIME",
    "DEV", "DEV60", "RSI14", "PRESSURE", "AC1",
    "VOL_RATIO", "VOL_Z", "PV_CORR",
    "REL_RET5", "REL_RET20", "REL_VOL",
    "VWAP_DEV", "BOLL_POS", "BOLL_WIDTH", "MACD_HIST", "OBV_SLOPE", "MFI14",
    "WILLR_14", "CCI_14", "ROC_12", "TYPICAL_DEV",
    "EMA_RATIO_12_26", "TREND_STRENGTH_50", "PRICE_POS_50",
    "TRIX_15", "PPO", "ULT_OSC", "RET_ACCEL",
    "GK_VOL", "PARKINSON_VOL", "YANG_ZHANG_VOL", "RS_VOL",
    "AMIHUD_ILLIQ", "KYLE_LAMBDA", "CMF_20", "AD_LINE_SLOPE",
    "STOCH_K_14", "STOCH_D_3", "AROON_OSC_25", "DMI_ADX_14", "DMI_DIFF_14", "TRIX_SIGNAL",
    "DONCHIAN_POS_20", "KELTNER_POS_20", "ICHIMOKU_KIJUN_DEV", "ICHIMOKU_TENKAN_DEV",
    "SUPERTREND_DIR", "SAR_DIST",
    "ROLL_SKEW_20", "ROLL_KURT_20", "HURST_50", "FRACTAL_DIM_30", "AC2", "RET_ENTROPY_20",
    "CS_RANK_RET5", "CS_ZSCORE_RET20",
)


def _as32(raw: dict) -> dict[str, np.ndarray]:
    out = {}
    for key in ("open", "high", "low", "close", "volume"):
        arr = np.asarray(raw[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out[key] = arr
    return out


def _prev(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:, :1], x[:, :-1]], axis=1)


def _rsi(close: np.ndarray, w: int = 14) -> np.ndarray:
    diff = close - _prev(close)
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    rs = (rolling_mean(gains, w) + 1e-9) / (rolling_mean(losses, w) + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi - 50.0) / 50.0


def _ac1(close: np.ndarray, w: int = 20) -> np.ndarray:
    ret = log_ret(close, 1)
    # cat([zeros(N, w), ret]).unfold(w+1) → T windows.
    xp = np.concatenate([np.zeros((close.shape[0], w), dtype=close.dtype), ret], axis=1)
    from numpy.lib.stride_tricks import sliding_window_view
    wnd = sliding_window_view(xp, w + 1, axis=1)
    x, y = wnd[:, :, :-1], wnd[:, :, 1:]
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = ((x - xm) * (y - ym)).mean(axis=-1)
    sx = np.sqrt(((x - xm) ** 2).mean(axis=-1))
    sy = np.sqrt(((y - ym) ** 2).mean(axis=-1))
    return clean(cov / (sx * sy + 1e-8))


def _rvol(close: np.ndarray, w: int = 20) -> np.ndarray:
    ret = log_ret(close, 1)
    wdw = rolling(ret, w)
    m = wdw.mean(axis=-1, keepdims=True)
    return np.sqrt(((wdw - m) ** 2).mean(axis=-1)) + 1e-9


def _boll(close: np.ndarray, w: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ma = rolling_mean(close, w)
    std = rolling_std(close, w)
    upper = ma + 2 * std
    lower = ma - 2 * std
    pos = np.clip((close - lower) / (upper - lower + EPS), 0.0, 1.0)
    width = (upper - lower) / (ma + EPS)
    return pos, width


def _vwap_dev(close, high, low, volume, w: int = 20) -> np.ndarray:
    typical = (high + low + close) / 3.0
    tpv = typical * volume
    vwap = rolling_sum(tpv, w) / (rolling_sum(volume, w) + EPS)
    return (close - vwap) / (vwap + EPS)


def _mfi(close, high, low, volume, w: int = 14) -> np.ndarray:
    typical = (high + low + close) / 3.0
    mf = typical * volume
    pc = _prev(typical)
    pos = np.where(typical > pc, mf, 0.0)
    neg = np.where(typical < pc, mf, 0.0)
    mfr = (rolling_mean(pos, w) * w) / (rolling_mean(neg, w) * w + EPS)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return (mfi - 50.0) / 50.0


def _trix(close: np.ndarray, span: int = 15) -> np.ndarray:
    e3 = ema_feature(ema_feature(ema_feature(close, span), span), span)
    prev = _prev(e3)
    return (e3 - prev) / (np.abs(prev) + EPS)


def _dmi_parts(close, high, low, w: int = 14):
    prev_h = _prev(high)
    prev_l = _prev(low)
    prev_c = _prev(close)
    dm_pos = np.maximum(high - prev_h, 0.0)
    dm_neg = np.maximum(prev_l - low, 0.0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    tr_mean = rolling_mean(tr, w)
    di_pos = rolling_mean(dm_pos, w) / (tr_mean + EPS)
    di_neg = rolling_mean(dm_neg, w) / (tr_mean + EPS)
    return di_pos, di_neg


def _stoch_k(close, high, low, w: int = 14) -> np.ndarray:
    max_h = rolling_max(high, w)
    min_l = rolling_min(low, w)
    return (close - min_l) / (max_h - min_l + EPS)


def compute_features(raw: dict) -> np.ndarray:
    """Return [N, F, T] float32 features in production order."""
    d = _as32(raw)
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    n, t = c.shape
    ret = log_ret(c, 1)
    ret5 = log_ret(c, 5)
    ret20 = log_ret(c, 20)
    feats: list[np.ndarray] = []

    def push(x: np.ndarray) -> None:
        feats.append(np.asarray(x, dtype=np.float32))

    # trend
    push(norm(ret))
    push(norm(ret5))
    push(norm(ret20))
    push(norm(rolling_mean(c, 10) / (rolling_mean(c, 30) + EPS) - 1.0))
    push(norm(linear_slope(c, 20)))
    # volatility
    atr14 = atr(c, h, l, 14)
    push(norm(np.log1p(clean(np.maximum(atr14, 0.0)))))
    rvol = _rvol(c)
    push(norm(np.log1p(clean(np.maximum(rvol, 0.0)))))
    push(norm((h - l) / (c + EPS)))
    ma_atr = rolling_mean(atr14, 20)
    push(norm(atr14 / (ma_atr + EPS) - 1.0))
    # reversal
    ma20 = rolling_mean(c, 20)
    push(norm((c - ma20) / (ma20 + EPS)))
    ma60 = rolling_mean(c, 60)
    push(norm((c - ma60) / (ma60 + EPS)))
    push(clean(np.clip(_rsi(c, 14), -1.0, 1.0)))
    hl = h - l
    no_range = np.abs(hl) < EPS
    ratio = (c - o) / np.where(no_range, 1.0, hl)
    ratio = np.where(no_range, 0.0, ratio)
    push(clean(np.clip(ratio, -1.0, 1.0)))
    push(clean(np.clip(_ac1(c), -1.0, 1.0)))
    # volume
    ma20v = rolling_mean(v, 20)
    vol_ratio = norm(v / (ma20v + EPS))
    push(vol_ratio)
    std20v = rolling_std(v, 20)
    push(clean(np.clip((v - ma20v) / (std20v + EPS), -5.0, 5.0)))
    ret_n = feats[0]
    log_vr = np.log1p(clean(np.maximum(vol_ratio, -0.99)))
    push(clean(np.clip(ts_corr(ret_n, log_vr, 10), -1.0, 1.0)))
    # cross-section (N=1 degenerates)
    if n == 1:
        push(clean(feats[1]))
        push(clean(feats[2]))
        push(clean(feats[6]))
    else:
        push(norm(feats[1] - feats[1].mean(axis=0, keepdims=True)))
        push(norm(feats[2] - feats[2].mean(axis=0, keepdims=True)))
        push(norm(feats[6] - feats[6].mean(axis=0, keepdims=True)))
    # v3
    push(norm(_vwap_dev(c, h, l, v)))
    bpos, bwidth = _boll(c)
    push(clean(bpos))
    push(norm(bwidth))
    macd = ema_feature(c, 12) - ema_feature(c, 26)
    push(norm(macd - ema_feature(macd, 9)))
    sign = np.concatenate([np.zeros((n, 1), dtype=c.dtype), np.sign(c[:, 1:] - c[:, :-1])], axis=1)
    obv = np.cumsum(sign * v, axis=1)
    push(norm(linear_slope(obv, 20)))
    push(clean(np.clip(_mfi(c, h, l, v), -1.0, 1.0)))
    # alpha101-style
    hw = rolling_max(h, 14)
    lw = rolling_min(l, 14)
    willr = (hw - c) / (hw - lw + EPS)
    push(clean(np.clip(willr, -1.0, 0.0)))
    typical = (h + l + c) / 3.0
    ma_t = rolling_mean(typical, 14)
    tw = rolling(typical, 14)
    mad = np.abs(tw - tw.mean(axis=-1, keepdims=True)).mean(axis=-1)
    cci = (typical - ma_t) / (0.015 * mad + EPS)
    push(clean(np.clip(cci / 200.0, -1.0, 1.0)))
    push(norm(log_ret(c, 12) * 0.0 + _roc(c, 12)))
    push(norm((typical - rolling_mean(typical, 20)) / (rolling_mean(typical, 20) + EPS)))
    e12 = ema_feature(c, 12)
    e26 = ema_feature(c, 26)
    push(norm(e12 / (e26 + EPS) - 1.0))
    push(norm(trend_strength(c, 50)))
    max50 = rolling_max(h, 50)
    min50 = rolling_min(l, 50)
    push(clean(np.clip((c - min50) / (max50 - min50 + EPS), 0.0, 1.0)))
    # momentum block is registered AFTER the OHLC vol block in the source list.
    # FEATURE_NAMES above follows _FEATURE_DEFS, which places TRIX before GK.
    # Compute in that order.
    # The pushes above ended at PRICE_POS_50. Next in _FEATURE_DEFS is TRIX.
    # Wait: I already structured FEATURE_NAMES with TRIX before GK. Continue.
    trix = _trix(c, 15)
    push(norm(trix))
    push(norm((e12 - e26) / (np.abs(e26) + EPS)))
    push(clean(_ult_osc(c, h, l)))
    prev5 = np.concatenate([np.zeros((n, 5), dtype=c.dtype), ret5[:, :-5]], axis=1)
    push(norm(ret5 - prev5))
    # OHLC vol
    ln2 = math.log(2.0)
    hl_term = 0.5 * (np.log((h + EPS) / (l + EPS))) ** 2
    co_term = (2.0 * ln2 - 1.0) * (np.log((c + EPS) / (o + EPS))) ** 2
    gk = np.maximum(hl_term - co_term, 0.0)
    push(norm(np.log1p(clean(np.sqrt(rolling_mean(gk, 20))))))
    pk = np.maximum((1.0 / (4.0 * ln2)) * (np.log((h + EPS) / (l + EPS))) ** 2, 0.0)
    push(norm(np.log1p(clean(np.sqrt(rolling_mean(pk, 20))))))
    pc = _prev(c)
    overnight = (np.log((o + EPS) / (pc + EPS))) ** 2
    open_bar = (np.log((o + EPS) / (c + EPS))) ** 2
    rs_bar = (
        np.log((h + EPS) / (c + EPS)) * np.log((h + EPS) / (o + EPS))
        + np.log((l + EPS) / (c + EPS)) * np.log((l + EPS) / (o + EPS))
    )
    yz = (overnight + open_bar + np.maximum(rs_bar, 0.0)) / 3.0
    push(norm(np.log1p(clean(np.sqrt(np.maximum(rolling_mean(yz, 20), 0.0))))))
    push(norm(np.log1p(clean(np.sqrt(np.maximum(rolling_mean(rs_bar, 20), 0.0))))))
    # liquidity
    illiq = np.abs(ret) / (v + EPS)
    push(norm(np.log1p(clean(np.maximum(rolling_mean(illiq, 20), 0.0)))))
    signed_vol = np.sign(ret) * v
    wa = rolling(np.abs(ret), 20)
    ws = rolling(signed_vol, 20)
    ma = wa.mean(axis=-1, keepdims=True)
    ms = ws.mean(axis=-1, keepdims=True)
    cov = ((wa - ma) * (ws - ms)).mean(axis=-1)
    var_s = ((ws - ms) ** 2).mean(axis=-1)
    push(norm(clean(cov / (var_s + EPS))))
    mf_mul = ((c - l) - (h - c)) / (h - l + EPS)
    mfv = mf_mul * v
    cmf = rolling_sum(mfv, 20) / (rolling_sum(v, 20) + EPS)
    push(clean(np.clip(cmf, -1.0, 1.0)))
    ad = np.cumsum(mfv, axis=1)
    push(norm(clean(linear_slope(ad, 20))))
    # oscillators
    pct_k = _stoch_k(c, h, l, 14)
    push(clean(np.clip(pct_k * 2.0 - 1.0, -1.0, 1.0)))
    pct_d = rolling_mean(pct_k, 3)
    push(clean(np.clip(pct_d * 2.0 - 1.0, -1.0, 1.0)))
    w = 25
    wh = rolling(h, w)
    wl = rolling(l, w)
    periods_h = (w - 1) - wh.argmax(axis=-1)
    periods_l = (w - 1) - wl.argmin(axis=-1)
    osc = (w - periods_h) / w - (w - periods_l) / w
    push(clean(np.clip(osc, -1.0, 1.0)))
    di_pos, di_neg = _dmi_parts(c, h, l, 14)
    dx = np.abs(di_pos - di_neg) / (di_pos + di_neg + EPS)
    push(clean(np.clip(rolling_mean(dx, 14), 0.0, 1.0)))
    push(clean(np.clip(di_pos - di_neg, -1.0, 1.0)))
    push(norm(clean(trix - rolling_mean(trix, 9))))
    # channels
    max20 = rolling_max(h, 20)
    min20 = rolling_min(l, 20)
    push(clean(np.clip((c - min20) / (max20 - min20 + EPS), 0.0, 1.0)))
    mid = ema_feature(c, 20)
    rng = ema_feature(atr14, 20)
    upper = mid + 2.0 * rng
    lower = mid - 2.0 * rng
    push(clean(np.clip((c - lower) / (upper - lower + EPS), 0.0, 1.0)))
    kijun = (rolling_max(h, 26) + rolling_min(l, 26)) / 2.0
    push(norm(clean((c - kijun) / (kijun + EPS))))
    tenkan = (rolling_max(h, 9) + rolling_min(l, 9)) / 2.0
    push(norm(clean((c - tenkan) / (tenkan + EPS))))
    push(clean(_supertrend(c, h, l)))
    push(norm(clean(ema_feature(c, 20) - ema_feature(c, 50))))
    # stats
    push(norm(clean(_moment(ret, 20, 3))))
    push(norm(clean(_moment(ret, 20, 4))))
    push(clean(_hurst(c)))
    push(clean(_fractal(c)))
    push(clean(np.clip(_ac_lag(c, 20, 2), -1.0, 1.0)))
    push(clean(_entropy(ret)))
    # cs supplements
    if n == 1:
        push(clean(norm(ret5)))
        push(clean(norm(ret20)))
    else:
        order = np.argsort(ret5, axis=0)
        ranks = np.empty_like(ret5)
        vals = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], ret5.shape).copy()
        np.put_along_axis(ranks, order, vals, axis=0)
        push(clean(np.clip(ranks / (n - 1), 0.0, 1.0)))
        z = (ret20 - ret20.mean(axis=0, keepdims=True)) / (
            ret20.std(axis=0, keepdims=True, ddof=1) + EPS
        )
        push(norm(clean(z)))

    if len(feats) != len(FEATURE_NAMES):
        raise RuntimeError(f"feature count {len(feats)} != {len(FEATURE_NAMES)}")
    stacked = np.stack(feats, axis=1)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def _roc(close: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(close)
    if close.shape[1] > w:
        out[:, w:] = close[:, w:] / (close[:, :-w] + EPS) - 1.0
    return out


def _ult_osc(close, high, low) -> np.ndarray:
    pc = _prev(close)
    true_low = np.minimum(low, pc)
    true_high = np.maximum(high, pc)
    bp = close - true_low
    tr = true_high - true_low

    def _avg(w: int) -> np.ndarray:
        return rolling_sum(bp, w) / (rolling_sum(tr, w) + EPS)

    uo = (4.0 * _avg(7) + 2.0 * _avg(14) + _avg(28)) / 7.0
    return np.clip(uo * 2.0 - 1.0, -1.0, 1.0)


def _moment(ret: np.ndarray, w: int, power: int) -> np.ndarray:
    wdw = rolling(ret, w)
    mean = wdw.mean(axis=-1, keepdims=True)
    diff = wdw - mean
    std = np.sqrt((diff ** 2).mean(axis=-1))
    if power == 3:
        return (diff ** 3).mean(axis=-1) / (std ** 3 + EPS)
    kurt = (diff ** 4).mean(axis=-1) / (std ** 4 + EPS) - 3.0
    return kurt


def _hurst(close: np.ndarray) -> np.ndarray:
    ret = log_ret(close, 1)
    w = 50
    wnd = rolling(ret, w)
    mean = wnd.mean(axis=-1, keepdims=True)
    centered = wnd - mean
    cumdev = np.cumsum(centered, axis=-1)
    r = cumdev.max(axis=-1) - cumdev.min(axis=-1)
    s = np.sqrt((centered ** 2).mean(axis=-1))
    no_signal = (r < EPS) | (s < EPS)
    safe_s = np.where(no_signal, 1.0, s)
    safe_r = np.where(no_signal, 1.0, r)
    hurst = np.log(safe_r / (safe_s + EPS) + EPS) / math.log(w)
    hurst = np.clip(hurst, 0.0, 1.0)
    hurst = np.where(no_signal, 0.5, hurst)
    return clean(hurst * 2.0 - 1.0)


def _fractal(close: np.ndarray) -> np.ndarray:
    w = 30
    wnd = rolling(close, w)
    rng = wnd.max(axis=-1) - wnd.min(axis=-1)
    diff = np.concatenate(
        [np.zeros((close.shape[0], 1), dtype=close.dtype), np.abs(close[:, 1:] - close[:, :-1])],
        axis=1,
    )
    mad = rolling(diff, w).mean(axis=-1)
    frac = np.clip(rng / (mad * math.sqrt(w) + EPS), 0.0, 3.0)
    return clean(frac / 3.0 * 2.0 - 1.0)


def _ac_lag(close: np.ndarray, w: int, lag: int) -> np.ndarray:
    ret = log_ret(close, 1)
    pad = np.zeros((ret.shape[0], w + lag - 1), dtype=ret.dtype)
    from numpy.lib.stride_tricks import sliding_window_view
    wnd = sliding_window_view(np.concatenate([pad, ret], axis=1), w + lag, axis=1)
    x = wnd[:, :, :-lag]
    y = wnd[:, :, lag:]
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = ((x - xm) * (y - ym)).mean(axis=-1)
    sx = np.sqrt(((x - xm) ** 2).mean(axis=-1))
    sy = np.sqrt(((y - ym) ** 2).mean(axis=-1))
    return cov / (sx * sy + 1e-8)


def _entropy(ret: np.ndarray) -> np.ndarray:
    wdw = rolling(ret, 20)
    p_pos = (wdw > 0).mean(axis=-1)
    p_neg = (wdw < 0).mean(axis=-1)
    p_zero = (wdw == 0).mean(axis=-1)

    def _h(p):
        return -p * np.log(p + EPS)

    h = _h(p_pos) + _h(p_neg) + _h(p_zero)
    return np.clip(h / math.log(3), 0.0, 1.0)


def _supertrend(close, high, low) -> np.ndarray:
    atr14 = atr(close, high, low, 14)
    mid = (high + low) / 2.0
    upper = mid + 1.5 * atr14
    lower = mid - 1.5 * atr14
    n, t = close.shape
    direction = np.zeros((n, t), dtype=np.float32)
    prev_u = upper[:, 0].copy()
    prev_l = lower[:, 0].copy()
    for i in range(1, t):
        new_dir = direction[:, i - 1].copy()
        flip_up = close[:, i] > prev_u
        flip_down = close[:, i] < prev_l
        new_dir[flip_up] = 1.0
        new_dir[flip_down] = -1.0
        direction[:, i] = new_dir
        prev_u = upper[:, i]
        prev_l = lower[:, i]
    return direction
