"""Stack VM and the causal output normalization used before tanh."""

from __future__ import annotations

import numpy as np

from minute_alpha.features import FEATURE_NAMES
from minute_alpha.kernels import rolling
from minute_alpha.operators import ARITY, OPERATORS, OPERATOR_NAMES, ShapeError

FEAT_OFFSET = len(FEATURE_NAMES)
VOCAB = FEATURE_NAMES + OPERATOR_NAMES
NAME_TO_ID = {name: i for i, name in enumerate(VOCAB)}


def tokens_from_names(names: tuple[str, ...] | list[str]) -> list[int]:
    missing = [n for n in names if n not in NAME_TO_ID]
    if missing:
        raise KeyError(f"unknown tokens: {missing}")
    return [NAME_TO_ID[n] for n in names]


def decode(tokens: list[int] | tuple[int, ...]) -> str:
    parts = []
    for tok in tokens:
        if 0 <= int(tok) < len(VOCAB):
            parts.append(VOCAB[int(tok)])
        else:
            parts.append(f"#{tok}")
    return " → ".join(parts)


def normalize_output(x: np.ndarray) -> np.ndarray:
    """Match ``StackVM._normalize_output``.

    N>1: cross-sectional z-score at each bar (sample std).
    N=1 and T>=500: causal rolling z-score, window 500, warm-up set to 0.
    N=1 and T<500: causal expanding z-score.
    Result clipped to [-3, 3]. A globally constant input is returned as-is.
    """
    x = np.asarray(x, dtype=np.float32)
    n, t = x.shape
    if n * t > 1 and float(x.std(ddof=1)) < 1e-6:
        return x
    if n > 1:
        mean = x.mean(axis=0, keepdims=True)
        std = np.maximum(x.std(axis=0, keepdims=True, ddof=1), np.float32(1e-8))
        return np.clip((x - mean) / std, -3.0, 3.0)
    if t < 500:
        cnt = np.arange(1, t + 1, dtype=np.float32).reshape(1, t)
        csum = np.cumsum(x, axis=1)
        mean = csum / cnt
        var = np.maximum(np.cumsum(x * x, axis=1) / cnt - mean * mean, np.float32(1e-8))
        return np.clip((x - mean) / np.sqrt(var), -3.0, 3.0)
    windows = rolling(x, 500)
    mean = windows.mean(axis=-1)
    var = ((windows - mean[..., None]) ** 2).sum(axis=-1) / np.float32(499.0)
    std = np.maximum(np.sqrt(np.maximum(var, np.float32(0.0))), np.float32(1e-8))
    z = (x - mean) / std
    z[:, :499] = 0.0
    return np.clip(z, -3.0, 3.0)


def execute(tokens: list[int] | tuple[int, ...], feat: np.ndarray) -> np.ndarray | None:
    """Execute one formula. ``feat`` is [N, F, T]. Returns [N, T] or None."""
    stack: list[np.ndarray] = []
    feat_offset = feat.shape[1]
    try:
        for token in tokens:
            token = int(token)
            if token < feat_offset:
                stack.append(np.asarray(feat[:, token, :], dtype=np.float32))
                continue
            op_i = token - FEAT_OFFSET
            if op_i < 0 or op_i >= len(OPERATORS):
                return None
            _name, arity, fn = OPERATORS[op_i]
            if len(stack) < arity:
                return None
            args = [stack.pop() for _ in range(arity)]
            args.reverse()
            res = np.asarray(fn(*args), dtype=np.float32)
            if not np.isfinite(res).all():
                res = np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
            stack.append(res)
    except (ShapeError, ValueError, FloatingPointError):
        return None
    if len(stack) != 1:
        return None
    return normalize_output(stack[0])


def arity_map() -> dict[int, int]:
    return {FEAT_OFFSET + i: arity for i, (_n, arity, _f) in enumerate(OPERATORS)}
