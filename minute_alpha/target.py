"""Training label used by the production data manager.

``target[n, t] = log(open[n, t+2] / open[n, t+1])`` for t <= T-3, else 0.

This is a one-bar execution lag: the factor known at the close of bar t is
scored on the next bar's open-to-open log return, not on the simple
close-to-close return of bar t+1. The last two columns are 0 because those
opens are not known yet.
"""

from __future__ import annotations

import numpy as np


def target_log_open(open_px: np.ndarray) -> np.ndarray:
    open_px = np.asarray(open_px, dtype=np.float64)
    if open_px.ndim == 1:
        open_px = open_px.reshape(1, -1)
    n, t = open_px.shape
    target = np.zeros((n, t), dtype=np.float64)
    if t >= 3:
        denom = open_px[:, 1:-1].copy()
        denom[denom == 0.0] = 1.0
        target[:, : t - 2] = np.log(open_px[:, 2:] / denom)
    return target
