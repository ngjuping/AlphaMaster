"""Production position map: tanh, then a 0.05 dead zone, clipped to [-1, 1]."""

from __future__ import annotations

import numpy as np

from minute_alpha.config import MIN_TRADE_EXPOSURE


def target_positions(factor: np.ndarray, min_exposure: float = MIN_TRADE_EXPOSURE) -> np.ndarray:
    """Map a standardized factor to a continuous position in [-1, 1].

    ``position = tanh(factor)``. Values with absolute size below
    ``min_exposure`` (production ``MIN_TRADE_EXPOSURE``) become 0. The result
    is the fraction of notional to hold over the next bar: +1 is fully long,
    -1 is fully short. It is not a lot-size and it is not re-levered.
    """
    pos = np.tanh(np.asarray(factor, dtype=np.float64))
    if min_exposure > 0:
        pos = np.where(np.abs(pos) >= min_exposure, pos, 0.0)
    return np.clip(pos, -1.0, 1.0)
