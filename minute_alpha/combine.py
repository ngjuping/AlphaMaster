"""Champion selection is the production combiner.

Live trading stores one formula per symbol (``strategies/best_{symbol}.json``).
During mining, a top-K pool applies a correlation penalty on the training
slice only. The deployed signal is the single best penalized validation score,
not a blend. Across symbols the book is an equal-weight mean of positions.
"""

from __future__ import annotations

import numpy as np

from minute_alpha.position import target_positions
from minute_alpha.search import Candidate


def equal_weight(positions: np.ndarray) -> np.ndarray:
    """``positions`` is [N, T]. Return the cross-sectional mean, shape [T]."""
    return np.asarray(positions, dtype=np.float64).mean(axis=0)


def champion_position(factor: np.ndarray) -> np.ndarray:
    return target_positions(factor)


def describe(candidate: Candidate | None) -> dict:
    if candidate is None:
        return {"name": None, "formula": None, "val_score": None, "train_score": None}
    return {
        "name": candidate.name,
        "formula": list(candidate.formula),
        "train_score": candidate.train_score,
        "val_score": candidate.val_score,
    }
