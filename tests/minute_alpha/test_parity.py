"""Parity against the production feature stack and StackVM.

N=1 is the production inference shape. Cross-sectional z-score amplification
on numerically flat bars is covered in the report, not in the strict bound.
"""

from __future__ import annotations

import numpy as np
import torch

from minute_alpha.config import ANCHOR_FORMULAS
from minute_alpha.features import FEATURE_NAMES, compute_features
from minute_alpha.operators import OPERATOR_NAMES
from minute_alpha.search import sample_formula
from minute_alpha.vm import execute, tokens_from_names
from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB


def _ohlcv(n: int, t: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, size=(n, t))
    close = 100 * np.exp(np.cumsum(ret, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    high = np.maximum(close * (1 + np.abs(rng.normal(0, 0.0008, (n, t)))), np.maximum(open_, close))
    low = np.minimum(close * (1 - np.abs(rng.normal(0, 0.0008, (n, t)))), np.minimum(open_, close))
    volume = rng.random((n, t)) * 10 + 0.1
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_vocab_order_matches_production():
    assert tuple(FEATURE_NAMES) == tuple(FORMULA_VOCAB.feature_names)
    assert tuple(OPERATOR_NAMES) == tuple(FORMULA_VOCAB.operator_names)


def test_anchor_formulas_match_stackvm():
    raw = _ohlcv(1, 900, 7)
    ours = compute_features(raw)
    theirs = MT5FeatureEngineer.compute_features(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    )
    vm = StackVM()
    for _name, tokens in ANCHOR_FORMULAS:
        ids = tokens_from_names(tokens)
        got = execute(ids, ours)
        exp = vm.execute(ids, theirs).numpy()
        assert got is not None
        np.testing.assert_allclose(got, exp, atol=1e-4, rtol=1e-4)


def test_vm_matches_on_identical_features():
    raw = _ohlcv(1, 700, 11)
    feat = MT5FeatureEngineer.compute_features(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    )
    vm = StackVM()
    rng = np.random.Generator(np.random.PCG64(5))
    for _ in range(12):
        tokens = sample_formula(rng)
        got = execute(tokens, feat.numpy())
        exp = vm.execute(tokens, feat)
        assert got is not None and exp is not None
        # Rolling z-score std differs by ~1e-4 from torch's Welford reduction.
        np.testing.assert_allclose(got, exp.numpy(), atol=2e-4, rtol=1e-3)


def test_feature_bulk_is_close():
    raw = _ohlcv(1, 800, 4)
    got = compute_features(raw)
    exp = MT5FeatureEngineer.compute_features(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    ).numpy()
    diff = np.abs(got - exp)
    assert np.median(diff) < 1e-5
    close = sum(float(diff[:, i].max()) < 1e-3 for i in range(diff.shape[1]))
    assert close >= 50
