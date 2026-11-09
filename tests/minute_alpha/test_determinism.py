"""Same seed and same candidates produce the same champion on 1 and 2 cores."""

from __future__ import annotations

import numpy as np

from minute_alpha.features import compute_features
from minute_alpha.search import make_candidates, search
from minute_alpha.target import target_log_open
from minute_alpha.vm import execute


def _raw(t: int = 480, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, size=(1, t))
    close = 100 * np.exp(np.cumsum(ret, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    high = np.maximum(close, open_) * 1.001
    low = np.minimum(close, open_) * 0.999
    volume = np.full_like(close, 3.0)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_candidates_repeatable():
    assert make_candidates(10, 123) == make_candidates(10, 123)


def test_search_repeatable_across_jobs():
    raw = _raw()
    feat = compute_features(raw)
    target = target_log_open(raw["open"])
    candidates = make_candidates(8, 99)
    a, _, _ = search(feat, target, periods=500_000, n_formulas=8, n_jobs=1, candidates=candidates)
    b, _, _ = search(feat, target, periods=500_000, n_formulas=8, n_jobs=1, candidates=candidates)
    c, _, _ = search(feat, target, periods=500_000, n_formulas=8, n_jobs=2, candidates=candidates)
    assert a is not None and b is not None and c is not None
    assert a.formula == b.formula == c.formula
    assert a.val_score == b.val_score == c.val_score


def test_sampled_formulas_execute():
    raw = _raw(360, 8)
    feat = compute_features(raw)
    rng = np.random.Generator(np.random.PCG64(1))
    for _ in range(6):
        out = execute(sample_ok(rng), feat)
        assert out is not None
        assert out.shape == (1, 360)


def sample_ok(rng):
    from minute_alpha.search import sample_formula

    return sample_formula(rng)
