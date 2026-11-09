"""Deterministic formula search.

Candidates are the published anchor formulas plus seeded samples from the same
constrained token grammar as ``ConstrainedSampler``. Formulas are scored in
parallel; the correlation penalty is applied afterwards in a fixed order so
the winner does not depend on process scheduling.

This is the production objective (FTMO walk-forward score, factor-pool
correlation penalty, champion = best penalized validation score). It is not
the 9000-step REINFORCE policy. That policy cannot finish inside a minute;
the benchmark reports its step time separately.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from minute_alpha.config import (
    ANCHOR_FORMULAS,
    CORR_PENALTY,
    CORR_THRESHOLD,
    FACTOR_TOP_K,
    MAX_FORMULA_LEN,
    SEED,
)
from minute_alpha.operators import INFECTED, POSITIVE_ONLY, RESTORE
from minute_alpha.score import repetition_penalty, score_formula, walk_forward_folds
from minute_alpha.vm import FEAT_OFFSET, VOCAB, arity_map, execute, tokens_from_names

_DELTA = {}
_AMAP = arity_map()
for _tid in range(len(VOCAB)):
    if _tid < FEAT_OFFSET:
        _DELTA[_tid] = 1
    else:
        _DELTA[_tid] = 1 - _AMAP.get(_tid, 1)

_POS_IDS = {i for i, name in enumerate(VOCAB) if name in POSITIVE_ONLY}
_INF_IDS = {i for i, name in enumerate(VOCAB) if name in INFECTED}
_RES_IDS = {i for i, name in enumerate(VOCAB) if name in RESTORE}

_CTX: dict = {}


@dataclass(frozen=True)
class Candidate:
    formula: tuple[int, ...]
    train_score: float
    val_score: float
    name: str


def _valid_mask(stack: int, step: int, total: int, infected: int) -> np.ndarray:
    remaining = total - step
    mask = np.ones(len(VOCAB), dtype=bool)
    for tid in range(len(VOCAB)):
        new_depth = stack + _DELTA[tid]
        if new_depth < 1:
            mask[tid] = False
            continue
        min_future = new_depth + (remaining - 1) * (-2)
        max_future = new_depth + (remaining - 1) * 1
        if 1 < min_future or 1 > max_future:
            mask[tid] = False
        if infected >= 2 and tid in _INF_IDS:
            mask[tid] = False
        if infected >= 3 and (tid in _INF_IDS or tid in _POS_IDS):
            mask[tid] = False
    if not mask.any():
        for tid in range(len(VOCAB)):
            if stack + _DELTA[tid] >= 1:
                mask[tid] = True
    return mask


def _update_infection(token: int, chain: int) -> int:
    if token in _POS_IDS:
        return chain + 1
    if token in _RES_IDS:
        return 0
    if token in _INF_IDS and chain > 0:
        return chain + 1
    return chain


def sample_formula(rng: np.random.Generator, length: int = MAX_FORMULA_LEN) -> list[int]:
    stack = 0
    infected = 0
    tokens: list[int] = []
    for step in range(length):
        mask = _valid_mask(stack, step, length, infected)
        choices = np.flatnonzero(mask)
        tid = int(rng.choice(choices))
        tokens.append(tid)
        stack += _DELTA[tid]
        infected = _update_infection(tid, infected)
    return tokens


def anchor_formulas() -> list[tuple[str, list[int]]]:
    out = []
    for name, tokens in ANCHOR_FORMULAS:
        out.append((name, tokens_from_names(tokens)))
    return out


def make_candidates(n_formulas: int, seed: int = SEED) -> list[tuple[str, list[int]]]:
    """Anchors first, then seeded samples. Identical on every call."""
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, toks in anchor_formulas():
        key = tuple(toks)
        if key not in seen:
            seen.add(key)
            chosen.append((name, toks))
    while len(chosen) < n_formulas:
        toks = sample_formula(rng)
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        chosen.append((f"sample_{len(chosen)}", toks))
    return chosen


def _eval_task(item: tuple[str, list[int]]) -> tuple:
    name, formula = item
    feat = _CTX["feat"]
    target = _CTX["target"]
    periods = _CTX["periods"]
    cost = _CTX["cost"]
    factor = execute(formula, feat)
    if factor is None:
        return name, tuple(formula), None, -5.0, -5.0
    if factor.size > 1 and float(factor.std(ddof=1)) < 1e-4:
        return name, tuple(formula), None, -2.0, -2.0
    train, val = score_formula(factor, target, periods, cost=cost)
    pen = repetition_penalty(formula)
    return name, tuple(formula), factor, train - pen, val - pen


def _corr_penalty(score: float, factor: np.ndarray, pool: list[np.ndarray], sl: tuple[int, int]) -> float:
    if not pool or factor is None:
        return score
    s, e = sl
    a = np.asarray(factor[:, s:e], dtype=np.float64).ravel()
    if a.size == 0 or float(a.std(ddof=0)) < 1e-4:
        return score
    ac = a - a.mean()
    na = float(np.linalg.norm(ac)) + 1e-8
    for prev in pool:
        b = np.asarray(prev[:, s:e], dtype=np.float64).ravel()
        bc = b - b.mean()
        corr = abs(float(np.dot(ac, bc)) / (na * (float(np.linalg.norm(bc)) + 1e-8)))
        if corr > CORR_THRESHOLD:
            return score * CORR_PENALTY
    return score


def search(
    feat: np.ndarray,
    target: np.ndarray,
    periods: int,
    n_formulas: int,
    n_jobs: int = 1,
    seed: int = SEED,
    cost: float = None,
    candidates: list[tuple[str, list[int]]] | None = None,
) -> tuple[Candidate | None, list[Candidate], list[np.ndarray]]:
    """Return the champion, the penalized ranking, and pool factor arrays.

    Pool arrays line up with the ranking only for formulas that received a
    finite factor. The champion is the highest penalized validation score.
    """
    from minute_alpha.config import TRAIN_COST_RATE

    if cost is None:
        cost = TRAIN_COST_RATE
    if candidates is None:
        candidates = make_candidates(n_formulas, seed)
    else:
        candidates = list(candidates)[:n_formulas] if n_formulas else list(candidates)

    global _CTX
    _CTX = {"feat": feat, "target": target, "periods": int(periods), "cost": float(cost)}

    if n_jobs <= 1 or len(candidates) <= 1:
        raw = [_eval_task(item) for item in candidates]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            raw = pool.map(_eval_task, candidates, chunksize=1)

    folds = walk_forward_folds(feat.shape[1])
    sl = (folds[0]["train_start"], folds[0]["train_end"])
    pool_factors: list[np.ndarray] = []
    ranked: list[Candidate] = []
    best: Candidate | None = None
    best_factor: np.ndarray | None = None
    # Min-heap of (val, seq, factor) would drop low scores. Keep top-K by
    # walking in candidate order, matching the engine's sequential pool.
    heap: list[tuple[float, int, np.ndarray]] = []
    for seq, (name, formula, factor, train, val) in enumerate(raw):
        if factor is None:
            ranked.append(Candidate(formula, train, val, name))
            continue
        train_p = _corr_penalty(train, factor, [h[2] for h in heap], sl)
        val_p = _corr_penalty(val, factor, [h[2] for h in heap], sl)
        cand = Candidate(formula, train_p, val_p, name)
        ranked.append(cand)
        if len(heap) < FACTOR_TOP_K:
            heap.append((val_p, seq, factor))
        elif val_p > min(heap, key=lambda item: item[0])[0]:
            worst = min(range(len(heap)), key=lambda i: heap[i][0])
            heap[worst] = (val_p, seq, factor)
        if best is None or val_p > best.val_score:
            best = cand
            best_factor = factor
    pool_factors = [item[2] for item in heap]
    return best, ranked, pool_factors if best_factor is None else pool_factors
