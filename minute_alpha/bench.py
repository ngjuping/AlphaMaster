"""Time the rewrite against the production StackVM on one fixed formula list.

Run as ``python -m minute_alpha.bench``. Each role is a fresh process so thread
counts are not inherited from a previous setting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ARTIFACTS = Path("/opt/cursor/artifacts/minute_alpha")


def _synthetic(t: int = 1500, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.0008, size=(1, t))
    close = 30_000 * np.exp(np.cumsum(ret, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.0004, close.shape)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.0004, close.shape)))
    volume = rng.random(close.shape) * 5 + 0.5
    times = np.arange(t, dtype=np.int64) * 60 + 1_700_000_000
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": times,
    }


def _role_rewrite(jobs: int, formulas: int, t: int) -> dict:
    from minute_alpha.features import compute_features
    from minute_alpha.search import make_candidates, search
    from minute_alpha.target import target_log_open
    from minute_alpha.backtest import annualization

    raw = _synthetic(t)
    t0 = time.perf_counter()
    feat = compute_features(raw)
    feature_s = time.perf_counter() - t0
    target = target_log_open(raw["open"])
    candidates = make_candidates(formulas, 20261109)
    t1 = time.perf_counter()
    best, _, _ = search(
        feat,
        target,
        annualization(raw["time"]),
        n_formulas=formulas,
        n_jobs=jobs,
        candidates=candidates,
    )
    search_s = time.perf_counter() - t1
    return {
        "impl": "rewrite",
        "jobs": jobs,
        "feature_s": feature_s,
        "search_s": search_s,
        "total_s": feature_s + search_s,
        "champion": None if best is None else list(best.formula),
        "val_score": None if best is None else best.val_score,
    }


def _role_original(threads: int, formulas: int, t: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    from minute_alpha.search import make_candidates
    from minute_alpha.target import target_log_open
    from model_core.backtest import MT5Backtest
    from model_core.features import MT5FeatureEngineer
    from model_core.vm import StackVM

    raw_np = _synthetic(t)
    raw = {k: torch.tensor(raw_np[k], dtype=torch.float32) for k in ("open", "high", "low", "close", "volume")}
    t0 = time.perf_counter()
    feat = MT5FeatureEngineer.compute_features(raw)
    feature_s = time.perf_counter() - t0
    target = torch.tensor(target_log_open(raw_np["open"]))
    vm = StackVM()
    bt = MT5Backtest(cost_rate=0.0003, periods_per_year=525_960)
    candidates = make_candidates(formulas, 20261109)
    t1 = time.perf_counter()
    scored = 0
    for _name, tokens in candidates:
        res = vm.execute(tokens, feat)
        if res is None:
            continue
        bt.evaluate_fold(res, target, 0, int(t * 0.6), int(t * 0.7), t - 2)
        scored += 1
    search_s = time.perf_counter() - t1
    return {
        "impl": "original_stackvm",
        "threads": threads,
        "torch_threads": torch.get_num_threads(),
        "feature_s": feature_s,
        "eval_s": search_s,
        "total_s": feature_s + search_s,
        "scored": scored,
    }


def _role_engine(threads: int) -> dict:
    """One original training step on a short synthetic book. Batch is reduced.

    The engine writes strategy JSON and checkpoints into the working directory,
    so the process chdirs to a temp folder and does not touch the repo.
    """
    import tempfile

    import torch

    torch.set_num_threads(threads)
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine
    from model_core.features import MT5FeatureEngineer

    raw_np = _synthetic(480)
    raw = {k: torch.tensor(raw_np[k], dtype=torch.float32) for k in ("open", "high", "low", "close", "volume")}
    raw["time"] = torch.tensor(raw_np["time"].reshape(1, -1))
    target = torch.zeros(1, raw["close"].shape[1])
    opened = raw["open"]
    target[:, :-2] = torch.log(opened[:, 2:] / opened[:, 1:-1].clamp(min=1e-8))

    class _DM:
        def __init__(self) -> None:
            self.raw_dict = raw
            self.target_ret = target
            self._feat = None

        @property
        def feat_tensor(self):
            if self._feat is None:
                self._feat = MT5FeatureEngineer.compute_features(self.raw_dict)
            return self._feat

    ModelConfig.BATCH_SIZE = 8
    ModelConfig.TRAIN_STEPS = 1
    ModelConfig.REWARD_MODE = "ftmo"
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        engine = AlphaEngine(data_manager=_DM(), use_lord_regularization=False, n_folds=3, target_symbol="BTC")
        t0 = time.perf_counter()
        engine.train(start_step=0, end_step=1, verbose_header=False)
        elapsed = time.perf_counter() - t0
    return {
        "impl": "original_engine_one_step",
        "threads": torch.get_num_threads(),
        "batch": 8,
        "seconds": elapsed,
        "best_score": engine.best_score,
    }


def _spawn(role: str, extra: list[str]) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    cmd = [sys.executable, "-m", "minute_alpha.bench", "--role", role, *extra]
    proc = subprocess.run(cmd, cwd="/workspace", env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return {"impl": role, "error": proc.stderr[-2000:], "stdout": proc.stdout[-1000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="drive")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--formulas", type=int, default=32)
    parser.add_argument("--bars", type=int, default=1500)
    args = parser.parse_args()
    if args.role == "rewrite":
        print(json.dumps(_role_rewrite(args.jobs, args.formulas, args.bars)))
        return
    if args.role == "original":
        print(json.dumps(_role_original(args.threads, args.formulas, args.bars)))
        return
    if args.role == "engine":
        try:
            print(json.dumps(_role_engine(args.threads)))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"impl": "original_engine_one_step", "error": f"{type(exc).__name__}: {exc}"}))
        return

    cpus = os.cpu_count() or 1
    formulas = args.formulas
    bars = args.bars
    common = ["--formulas", str(formulas), "--bars", str(bars)]
    results = {
        "rewrite_1": _spawn("rewrite", ["--jobs", "1", *common]),
        "rewrite_multi": _spawn("rewrite", ["--jobs", str(cpus), *common]),
        "original_1": _spawn("original", ["--threads", "1", *common]),
        "original_multi": _spawn("original", ["--threads", str(cpus), *common]),
        "engine_1": _spawn("engine", ["--threads", "1"]),
        "cpu_count": cpus,
        "formulas": formulas,
        "bars": bars,
    }
    r1 = results["rewrite_1"].get("total_s")
    rm = results["rewrite_multi"].get("total_s")
    o1 = results["original_1"].get("total_s")
    om = results["original_multi"].get("total_s")
    results["speedup_rewrite"] = (r1 / rm) if r1 and rm else None
    results["speedup_original_threads"] = (o1 / om) if o1 and om else None
    results["rewrite_vs_original_1core"] = (o1 / r1) if r1 and o1 else None
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "timing_vs_original.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("speedup_rewrite", "speedup_original_threads", "rewrite_vs_original_1core")}, indent=2))


if __name__ == "__main__":
    main()
