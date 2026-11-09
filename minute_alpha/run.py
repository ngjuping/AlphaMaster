"""Entry point. Set thread env before NumPy so one process stays one core."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from minute_alpha.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain production factors on Bitfinex 1m bars")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--formulas", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--budget", type=float, default=12 * 60)
    args = parser.parse_args()
    kwargs = {"n_jobs": args.jobs, "budget_seconds": args.budget}
    if args.bars is not None:
        kwargs["bars"] = args.bars
    if args.formulas is not None:
        kwargs["n_formulas"] = args.formulas
    if args.retrain_every is not None:
        kwargs["retrain_every"] = args.retrain_every
    if args.window is not None:
        kwargs["train_window"] = args.window
    result = run(**kwargs)
    print(
        f"retrain 1-core {result['one_core_seconds']:.2f}s  "
        f"multi {result['multi_core_seconds']:.2f}s  "
        f"speedup {result['speedup']:.2f}x  "
        f"every {result['retrain_every_min']} min"
    )
    print(f"OOS sharpe 1bp {result['metrics_1bp']['sharpe']:.3f}  "
          f"bfx {result['metrics_bfx']['sharpe']:.3f}")


if __name__ == "__main__":
    main()
