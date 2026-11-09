"""Signal diagnostics. These are not strategy returns.

Group 1 multiplies the standardized factor (the value tanh sees) by the
training label. Group 2 multiplies the target position by that same label.
Neither subtracts fees or slippage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def time_utc(unix_seconds: int) -> str:
    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _segment(index: int, oos_start: int) -> str:
    return "oos" if index >= oos_start else "train"


def factor_target_frame(
    times: np.ndarray,
    symbols: list[str],
    factors: dict[str, np.ndarray],
    target: np.ndarray,
    oos_start: int,
    factor_name: str,
) -> pd.DataFrame:
    """Per symbol and an equal-weight aggregate of factor × target."""
    rows = []
    products = []
    for i, symbol in enumerate(symbols):
        fac = np.asarray(factors[symbol], dtype=np.float64)
        tgt = np.asarray(target[i], dtype=np.float64)
        prod = fac * tgt
        products.append(prod)
        csum = np.cumsum(prod)
        oos = np.full(len(prod), np.nan)
        if oos_start < len(prod):
            oos[oos_start:] = np.cumsum(prod[oos_start:])
        for t in range(len(prod)):
            rows.append(
                {
                    "time": int(times[t]),
                    "time_utc": time_utc(int(times[t])),
                    "symbol": symbol,
                    "factor_name": factor_name if isinstance(factor_name, str) else factor_name[symbol],
                    "segment": _segment(t, oos_start),
                    "factor": fac[t],
                    "target": tgt[t],
                    "product": prod[t],
                    "cumsum": csum[t],
                    "oos_cumsum": oos[t],
                }
            )
    stacked = np.vstack(products)
    mean_prod = stacked.mean(axis=0)
    sum_prod = stacked.sum(axis=0)
    csum = np.cumsum(mean_prod)
    oos = np.full(len(mean_prod), np.nan)
    if oos_start < len(mean_prod):
        oos[oos_start:] = np.cumsum(mean_prod[oos_start:])
    name = "equal_weight_mean"
    for t in range(len(mean_prod)):
        rows.append(
            {
                "time": int(times[t]),
                "time_utc": time_utc(int(times[t])),
                "symbol": "ALL",
                "factor_name": name,
                "segment": _segment(t, oos_start),
                "factor": np.nan,
                "target": np.nan,
                "product": mean_prod[t],
                "product_sum": sum_prod[t],
                "cumsum": csum[t],
                "oos_cumsum": oos[t],
            }
        )
    return pd.DataFrame(rows)


def position_target_frame(
    times: np.ndarray,
    symbols: list[str],
    positions: np.ndarray,
    target: np.ndarray,
    oos_start: int,
) -> pd.DataFrame:
    rows = []
    products = []
    for i, symbol in enumerate(symbols):
        pos = positions[i]
        tgt = target[i]
        prod = pos * tgt
        products.append(prod)
        csum = np.cumsum(prod)
        oos = np.full(len(prod), np.nan)
        if oos_start < len(prod):
            oos[oos_start:] = np.cumsum(prod[oos_start:])
        for t in range(len(prod)):
            rows.append(
                {
                    "time": int(times[t]),
                    "time_utc": time_utc(int(times[t])),
                    "symbol": symbol,
                    "segment": _segment(t, oos_start),
                    "position": pos[t],
                    "target": tgt[t],
                    "product": prod[t],
                    "cumsum": csum[t],
                    "oos_cumsum": oos[t],
                }
            )
    mean_prod = np.vstack(products).mean(axis=0)
    csum = np.cumsum(mean_prod)
    oos = np.full(len(mean_prod), np.nan)
    if oos_start < len(mean_prod):
        oos[oos_start:] = np.cumsum(mean_prod[oos_start:])
    for t in range(len(mean_prod)):
        rows.append(
            {
                "time": int(times[t]),
                "time_utc": time_utc(int(times[t])),
                "symbol": "ALL",
                "segment": _segment(t, oos_start),
                "position": float(positions[:, t].mean()),
                "target": np.nan,
                "product": mean_prod[t],
                "cumsum": csum[t],
                "oos_cumsum": oos[t],
            }
        )
    return pd.DataFrame(rows)


def position_history_frame(
    times: np.ndarray,
    symbols: list[str],
    positions: np.ndarray,
    factors: np.ndarray,
    names,
    oos_start: int,
    valid_from: np.ndarray,
) -> pd.DataFrame:
    """One row per bar and symbol, plus an equal-weight ALL row.

    ``valid_from[i]`` is the first index whose position for symbol i was
    produced by a fitted formula. Earlier rows are omitted.
    """
    rows = []
    start = int(valid_from.min()) if len(valid_from) else 0
    for t in range(start, len(times)):
        ready = []
        for i, symbol in enumerate(symbols):
            if t < int(valid_from[i]):
                continue
            ready.append(positions[i, t])
            rows.append(
                {
                    "time": int(times[t]),
                    "time_utc": time_utc(int(times[t])),
                    "segment": _segment(t, oos_start),
                    "symbol": symbol,
                    "target_position": float(positions[i, t]),
                    "factor_value": float(factors[i, t]),
                    "factor_name": names[i, t] if isinstance(names, np.ndarray) else names.get(symbol, ""),
                }
            )
        if ready:
            rows.append(
                {
                    "time": int(times[t]),
                    "time_utc": time_utc(int(times[t])),
                    "segment": _segment(t, oos_start),
                    "symbol": "ALL",
                    "target_position": float(np.mean(ready)),
                    "factor_value": np.nan,
                    "factor_name": "equal_weight_mean",
                }
            )
    return pd.DataFrame(rows)


def _vline(ax, times: np.ndarray, oos_start: int) -> None:
    if 0 <= oos_start < len(times):
        ax.axvline(times[oos_start], color="black", ls="--", lw=1, label="OOS start")


def plot_factor_cumsum(frame: pd.DataFrame, times: np.ndarray, oos_start: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for symbol, sub in frame.groupby("symbol"):
        sub = sub.sort_values("time")
        ax.plot(sub["time"], sub["cumsum"], lw=1.2 if symbol == "ALL" else 0.9, label=symbol)
    _vline(ax, times, oos_start)
    ax.set_title("Diagnostic: standardized factor × training label (cumsum)")
    ax.set_xlabel("unix seconds")
    ax.set_ylabel("cumsum of factor × target")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_position_cumsum(frame: pd.DataFrame, times: np.ndarray, oos_start: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for symbol, sub in frame.groupby("symbol"):
        sub = sub.sort_values("time")
        ax.plot(sub["time"], sub["cumsum"], lw=1.2 if symbol == "ALL" else 0.9, label=symbol)
    _vline(ax, times, oos_start)
    ax.set_title("Diagnostic: target position × training label (cumsum, no costs)")
    ax.set_xlabel("unix seconds")
    ax.set_ylabel("cumsum of position × target")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_equity(times, equity_1bp, equity_bfx, oos_start: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(times, equity_1bp, label="0 fee + 1bp slippage", lw=1.4)
    ax.plot(times, equity_bfx, label="Bitfinex taker fee (published 0)", lw=1.2)
    _vline(ax, times, oos_start)
    ax.set_title("Walk-forward wealth (exp of cumulative log-PnL)")
    ax.set_xlabel("unix seconds")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
