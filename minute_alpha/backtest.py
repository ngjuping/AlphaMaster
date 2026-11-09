"""Walk-forward accounting. Costs are charged on absolute position change."""

from __future__ import annotations

import numpy as np

from minute_alpha.config import BFX_TAKER_FEE, SLIPPAGE_1BP
from minute_alpha.kernels import periods_per_year


def bar_pnl(position: np.ndarray, target: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """position, target: [T] or [N, T]. Cost is subtracted from |Δposition|."""
    pos = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    prev = np.roll(pos, 1, axis=-1)
    prev[..., 0] = 0.0
    turnover = np.abs(pos - prev)
    pnl = pos * tgt - turnover * cost
    return pnl, turnover


def compound_equity(pnl: np.ndarray) -> np.ndarray:
    """Wealth index from summing bar log-PnL, starting at 1."""
    return np.exp(np.cumsum(np.asarray(pnl, dtype=np.float64)))


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough drawdown of a wealth index. 0.2 means 20%."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    return float(dd.max())


def sharpe(pnl: np.ndarray, periods: int) -> float:
    x = np.asarray(pnl, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(x.mean() / sd * np.sqrt(periods))


def summarize(pnl: np.ndarray, turnover: np.ndarray, periods: int) -> dict:
    eq = compound_equity(pnl)
    return {
        "sharpe": sharpe(pnl, periods),
        "mdd": max_drawdown(eq),
        "turnover_mean": float(np.asarray(turnover).mean()),
        "total_log_pnl": float(np.asarray(pnl).sum()),
        "equity_end": float(eq[-1]) if eq.size else 1.0,
        "bars": int(np.asarray(pnl).size),
        "periods_per_year": int(periods),
    }


def cost_1bp() -> float:
    return SLIPPAGE_1BP


def cost_bitfinex() -> float:
    """Published derivatives taker fee. Maker is also zero; taker is the conservative side."""
    return float(BFX_TAKER_FEE)


def annualization(times_seconds: np.ndarray) -> int:
    return periods_per_year(times_seconds)
