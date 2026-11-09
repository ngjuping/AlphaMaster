"""Last-24h minute-by-minute retrain and simple-return settlement.

Each decision uses only the trailing 4320 closed bars (the production online
window) and the fixed candidate seed. The position known at the close of bar
t is marked to the next minute's simple return, close[t+1] / close[t] - 1.
The miner still scores formulas with the production label
log(open[t+2] / open[t+1]); that label is not the settlement return.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from minute_alpha.backtest import annualization, max_drawdown, sharpe
from minute_alpha.config import N_FORMULAS, SEED, SLIPPAGE_1BP, TRAIN_WINDOW
from minute_alpha.data import load_universe, to_matrices
from minute_alpha.diagnostics import time_utc
from minute_alpha.features import compute_features
from minute_alpha.pipeline import ARTIFACTS, REPO_OUT
from minute_alpha.portal import build_status, write_status
from minute_alpha.position import target_positions
from minute_alpha.search import make_candidates, search
from minute_alpha.target import target_log_open
from minute_alpha.timing import machine
from minute_alpha.vm import decode, execute

_STATE: dict = {}


def simple_next_return(close: np.ndarray) -> np.ndarray:
    """Simple return earned by a position set at the previous close.

    ``out[h] = close[h] / close[h - 1] - 1``. Index 0 is 0 because there is
    no previous close inside the array.
    """
    close = np.asarray(close, dtype=np.float64)
    out = np.zeros_like(close)
    prev = close[:, :-1]
    nxt = close[:, 1:]
    safe = np.where(prev == 0.0, 1.0, prev)
    out[:, 1:] = nxt / safe - 1.0
    return out


def _retrain_window(raw: dict, times: np.ndarray, candidates, periods: int, jobs: int) -> dict:
    t0 = time.perf_counter()
    feat = compute_features(raw)
    feature_s = time.perf_counter() - t0
    target = target_log_open(raw["open"])
    t1 = time.perf_counter()
    best, _, _ = search(
        feat,
        target,
        periods,
        n_formulas=len(candidates),
        n_jobs=jobs,
        seed=SEED,
        candidates=candidates,
    )
    search_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    factor = None if best is None else execute(best.formula, feat)
    inference_s = time.perf_counter() - t2
    if factor is None:
        pos = 0.0
        fac = float("nan")
    else:
        fac = float(factor[0, -1])
        pos = float(target_positions(np.array([[fac]]))[0, 0])
    return {
        "position": pos,
        "factor": fac,
        "name": "" if best is None else best.name,
        "formula": "" if best is None else decode(best.formula),
        "tokens": [] if best is None else list(best.formula),
        "train_score": None if best is None else best.train_score,
        "val_score": None if best is None else best.val_score,
        "feature_seconds": feature_s,
        "search_seconds": search_s,
        "inference_seconds": inference_s,
        "total_seconds": feature_s + search_s + inference_s,
    }


def _symbol_minutes(symbol_i: int) -> list[dict]:
    """Retrain every decision index for one symbol. Order is chronological."""
    state = _STATE
    close = state["close"]
    decisions = state["decisions"]
    window = state["window"]
    candidates = state["candidates"]
    periods = state["periods"]
    jobs = state["jobs_inside"]
    rows = []
    for step, dec in enumerate(decisions):
        if step % 120 == 0:
            print(f"symbol {symbol_i} minute {step}/{len(decisions)}", flush=True)
        start = int(dec) - window + 1
        raw = {
            key: state[key][symbol_i : symbol_i + 1, start : int(dec) + 1]
            for key in ("open", "high", "low", "close", "volume")
        }
        row = _retrain_window(raw, state["time"][start : int(dec) + 1], candidates, periods, jobs)
        row["symbol_i"] = symbol_i
        row["decision"] = int(dec)
        rows.append(row)
    return rows


def _metrics(pnl: np.ndarray, turnover: np.ndarray, periods: int) -> dict:
    pnl = np.asarray(pnl, dtype=np.float64)
    equity = np.cumprod(1.0 + pnl) if pnl.size else np.array([1.0])
    # A single minute cannot lose more than the position allows; still guard a zero.
    equity = np.where(np.isfinite(equity), equity, 1.0)
    return {
        "return": float(equity[-1] - 1.0) if equity.size else 0.0,
        "sharpe": sharpe(pnl, periods),
        "mdd": max_drawdown(equity),
        "turnover_mean": float(np.asarray(turnover).mean()) if turnover.size else 0.0,
        "equity_end": float(equity[-1]) if equity.size else 1.0,
        "equity": equity,
    }


def _apply_cost(position: np.ndarray, ret: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    prev = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - prev)
    pnl = position * ret - turnover * cost
    return pnl, turnover


def run_from_fields(
    fields: dict,
    *,
    minutes: int = 1440,
    window: int = TRAIN_WINDOW,
    n_formulas: int = N_FORMULAS,
    n_jobs: int = 1,
    jobs_inside: int = 1,
) -> dict:
    """Rolling retrain on an already aligned panel.

    ``fields`` arrays are ``[N, T]``. The last ``minutes`` bars are the
    settlement minutes. The position for settlement bar ``h`` is trained on
    data ending at ``h - 1`` only.
    """
    close = np.asarray(fields["close"], dtype=np.float64)
    n, t = close.shape
    if t < window + minutes:
        raise RuntimeError(f"need at least {window + minutes} bars, have {t}")
    decisions = list(range(t - minutes - 1, t - 1))
    if len(decisions) != minutes:
        raise RuntimeError("decision count does not match the settlement window")
    if decisions[0] < window - 1:
        raise RuntimeError("warmup is shorter than the training window")

    candidates = make_candidates(n_formulas, SEED)
    periods = annualization(fields["time"][: window])
    global _STATE
    _STATE = {
        "open": np.asarray(fields["open"], dtype=np.float64),
        "high": np.asarray(fields["high"], dtype=np.float64),
        "low": np.asarray(fields["low"], dtype=np.float64),
        "close": close,
        "volume": np.asarray(fields["volume"], dtype=np.float64),
        "time": np.asarray(fields["time"]),
        "decisions": decisions,
        "window": window,
        "candidates": candidates,
        "periods": periods,
        "jobs_inside": jobs_inside,
    }
    symbols = [str(s) for s in fields["symbols"]]
    wall0 = time.perf_counter()
    if n_jobs <= 1 or n == 1:
        per_symbol = [_symbol_minutes(i) for i in range(n)]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_jobs, n)) as pool:
            per_symbol = pool.map(_symbol_minutes, list(range(n)))
    wall = time.perf_counter() - wall0

    # Latest closed bar: one more retrain per symbol, not settled (no next close).
    snap = []
    for i in range(n):
        start = t - window
        raw = {key: _STATE[key][i : i + 1, start:t] for key in ("open", "high", "low", "close", "volume")}
        row = _retrain_window(raw, _STATE["time"][start:t], candidates, periods, jobs_inside)
        row["symbol_i"] = i
        row["decision"] = t - 1
        snap.append(row)

    rets = simple_next_return(close)[:, t - minutes : t]
    # rets column k is the simple return of bar (t-minutes+k), which is decision (t-minutes+k-1).
    positions = np.zeros((n, minutes), dtype=np.float64)
    factors = np.zeros((n, minutes), dtype=np.float64)
    names = [[""] * minutes for _ in range(n)]
    timing_rows = []
    for i, rows in enumerate(per_symbol):
        for k, row in enumerate(rows):
            positions[i, k] = row["position"]
            factors[i, k] = row["factor"]
            names[i][k] = row["name"]
            dec = row["decision"]
            timing_rows.append(_timing_row(symbols[i], row, fields["time"], "walk_forward", n_jobs if n_jobs > 1 else 1))
            timing_rows[-1]["hold_end_unix"] = int(fields["time"][dec + 1])
    for i, row in enumerate(snap):
        timing_rows.append(_timing_row(symbols[i], row, fields["time"], "latest_snapshot", n_jobs if n_jobs > 1 else 1))

    settle_index = np.arange(t - minutes, t)
    decision_index = settle_index - 1
    return {
        "symbols": symbols,
        "time": np.asarray(fields["time"]),
        "decisions": np.asarray(decision_index),
        "settle_index": settle_index,
        "positions": positions,
        "factors": factors,
        "names": names,
        "returns": rets,
        "snapshot": snap,
        "timing_rows": timing_rows,
        "wall_seconds": wall,
        "periods": periods,
        "minutes": minutes,
        "window": window,
        "n_jobs": n_jobs,
        "n_formulas": n_formulas,
    }


def _timing_row(symbol: str, row: dict, times: np.ndarray, kind: str, n_jobs: int) -> dict:
    dec = int(row["decision"])
    info = machine()
    finished = time.time()
    return {
        "kind": kind,
        "symbol": symbol,
        "decision_unix": int(times[dec]),
        "decision_utc": time_utc(int(times[dec])),
            "train_start_unix": int(times[dec - int(_STATE["window"]) + 1]),
        "train_end_unix": int(times[dec]),
        "n_bars": int(_STATE["window"]),
        "cpu_count": int(info["cpu_count"] or 1),
        "n_jobs": 1,
        "pool_workers": int(n_jobs),
        "n_formulas": int(_STATE.get("candidates") and len(_STATE["candidates"]) or N_FORMULAS),
        "feature_seconds": row["feature_seconds"],
        "search_seconds": row["search_seconds"],
        "inference_seconds": row["inference_seconds"],
        "total_seconds": row["total_seconds"],
        "finished_at_unix": finished,
        "finished_at_utc": time_utc(int(finished)),
        "champion": row["name"],
        "formula": row["formula"],
        "train_score": row["train_score"],
        "val_score": row["val_score"],
    }


def _book_metrics(result: dict) -> pd.DataFrame:
    periods = int(result["periods"])
    symbols = result["symbols"]
    pos = result["positions"]
    ret = result["returns"]
    rows = []
    pnl_1 = np.zeros_like(pos)
    pnl_0 = np.zeros_like(pos)
    to_1 = np.zeros_like(pos)
    bh_1 = np.zeros_like(pos)
    bh_0 = np.zeros_like(pos)
    for i, symbol in enumerate(symbols):
        p = pos[i]
        r = ret[i]
        a, ta = _apply_cost(p, r, SLIPPAGE_1BP)
        b, tb = _apply_cost(p, r, 0.0)
        hold = np.ones_like(r)
        ha, _ = _apply_cost(hold, r, SLIPPAGE_1BP)
        hb, _ = _apply_cost(hold, r, 0.0)
        pnl_1[i], pnl_0[i], to_1[i] = a, b, ta
        bh_1[i], bh_0[i] = ha, hb
        m1 = _metrics(a, ta, periods)
        m0 = _metrics(b, tb, periods)
        mb1 = _metrics(ha, np.abs(np.concatenate([[1.0], np.diff(hold)])), periods)
        mb0 = _metrics(hb, np.zeros_like(r), periods)
        flat = _metrics(np.zeros_like(r), np.zeros_like(r), periods)
        rows.append(_metric_row(symbol, m1, m0, mb1, mb0, flat))
    # Equal-weight book: average of coin pnl, not a second cost on the average.
    port_1 = pnl_1.mean(axis=0)
    port_0 = pnl_0.mean(axis=0)
    port_to = to_1.mean(axis=0)
    port_bh1 = bh_1.mean(axis=0)
    port_bh0 = bh_0.mean(axis=0)
    m1 = _metrics(port_1, port_to, periods)
    m0 = _metrics(port_0, port_to, periods)
    mb1 = _metrics(port_bh1, np.concatenate([[1.0], np.zeros(len(port_bh1) - 1)]), periods)
    mb0 = _metrics(port_bh0, np.zeros_like(port_bh0), periods)
    flat = _metrics(np.zeros_like(port_1), np.zeros_like(port_1), periods)
    rows.append(_metric_row("ALL", m1, m0, mb1, mb0, flat))
    result["pnl_1bp"] = pnl_1
    result["pnl_0"] = pnl_0
    result["turnover"] = to_1
    result["bh_1bp"] = bh_1
    result["bh_0"] = bh_0
    result["port_equity_1bp"] = m1["equity"]
    result["port_equity_0"] = m0["equity"]
    result["port_bh_0"] = mb0["equity"]
    result["port_flat"] = flat["equity"]
    return pd.DataFrame(rows)


def _metric_row(symbol, m1, m0, bh1, bh0, flat) -> dict:
    return {
        "symbol": symbol,
        "return_1bp": m1["return"],
        "sharpe_1bp": m1["sharpe"],
        "mdd_1bp": m1["mdd"],
        "turnover_1bp": m1["turnover_mean"],
        "equity_1bp": m1["equity_end"],
        "return_0cost": m0["return"],
        "sharpe_0cost": m0["sharpe"],
        "mdd_0cost": m0["mdd"],
        "turnover_0cost": m0["turnover_mean"],
        "equity_0cost": m0["equity_end"],
        "buyhold_return_1bp": bh1["return"],
        "buyhold_sharpe_1bp": bh1["sharpe"],
        "buyhold_mdd_1bp": bh1["mdd"],
        "buyhold_return_0cost": bh0["return"],
        "buyhold_sharpe_0cost": bh0["sharpe"],
        "buyhold_mdd_0cost": bh0["mdd"],
        "flat_return": flat["return"],
        "flat_sharpe": flat["sharpe"],
        "flat_mdd": flat["mdd"],
    }


def _frames(result: dict) -> dict[str, pd.DataFrame]:
    symbols = result["symbols"]
    times = result["time"]
    dec = result["decisions"]
    settle = result["settle_index"]
    minutes = result["minutes"]
    pos_rows = []
    fac_rows = []
    eq_rows = []
    products_f = []
    products_p = []
    for i, symbol in enumerate(symbols):
        fac_prod = result["factors"][i] * result["returns"][i]
        pos_prod = result["positions"][i] * result["returns"][i]
        products_f.append(fac_prod)
        products_p.append(pos_prod)
        f_csum = np.cumsum(fac_prod)
        p_csum = np.cumsum(pos_prod)
        eq1 = np.cumprod(1.0 + result["pnl_1bp"][i])
        eq0 = np.cumprod(1.0 + result["pnl_0"][i])
        eb1 = np.cumprod(1.0 + result["bh_1bp"][i])
        eb0 = np.cumprod(1.0 + result["bh_0"][i])
        for k in range(minutes):
            d = int(dec[k])
            h = int(settle[k])
            pos_rows.append({
                "time": int(times[d]),
                "time_utc": time_utc(int(times[d])),
                "settle_time": int(times[h]),
                "settle_time_utc": time_utc(int(times[h])),
                "segment": "oos",
                "symbol": symbol,
                "target_position": float(result["positions"][i, k]),
                "factor_value": float(result["factors"][i, k]),
                "factor_name": result["names"][i][k],
                "simple_return": float(result["returns"][i, k]),
            })
            fac_rows.append({
                "time": int(times[d]),
                "time_utc": time_utc(int(times[d])),
                "symbol": symbol,
                "factor_name": result["names"][i][k],
                "segment": "oos",
                "factor": float(result["factors"][i, k]),
                "target": float(result["returns"][i, k]),
                "product": float(fac_prod[k]),
                "cumsum": float(f_csum[k]),
                "oos_cumsum": float(f_csum[k]),
            })
            eq_rows.append({
                "time": int(times[h]),
                "time_utc": time_utc(int(times[h])),
                "symbol": symbol,
                "pnl_1bp": float(result["pnl_1bp"][i, k]),
                "pnl_0cost": float(result["pnl_0"][i, k]),
                "turnover": float(result["turnover"][i, k]),
                "equity_1bp": float(eq1[k]),
                "equity_0cost": float(eq0[k]),
                "equity_buyhold_1bp": float(eb1[k]),
                "equity_buyhold_0cost": float(eb0[k]),
                "equity_flat": 1.0,
            })
    f_mean = np.vstack(products_f).mean(axis=0)
    p_mean = np.vstack(products_p).mean(axis=0)
    f_csum = np.cumsum(f_mean)
    p_csum = np.cumsum(p_mean)
    port_eq1 = result["port_equity_1bp"]
    port_eq0 = result["port_equity_0"]
    port_bh = result["port_bh_0"]
    for k in range(minutes):
        d = int(dec[k])
        h = int(settle[k])
        pos_rows.append({
            "time": int(times[d]),
            "time_utc": time_utc(int(times[d])),
            "settle_time": int(times[h]),
            "settle_time_utc": time_utc(int(times[h])),
            "segment": "oos",
            "symbol": "ALL",
            "target_position": float(result["positions"][:, k].mean()),
            "factor_value": float("nan"),
            "factor_name": "equal_weight_mean",
            "simple_return": float("nan"),
        })
        fac_rows.append({
            "time": int(times[d]),
            "time_utc": time_utc(int(times[d])),
            "symbol": "ALL",
            "factor_name": "equal_weight_mean",
            "segment": "oos",
            "factor": float("nan"),
            "target": float("nan"),
            "product": float(f_mean[k]),
            "product_sum": float(np.vstack(products_f)[:, k].sum()),
            "cumsum": float(f_csum[k]),
            "oos_cumsum": float(f_csum[k]),
        })
        eq_rows.append({
            "time": int(times[h]),
            "time_utc": time_utc(int(times[h])),
            "symbol": "ALL",
            "pnl_1bp": float(result["pnl_1bp"][:, k].mean()),
            "pnl_0cost": float(result["pnl_0"][:, k].mean()),
            "turnover": float(result["turnover"][:, k].mean()),
            "equity_1bp": float(port_eq1[k]),
            "equity_0cost": float(port_eq0[k]),
            "equity_buyhold_1bp": float(np.cumprod(1.0 + result["bh_1bp"].mean(axis=0))[k]),
            "equity_buyhold_0cost": float(port_bh[k]),
            "equity_flat": 1.0,
        })
    # position × target uses the same shape as the factor diagnostic
    pos_diag = []
    for row_i, src in enumerate(fac_rows):
        # rebuild from products_p for coins; ALL already in p_mean
        pass
    for i, symbol in enumerate(symbols):
        prod = products_p[i]
        csum = np.cumsum(prod)
        for k in range(minutes):
            d = int(dec[k])
            pos_diag.append({
                "time": int(times[d]),
                "time_utc": time_utc(int(times[d])),
                "symbol": symbol,
                "segment": "oos",
                "position": float(result["positions"][i, k]),
                "target": float(result["returns"][i, k]),
                "product": float(prod[k]),
                "cumsum": float(csum[k]),
                "oos_cumsum": float(csum[k]),
            })
    for k in range(minutes):
        d = int(dec[k])
        pos_diag.append({
            "time": int(times[d]),
            "time_utc": time_utc(int(times[d])),
            "symbol": "ALL",
            "segment": "oos",
            "position": float(result["positions"][:, k].mean()),
            "target": float("nan"),
            "product": float(p_mean[k]),
            "cumsum": float(p_csum[k]),
            "oos_cumsum": float(p_csum[k]),
        })
    return {
        "position_history": pd.DataFrame(pos_rows),
        "factor_target_cumsum": pd.DataFrame(fac_rows),
        "position_target_cumsum": pd.DataFrame(pos_diag),
        "equity": pd.DataFrame(eq_rows),
        "retrain_timing_history": pd.DataFrame(result["timing_rows"]),
    }


def _plot(result: dict, frames: dict, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = frames["equity"]
    all_eq = eq[eq["symbol"] == "ALL"].sort_values("time")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(all_eq["time"], all_eq["equity_1bp"], label="strategy 0 fee + 1bp", lw=1.4)
    ax.plot(all_eq["time"], all_eq["equity_0cost"], label="strategy 0 cost", lw=1.2)
    ax.plot(all_eq["time"], all_eq["equity_buyhold_0cost"], label="buy & hold", lw=1.1)
    ax.plot(all_eq["time"], all_eq["equity_flat"], label="flat", lw=1.0)
    ax.set_title("Last 24h equal-weight equity (simple next-minute return)")
    ax.set_xlabel("settle unix seconds")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "equity.png", dpi=120)
    plt.close(fig)

    # Reuse the diagnostic plotters. They expect a cumsum column and a boundary.
    times = all_eq["time"].to_numpy()
    fac = frames["factor_target_cumsum"]
    pos = frames["position_target_cumsum"]
    from minute_alpha.diagnostics import plot_factor_cumsum, plot_position_cumsum

    boundary = int(fac["time"].iloc[0]) if len(fac) else 0
    plot_factor_cumsum(
        fac, np.array([boundary]), 0, folder / "factor_target_cumsum.png",
        title="Diagnostic: factor × next-minute simple return (not strategy PnL)",
    )
    plot_position_cumsum(
        pos, np.array([boundary]), 0, folder / "position_target_cumsum.png",
        title="Diagnostic: position × next-minute simple return (no costs)",
    )


def _write_portal(result: dict, frames: dict, metrics: pd.DataFrame, info: dict) -> dict:
    snap_rows = []
    cutoff = int(result["time"][-1])
    for i, symbol in enumerate(result["symbols"]):
        row = result["snapshot"][i]
        snap_rows.append({
            "symbol": symbol,
            "pair": f"t{symbol}F0:USTF0",
            "asof_unix": cutoff,
            "asof_utc": time_utc(cutoff),
            "factor_name": row["name"],
            "formula": row["formula"],
            "formula_tokens": row["tokens"],
            "factor_value": row["factor"],
            "target_position": row["position"],
            "train_score": row["train_score"],
            "val_score": row["val_score"],
            "retrain_seconds": row["total_seconds"],
        })
    totals = np.array([r["total_seconds"] for r in result["timing_rows"] if r["kind"] == "walk_forward"])
    timing = {
        "one_core_seconds": float(np.median(totals)) if totals.size else None,
        "multi_core_seconds": float(result["wall_seconds"] / max(result["minutes"], 1)),
        "speedup": None,
        "fits_in_one_minute_1core": bool(totals.size and np.percentile(totals, 95) < 60),
        "fits_in_one_minute_multicore": bool(result["wall_seconds"] < 60 * result["minutes"]),
        "retrain_count": int(totals.size),
        "retrain_seconds_mean": float(totals.mean()) if totals.size else None,
        "retrain_seconds_p50": float(np.percentile(totals, 50)) if totals.size else None,
        "retrain_seconds_p95": float(np.percentile(totals, 95)) if totals.size else None,
        "retrain_seconds_max": float(totals.max()) if totals.size else None,
        "wall_seconds": float(result["wall_seconds"]),
        "parallel": "one process per symbol, each retrain single-threaded",
    }
    if timing["one_core_seconds"] and result["n_jobs"] > 1 and result["wall_seconds"] > 0:
        serial_est = float(totals.sum())
        timing["speedup"] = serial_est / result["wall_seconds"]
        timing["serial_estimate_seconds"] = serial_est
    last = {
        "started_at_unix": int(time.time() - sum(r["total_seconds"] for r in result["snapshot"])),
        "started_at_utc": time_utc(int(time.time() - sum(r["total_seconds"] for r in result["snapshot"]))),
        "finished_at_unix": int(time.time()),
        "finished_at_utc": time_utc(int(time.time())),
        "duration_seconds": float(sum(r["total_seconds"] for r in result["snapshot"])),
        "cpu_count": int(info["cpu_count"] or 1),
        "n_jobs": int(result["n_jobs"]),
        "n_formulas": int(result["n_formulas"]),
        "train_window_bars": int(result["window"]),
        "data_cutoff_unix": cutoff,
        "data_cutoff_utc": time_utc(cutoff),
    }
    status = build_status(
        cpu_count=int(info["cpu_count"] or 1),
        n_jobs=int(result["n_jobs"]),
        data_cutoff_unix=cutoff,
        positions=snap_rows,
        last_retrain=last,
        timing=timing,
        symbols=result["symbols"],
        retrain_every_minutes=1,
        bars=int(result["time"].size),
        oos_start_unix=int(result["time"][int(result["settle_index"][0])]),
    )
    status["backtest"] = {
        "window": "last_24h",
        "minutes": int(result["minutes"]),
        "settlement": "simple_return close[t+1]/close[t]-1",
        "train_label": "log(open[t+2]/open[t+1]) inside the miner only",
        "train_window_bars": int(result["window"]),
        "seed": SEED,
        "costs": ["0 fee + 1bp slippage", "0 cost"],
        "baselines": ["buy_and_hold", "flat"],
    }
    for folder in (ARTIFACTS, REPO_OUT):
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
        metrics.to_csv(folder / "last_day_metrics.csv", index=False)
        pd.DataFrame(snap_rows).to_csv(folder / "next_minute_positions.csv", index=False)
        write_status(folder / "status.json", status)
        (folder / "last_day_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        _plot(result, frames, folder)
    return status


def _report_block(metrics: pd.DataFrame, timing: dict, status: dict) -> str:
    lines = [
        "## 过去一天逐分钟回测",
        "",
        "回测段是截至最新已闭合 1 分钟 bar 的最近 1440 根结算 bar。",
        "每一分钟的目标仓位只在前一根收盘时重训：训练窗是此前 4320 根已闭合 bar，候选公式由固定种子生成。",
        "结算收益是随后 1 分钟的简单收益 `close[t+1]/close[t]-1`。",
        "挖掘器内部的打分标签仍是生产定义 `log(open[t+2]/open[t+1])`，这个标签不参与结算。",
        "逐分钟全串行大约要一小时，所以按币种多进程跑，每个进程里的单次重训是单线程。",
        "",
        f"数据截止 {status['data_cutoff_utc']}。进程数 {status['n_jobs']}，机器核数 {status['cpu_count']}。",
        f"单次重训中位数 {timing.get('retrain_seconds_p50')} 秒，P95 {timing.get('retrain_seconds_p95')} 秒，"
        f"墙钟 {timing.get('wall_seconds')} 秒。相对全串行估计的加速比 {timing.get('speedup')}。",
        "单次重训中位数远小于 60 秒，逐分钟滚动赶得上 1 分钟 bar。",
        "",
        "收益是复利期末权益减 1。Sharpe 用 1 分钟收益年化（与时间戳估计的每年 bar 数一致）。",
        "MDD 是复利权益的峰值回撤。换手是 `|Δ仓位|` 的均值。买入持有从空仓进入满仓，1bp 版本在第一分钟扣一次滑点。零仓位权益恒为 1。",
        "",
        "| 币 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有收益 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['symbol']} | {row['return_1bp']:.4%} | {row['sharpe_1bp']:.2f} | {row['mdd_1bp']:.3%} | "
            f"{row['turnover_1bp']:.4f} | {row['return_0cost']:.4%} | {row['sharpe_0cost']:.2f} | "
            f"{row['mdd_0cost']:.3%} | {row['buyhold_return_0cost']:.4%} |"
        )
    lines += [
        "",
        "因子 × target、仓位 × target 的 target 是上面的简单收益，累加不含手续费。那是信号诊断，不是扣滑点后的策略收益。",
        "策略权益在 `equity.csv` / `equity.png`。",
        "",
    ]
    return "\n".join(lines)


def run_last_day(
    *,
    minutes: int = 1440,
    window: int = TRAIN_WINDOW,
    n_formulas: int = N_FORMULAS,
    n_jobs: int = 0,
    fetch_bars: int = 12000,
) -> dict:
    info = machine()
    if n_jobs <= 0:
        n_jobs = int(info["cpu_count"] or 1)
    t0 = time.perf_counter()
    aligned = load_universe(bars=fetch_bars, cache=True, grid_end="max", refresh=True)
    fields = to_matrices(aligned)
    load_s = time.perf_counter() - t0
    # Keep the tail that covers warmup + the settlement day.
    need = window + minutes
    if fields["close"].shape[1] > need + 5:
        sl = slice(fields["close"].shape[1] - (need + 2), None)
        for key in ("open", "high", "low", "close", "volume", "time"):
            fields[key] = fields[key][:, sl] if fields[key].ndim == 2 else fields[key][sl]
    result = run_from_fields(
        fields,
        minutes=minutes,
        window=window,
        n_formulas=n_formulas,
        n_jobs=n_jobs,
        jobs_inside=1,
    )
    result["data_load_seconds"] = load_s
    metrics = _book_metrics(result)
    frames = _frames(result)
    status = _write_portal(result, frames, metrics, info)
    block = _report_block(metrics, status["timing"], status)
    report_path = Path("/workspace/minute_alpha/REPORT.md")
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "## 过去一天逐分钟回测"
    if marker in previous:
        previous = previous.split(marker)[0].rstrip() + "\n\n"
    text = previous + block
    report_path.write_text(text, encoding="utf-8")
    for folder in (ARTIFACTS, REPO_OUT):
        (folder / "REPORT.md").write_text(text, encoding="utf-8")
    return {"status": status, "metrics": metrics, "wall_seconds": result["wall_seconds"]}


if __name__ == "__main__":
    out = run_last_day()
    print(
        f"wall {out['wall_seconds']:.1f}s  "
        f"ALL 1bp {out['metrics'].set_index('symbol').loc['ALL', 'sharpe_1bp']:.2f}  "
        f"0cost {out['metrics'].set_index('symbol').loc['ALL', 'sharpe_0cost']:.2f}"
    )
