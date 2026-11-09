"""Past-week minute retrain. Gaps stay missing; they are not filled with 0.

Same seed, symbols, miner label, tanh map, and settlement as the 24h run:
position known at the close of a real bar t is multiplied by
close[t+60] / close[t] - 1. Each decision uses the previous 4320 real prints
of that symbol only. Output goes to minute_alpha/portal_out_7d/.
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

from minute_alpha.backtest import annualization
from minute_alpha.config import N_FORMULAS, PAIRS, SEED, TRAIN_WINDOW
from minute_alpha.data import load_raw_frames
from minute_alpha.diagnostics import time_utc
from minute_alpha.last_day import _apply_cost, _metrics, _retrain_window
from minute_alpha.timing import machine

OUT_DIR = Path("/workspace/minute_alpha/portal_out_7d")
ARTIFACTS = Path("/opt/cursor/artifacts/minute_alpha_7d")
MINUTES = 7 * 24 * 60
HISTORY = MINUTES + TRAIN_WINDOW

_STATE: dict = {}


def latest_closed_open(now: int | None = None) -> int:
    """Open time of the newest 1-minute bar that has already closed."""
    now = int(time.time()) if now is None else int(now)
    return (now // 60) * 60 - 60


def bars_per_year(bar_seconds: int) -> int:
    """Calendar bars per year for a regular grid. 1m=525960, 5m=105192, 15m=35064."""
    return int(round(365.25 * 86400 / bar_seconds))


def week_clock(latest_open: int, history: int = HISTORY, bar_seconds: int = 60) -> np.ndarray:
    start = latest_open - (history - 1) * bar_seconds
    return np.arange(start, latest_open + bar_seconds, bar_seconds, dtype=np.int64)


def _gap_row(symbol: str, decision: int, reason: str) -> dict:
    return {
        "gap": True,
        "reason": reason,
        "symbol": symbol,
        "decision": int(decision),
        "position": np.nan,
        "factor": np.nan,
        "name": "",
        "simple_return": np.nan,
        "feature_seconds": 0.0,
        "search_seconds": 0.0,
        "inference_seconds": 0.0,
        "total_seconds": 0.0,
        "train_score": np.nan,
        "val_score": np.nan,
    }


def _symbol_week(symbol_i: int) -> list[dict]:
    state = _STATE
    pack = state["packs"][symbol_i]
    symbol = state["symbols"][symbol_i]
    times = pack["time"]
    close = pack["close"]
    index = {int(t): i for i, t in enumerate(times)}
    window = state["window"]
    rows = []
    for step, decision in enumerate(state["decisions"]):
        if step % 500 == 0:
            print(f"{symbol} bar {step}/{len(state['decisions'])}", flush=True)
        decision = int(decision)
        settle = decision + int(state["bar_seconds"])
        i_now = index.get(decision)
        i_next = index.get(settle)
        if i_now is None or i_next is None:
            reason = "missing_print" if i_now is None else "missing_next_close"
            rows.append(_gap_row(symbol, decision, reason))
            continue
        if i_now + 1 < window:
            rows.append(_gap_row(symbol, decision, "insufficient_history"))
            continue
        sl = slice(i_now + 1 - window, i_now + 1)
        raw = {key: pack[key][sl][None, :] for key in ("open", "high", "low", "close", "volume")}
        rec = _retrain_window(raw, times[sl], state["candidates"], state["periods"], 1)
        prev = close[i_now]
        nxt = close[i_next]
        rec["simple_return"] = float(nxt / prev - 1.0) if prev != 0 else np.nan
        rec["gap"] = False
        rec["reason"] = ""
        rec["symbol"] = symbol
        rec["decision"] = decision
        rows.append(rec)
    return rows


def _valid_pnl(position: np.ndarray, ret: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Cost along the valid subsequence. Gaps stay NaN and do not reset the position to 0."""
    pnl = np.full(position.shape, np.nan)
    turnover = np.full(position.shape, np.nan)
    valid = np.isfinite(position) & np.isfinite(ret)
    if not valid.any():
        return pnl, turnover
    sub_pos = position[valid]
    sub_ret = ret[valid]
    sub_pnl, sub_to = _apply_cost(sub_pos, sub_ret, cost)
    pnl[valid] = sub_pnl
    turnover[valid] = sub_to
    return pnl, turnover


def run_from_packs(
    packs: dict[str, dict[str, np.ndarray]],
    *,
    latest_open: int,
    minutes: int = MINUTES,
    window: int = TRAIN_WINDOW,
    n_formulas: int = N_FORMULAS,
    n_jobs: int = 1,
    bar_seconds: int = 60,
    periods: int | None = None,
    carry_position: bool = False,
) -> dict:
    from minute_alpha.search import make_candidates

    clock = week_clock(latest_open, history=minutes + window, bar_seconds=bar_seconds)
    if len(clock) != minutes + window:
        raise RuntimeError(f"clock length {len(clock)} != {minutes + window}")
    decisions = clock[window - 1 : -1]
    if len(decisions) != minutes:
        raise RuntimeError(f"expected {minutes} decisions, got {len(decisions)}")
    if periods is None:
        periods = annualization(np.arange(window, dtype=np.int64) * bar_seconds)
    global _STATE
    symbols = list(packs)
    _STATE = {
        "packs": [packs[s] for s in symbols],
        "symbols": symbols,
        "decisions": decisions,
        "window": window,
        "bar_seconds": int(bar_seconds),
        "candidates": make_candidates(n_formulas, SEED),
        "periods": int(periods),
    }
    wall0 = time.perf_counter()
    if n_jobs <= 1 or len(symbols) == 1:
        per_symbol = [_symbol_week(i) for i in range(len(symbols))]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_jobs, len(symbols))) as pool:
            per_symbol = pool.map(_symbol_week, list(range(len(symbols))))
    wall = time.perf_counter() - wall0
    return {
        "symbols": symbols,
        "decisions": decisions,
        "per_symbol": per_symbol,
        "wall_seconds": wall,
        "periods": int(_STATE["periods"]),
        "minutes": minutes,
        "window": window,
        "n_jobs": n_jobs,
        "n_formulas": n_formulas,
        "latest_open": int(latest_open),
        "bar_seconds": int(bar_seconds),
        "carry_position": bool(carry_position),
    }


def _tables(result: dict) -> dict[str, pd.DataFrame]:
    symbols = result["symbols"]
    decisions = result["decisions"]
    periods = result["periods"]
    n = len(symbols)
    m = len(decisions)
    position = np.full((n, m), np.nan)
    factor = np.full((n, m), np.nan)
    ret = np.full((n, m), np.nan)
    names = [[""] * m for _ in range(n)]
    timing_rows = []
    gap_rows = []
    for i, rows in enumerate(result["per_symbol"]):
        for k, row in enumerate(rows):
            position[i, k] = row["position"]
            factor[i, k] = row["factor"]
            ret[i, k] = row["simple_return"]
            names[i][k] = row.get("name") or ""
            if row["gap"]:
                gap_rows.append({
                    "time": int(decisions[k]),
                    "time_utc": time_utc(int(decisions[k])),
                    "symbol": symbols[i],
                    "reason": row["reason"],
                })
            elif row["total_seconds"] > 0:
                timing_rows.append({
                    "symbol": symbols[i],
                    "decision_unix": int(decisions[k]),
                    "decision_utc": time_utc(int(decisions[k])),
                    "total_seconds": row["total_seconds"],
                    "feature_seconds": row["feature_seconds"],
                    "search_seconds": row["search_seconds"],
                    "inference_seconds": row["inference_seconds"],
                    "champion": row["name"],
                })
    if result.get("carry_position"):
        # A missing bar keeps the last target. It is not flattened to 0.
        for i in range(n):
            last = np.nan
            for k in range(m):
                if np.isfinite(position[i, k]):
                    last = position[i, k]
                elif np.isfinite(last):
                    position[i, k] = last
    pnl_1 = np.full((n, m), np.nan)
    pnl_0 = np.full((n, m), np.nan)
    turn = np.full((n, m), np.nan)
    bh_1 = np.full((n, m), np.nan)
    bh_0 = np.full((n, m), np.nan)
    metric_rows = []
    for i, symbol in enumerate(symbols):
        hold = np.ones(m)
        hold[~np.isfinite(ret[i])] = np.nan
        pnl_1[i], turn[i] = _valid_pnl(position[i], ret[i], 0.0001)
        pnl_0[i], _ = _valid_pnl(position[i], ret[i], 0.0)
        bh_1[i], _ = _valid_pnl(hold, ret[i], 0.0001)
        bh_0[i], _ = _valid_pnl(hold, ret[i], 0.0)
        valid = np.isfinite(pnl_1[i])
        metric_rows.append(_metric_line(
            symbol,
            pnl_1[i, valid], turn[i, valid],
            pnl_0[i, valid],
            bh_1[i, np.isfinite(bh_1[i])],
            bh_0[i, np.isfinite(bh_0[i])],
            periods,
            gap_count=int((~np.isfinite(ret[i])).sum()),
            valid_count=int(valid.sum()),
        ))
    port_1 = np.nanmean(pnl_1, axis=0)
    port_0 = np.nanmean(pnl_0, axis=0)
    port_to = np.nanmean(turn, axis=0)
    port_bh1 = np.nanmean(bh_1, axis=0)
    port_bh0 = np.nanmean(bh_0, axis=0)
    # A minute with no symbol is a gap, not a zero return.
    port_valid = np.isfinite(port_1)
    metric_rows.append(_metric_line(
        "ALL",
        port_1[port_valid], port_to[port_valid],
        port_0[port_valid],
        port_bh1[np.isfinite(port_bh1)],
        port_bh0[np.isfinite(port_bh0)],
        periods,
        gap_count=int((~port_valid).sum()),
        valid_count=int(port_valid.sum()),
    ))
    fac_rows = []
    pos_rows = []
    fac_csum = np.zeros(n)
    pos_csum = np.zeros(n)
    fac_seen = np.zeros(n, dtype=bool)
    pos_seen = np.zeros(n, dtype=bool)
    all_fac_csum = 0.0
    all_pos_csum = 0.0
    all_fac_seen = False
    all_pos_seen = False
    for k, decision in enumerate(decisions):
        products_f = []
        products_p = []
        for i, symbol in enumerate(symbols):
            gap = not (np.isfinite(factor[i, k]) and np.isfinite(ret[i, k]))
            if gap:
                product_f = np.nan
                product_p = np.nan
                f_show = np.nan if not fac_seen[i] else fac_csum[i]
                p_show = np.nan if not pos_seen[i] else pos_csum[i]
            else:
                product_f = float(factor[i, k] * ret[i, k])
                product_p = float(position[i, k] * ret[i, k])
                fac_csum[i] += product_f
                pos_csum[i] += product_p
                fac_seen[i] = True
                pos_seen[i] = True
                f_show = fac_csum[i]
                p_show = pos_csum[i]
                products_f.append(product_f)
                products_p.append(product_p)
            stamp = int(decision)
            fac_rows.append({
                "time": stamp,
                "time_utc": time_utc(stamp),
                "symbol": symbol,
                "factor_name": names[i][k],
                "segment": "oos",
                "factor": factor[i, k],
                "target": ret[i, k],
                "product": product_f,
                "cumsum": f_show,
                "oos_cumsum": f_show,
            })
            pos_rows.append({
                "time": stamp,
                "time_utc": time_utc(stamp),
                "symbol": symbol,
                "segment": "oos",
                "position": position[i, k],
                "target": ret[i, k],
                "product": product_p,
                "cumsum": p_show,
                "oos_cumsum": p_show,
            })
        if products_f:
            mean_f = float(np.mean(products_f))
            all_fac_csum += mean_f
            all_fac_seen = True
            show_f = all_fac_csum
        else:
            mean_f = np.nan
            show_f = all_fac_csum if all_fac_seen else np.nan
        if products_p:
            mean_p = float(np.mean(products_p))
            all_pos_csum += mean_p
            all_pos_seen = True
            show_p = all_pos_csum
        else:
            mean_p = np.nan
            show_p = all_pos_csum if all_pos_seen else np.nan
        stamp = int(decision)
        fac_rows.append({
            "time": stamp,
            "time_utc": time_utc(stamp),
            "symbol": "ALL",
            "factor_name": "equal_weight_mean",
            "segment": "oos",
            "factor": np.nan,
            "target": np.nan,
            "product": mean_f,
            "product_sum": float(np.sum(products_f)) if products_f else np.nan,
            "cumsum": show_f,
            "oos_cumsum": show_f,
        })
        pos_rows.append({
            "time": stamp,
            "time_utc": time_utc(stamp),
            "symbol": "ALL",
            "segment": "oos",
            "position": float(np.nanmean(position[:, k])),
            "target": np.nan,
            "product": mean_p,
            "cumsum": show_p,
            "oos_cumsum": show_p,
        })
    metrics = pd.DataFrame(metric_rows)
    daily = _daily(decisions, port_1=port_1, port_0=port_0, port_bh=port_bh0)
    totals = np.array([r["total_seconds"] for r in timing_rows], dtype=np.float64)
    timing = {
        "retrain_count": int(totals.size),
        "retrain_seconds_median": float(np.median(totals)) if totals.size else None,
        "retrain_seconds_p95": float(np.percentile(totals, 95)) if totals.size else None,
        "retrain_seconds_mean": float(totals.mean()) if totals.size else None,
        "wall_seconds": float(result_wall := result.get("wall_seconds", 0.0) if False else 0.0),
        "serial_estimate_seconds": float(totals.sum()) if totals.size else 0.0,
        "n_jobs": int(result["n_jobs"]),
        "gap_rows": int(len(gap_rows)),
    }
    # wall is filled by the caller; keep the key for the writer.
    return {
        "metrics": metrics,
        "factor_target_cumsum": pd.DataFrame(fac_rows),
        "position_target_cumsum": pd.DataFrame(pos_rows),
        "gaps": pd.DataFrame(gap_rows),
        "timing_rows": pd.DataFrame(timing_rows),
        "daily": daily,
        "timing": timing,
        "port_1": port_1,
        "port_0": port_0,
        "port_bh0": port_bh0,
        "decisions": decisions,
    }


def _metric_line(symbol, pnl_1, to_1, pnl_0, bh_1, bh_0, periods, gap_count, valid_count) -> dict:
    m1 = _metrics(pnl_1, to_1, periods)
    m0 = _metrics(pnl_0, to_1, periods)
    b1 = _metrics(bh_1, np.zeros(len(bh_1)), periods)
    b0 = _metrics(bh_0, np.zeros(len(bh_0)), periods)
    flat = _metrics(np.zeros(valid_count), np.zeros(valid_count), periods)
    return {
        "symbol": symbol,
        "return_1bp": m1["return"],
        "sharpe_1bp": m1["sharpe"],
        "mdd_1bp": m1["mdd"],
        "turnover_1bp": m1["turnover_mean"],
        "return_0cost": m0["return"],
        "sharpe_0cost": m0["sharpe"],
        "mdd_0cost": m0["mdd"],
        "turnover_0cost": m0["turnover_mean"],
        "buyhold_return_1bp": b1["return"],
        "buyhold_sharpe_1bp": b1["sharpe"],
        "buyhold_mdd_1bp": b1["mdd"],
        "buyhold_return_0cost": b0["return"],
        "buyhold_sharpe_0cost": b0["sharpe"],
        "buyhold_mdd_0cost": b0["mdd"],
        "flat_return": flat["return"],
        "valid_minutes": valid_count,
        "gap_minutes": gap_count,
    }


def _daily(decisions, port_1, port_0, port_bh) -> pd.DataFrame:
    frame = pd.DataFrame({
        "time": decisions.astype(np.int64),
        "pnl_1bp": port_1,
        "pnl_0cost": port_0,
        "pnl_buyhold_0cost": port_bh,
    })
    frame = frame[np.isfinite(frame["pnl_1bp"])].copy()
    if frame.empty:
        return frame
    stamps = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["date"] = stamps.dt.strftime("%Y-%m-%d")
    rows = []
    for date, sub in frame.groupby("date", sort=True):
        rows.append({
            "date": date,
            "return_1bp": float(np.prod(1.0 + sub["pnl_1bp"]) - 1.0),
            "return_0cost": float(np.prod(1.0 + sub["pnl_0cost"]) - 1.0),
            "buyhold_return_0cost": float(np.prod(1.0 + sub["pnl_buyhold_0cost"]) - 1.0),
            "minutes": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _plot(tables: dict, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from minute_alpha.diagnostics import plot_factor_cumsum, plot_position_cumsum

    decisions = tables["decisions"]
    valid = np.isfinite(tables["port_1"])
    if valid.any():
        eq1 = np.cumprod(1.0 + tables["port_1"][valid])
        eq0 = np.cumprod(1.0 + tables["port_0"][valid])
        ebh = np.cumprod(1.0 + tables["port_bh0"][valid])
        times = decisions[valid]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(times, eq1, label="strategy 0 fee + 1bp", lw=1.3)
        ax.plot(times, eq0, label="strategy 0 cost", lw=1.2)
        ax.plot(times, ebh, label="buy & hold", lw=1.1)
        ax.plot(times, np.ones_like(eq1), label="flat", lw=1.0)
        ax.set_title("Past week equal-weight equity (gaps omitted, not filled with 0)")
        ax.set_xlabel("decision unix seconds")
        ax.set_ylabel("equity (start = 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / "equity.png", dpi=120)
        plt.close(fig)
    boundary = int(decisions[0])
    plot_factor_cumsum(
        tables["factor_target_cumsum"], np.array([boundary]), 0,
        folder / "factor_target_cumsum.png",
        title="7d factor × next-minute simple return (gaps are blank, not 0)",
    )
    plot_position_cumsum(
        tables["position_target_cumsum"], np.array([boundary]), 0,
        folder / "position_target_cumsum.png",
        title="7d position × next-minute simple return (no costs; gaps blank)",
    )


def _write(result: dict, tables: dict, folders: tuple[Path, ...] | None = None) -> None:
    tables["timing"]["wall_seconds"] = float(result["wall_seconds"])
    if tables["timing"]["serial_estimate_seconds"] and result["wall_seconds"] > 0:
        tables["timing"]["speedup"] = tables["timing"]["serial_estimate_seconds"] / result["wall_seconds"]
    totals = tables["timing_rows"]["total_seconds"].to_numpy() if len(tables["timing_rows"]) else np.array([])
    if totals.size:
        tables["timing"]["retrain_seconds_median"] = float(np.median(totals))
        tables["timing"]["retrain_seconds_p95"] = float(np.percentile(totals, 95))
    report = _report(tables)
    for folder in (folders or (OUT_DIR, ARTIFACTS)):
        folder.mkdir(parents=True, exist_ok=True)
        tables["factor_target_cumsum"].to_csv(folder / "factor_target_cumsum.csv", index=False)
        tables["position_target_cumsum"].to_csv(folder / "position_target_cumsum.csv", index=False)
        tables["metrics"].to_csv(folder / "week_metrics.csv", index=False)
        tables["daily"].to_csv(folder / "daily_portfolio_returns.csv", index=False)
        tables["gaps"].to_csv(folder / "gaps.csv", index=False)
        tables["timing_rows"].to_csv(folder / "retrain_timing_history.csv", index=False)
        (folder / "week_timing.json").write_text(json.dumps(tables["timing"], indent=2), encoding="utf-8")
        (folder / "REPORT.md").write_text(report, encoding="utf-8")
        _plot(tables, folder)


def _report(tables: dict) -> str:
    timing = tables["timing"]
    lines = [
        "# 过去一周逐分钟回测",
        "",
        "结算段是截至最新已闭合 1 分钟 bar 的最近 10080 个时钟分钟。",
        "每个测试分钟只用该币此前 4320 根真实已闭合 candle 重训，种子与 24 小时版相同。",
        "结算仍是 `close[t+60]/close[t]-1`。没有成交的分钟记为缺口，收益留空，不用 0 填充。",
        "等权组合只平均当分钟有真实结算的币。24 小时结果仍在 `minute_alpha/portal_out/`，没有被覆盖。",
        "",
        f"重训次数 {timing.get('retrain_count')}，中位数 {timing.get('retrain_seconds_median')} 秒，"
        f"P95 {timing.get('retrain_seconds_p95')} 秒。",
        f"全串行估计 {timing.get('serial_estimate_seconds')} 秒，{timing.get('n_jobs')} 进程墙钟 "
        f"{timing.get('wall_seconds')} 秒，加速比 {timing.get('speedup')}。",
        f"缺口行数 {timing.get('gap_rows')}（见 gaps.csv）。",
        "",
        "| 币 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有 | 有效分钟 | 缺口分钟 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in tables["metrics"].iterrows():
        lines.append(
            f"| {row['symbol']} | {row['return_1bp']:.4%} | {row['sharpe_1bp']:.2f} | {row['mdd_1bp']:.3%} | "
            f"{row['turnover_1bp']:.4f} | {row['return_0cost']:.4%} | {row['sharpe_0cost']:.2f} | "
            f"{row['mdd_0cost']:.3%} | {row['buyhold_return_0cost']:.4%} | "
            f"{int(row['valid_minutes'])} | {int(row['gap_minutes'])} |"
        )
    lines += ["", "## 按天组合收益", "", "| 日期 | 1bp | 0 成本 | 买入持有 | 分钟数 |", "| --- | --- | --- | --- | --- |"]
    for _, row in tables["daily"].iterrows():
        lines.append(
            f"| {row['date']} | {row['return_1bp']:.4%} | {row['return_0cost']:.4%} | "
            f"{row['buyhold_return_0cost']:.4%} | {int(row['minutes'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _packs_from_frames(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    packs = {}
    for symbol, frame in frames.items():
        packs[symbol] = {
            key: frame[key].to_numpy(dtype=np.float64) if key != "time" else frame[key].to_numpy(dtype=np.int64)
            for key in ("time", "open", "high", "low", "close", "volume")
        }
    return packs


def run_last_week(n_jobs: int = 0, fetch_bars: int = 20000) -> dict:
    info = machine()
    if n_jobs <= 0:
        n_jobs = int(info["cpu_count"] or 1)
    latest_wall = latest_closed_open()
    cached = load_raw_frames(bars=fetch_bars, cache=True, refresh=False)
    newest = max(int(frame["time"].iloc[-1]) for frame in cached.values()) if cached else 0
    # Refresh only when the cache does not already reach the newest closed minute.
    frames = cached if newest >= latest_wall - 60 else load_raw_frames(bars=fetch_bars, cache=True, refresh=True)
    newest = max(int(frame["time"].iloc[-1]) for frame in frames.values())
    # End on a real print. A clock minute nobody printed is a gap, not a bar of zeros,
    # and it should not push the window past the data we actually downloaded.
    latest = min(latest_wall, newest)
    # Do not invent minutes after a symbol's last real print; the clock still
    # ends at the newest closed bar in the universe, and later minutes are gaps.
    packs = _packs_from_frames(frames)
    have = {s: (int(p["time"][0]), int(p["time"][-1]), int(p["time"].size)) for s, p in packs.items()}
    print("coverage", have, "latest", latest, flush=True)
    result = run_from_packs(packs, latest_open=latest, n_jobs=n_jobs)
    tables = _tables(result)
    tables["timing"]["coverage"] = {
        s: {"first": a, "last": b, "prints": c} for s, (a, b, c) in have.items()
    }
    _write(result, tables)
    # Keep the 24h report intact; add a pointer only if it is not already there.
    pointer = "一周结果在 `minute_alpha/portal_out_7d/`，不覆盖 24 小时目录。"
    report_path = Path("/workspace/minute_alpha/REPORT.md")
    if report_path.exists() and pointer not in report_path.read_text(encoding="utf-8"):
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + pointer + "\n")
    return {"timing": tables["timing"], "metrics": tables["metrics"]}


if __name__ == "__main__":
    # PAIRS is the production symbol order; referenced so a missing import fails loudly.
    assert set(PAIRS) >= {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC"}
    out = run_last_week()
    print(
        f"wall {out['timing'].get('wall_seconds')} "
        f"median {out['timing'].get('retrain_seconds_median')} "
        f"p95 {out['timing'].get('retrain_seconds_p95')}"
    )
