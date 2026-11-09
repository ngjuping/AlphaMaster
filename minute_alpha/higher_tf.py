"""5m and 15m variants of the same week used by portal_out_7d.

Candles come from Bitfinex's public candle API (``trade:5m`` and ``trade:15m``),
not from a local resample of 1m prints. The exchange only emits a bar when
that interval had trades. A missing bar keeps the previous target position
and leaves the return blank. It is not filled with 0.

The settlement window ends at the last bar of that timeframe that had closed
by 2026-11-09 04:25 UTC, the close of the 1m bar that opened at 04:24 UTC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from minute_alpha.config import N_FORMULAS, SEED
from minute_alpha.data import load_raw_frames
from minute_alpha.last_week import (
    _packs_from_frames,
    _tables,
    _write,
    bars_per_year,
    run_from_packs,
)
from minute_alpha.timing import machine

CUTOFF = 1794198240  # 2026-11-09 04:24:00 UTC, same 1m bar as portal_out_7d
KNOWN_THROUGH = CUTOFF + 60


def last_closed_open(bar_seconds: int, cutoff: int = CUTOFF) -> int:
    open_ = cutoff - (cutoff % bar_seconds)
    if open_ + bar_seconds > KNOWN_THROUGH:
        open_ -= bar_seconds
    return int(open_)


def render_report(tables: dict, title: str) -> str:
    timing = tables["timing"]
    lines = [
        f"# {title}",
        "",
        timing.get("note", ""),
        "",
        f"年化系数 {timing.get('periods_per_year')}（365.25 天 × 每天的 bar 数，Sharpe = 均值/标准差 × sqrt(年化系数)）。",
        f"重训 {timing.get('retrain_count')} 次，中位数 {timing.get('retrain_seconds_median')} 秒，"
        f"P95 {timing.get('retrain_seconds_p95')} 秒，墙钟 {timing.get('wall_seconds')} 秒，"
        f"{timing.get('n_jobs')} 进程，加速比 {timing.get('speedup')}。",
        f"缺口行 {timing.get('gap_rows')}。没有成交的 bar 沿用上一笔目标仓位，收益留空，不写 0。",
        "",
        "| 币 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有 | 有效 bar | 缺口 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in tables["metrics"].iterrows():
        lines.append(
            f"| {row['symbol']} | {row['return_1bp']:.4%} | {row['sharpe_1bp']:.2f} | {row['mdd_1bp']:.3%} | "
            f"{row['turnover_1bp']:.4f} | {row['return_0cost']:.4%} | {row['sharpe_0cost']:.2f} | "
            f"{row['mdd_0cost']:.3%} | {row['buyhold_return_0cost']:.4%} | "
            f"{int(row['valid_minutes'])} | {int(row['gap_minutes'])} |"
        )
    lines += ["", "## 按天组合收益", "", "| 日期 | 1bp | 0 成本 | 买入持有 | bar 数 |", "| --- | --- | --- | --- | --- |"]
    for _, row in tables["daily"].iterrows():
        lines.append(
            f"| {row['date']} | {row['return_1bp']:.4%} | {row['return_0cost']:.4%} | "
            f"{row['buyhold_return_0cost']:.4%} | {int(row['minutes'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_variant(
    timeframe: str,
    window: int,
    out_dir: Path,
    artifact_dir: Path,
    *,
    n_jobs: int = 0,
    fetch_bars: int = 8000,
    refresh: bool = True,
) -> dict:
    step = {"5m": 300, "15m": 900}[timeframe]
    info = machine()
    if n_jobs <= 0:
        n_jobs = int(info["cpu_count"] or 1)
    frames = load_raw_frames(bars=fetch_bars, cache=True, refresh=refresh, timeframe=timeframe)
    latest = last_closed_open(step)
    newest = max(int(frame["time"].iloc[-1]) for frame in frames.values())
    latest = min(latest, newest - (newest % step))
    minutes = 7 * 24 * 3600 // step
    packs = _packs_from_frames(frames)
    print(f"{timeframe} window {window} latest {latest} decisions {minutes}", flush=True)
    result = run_from_packs(
        packs,
        latest_open=latest,
        minutes=minutes,
        window=window,
        n_formulas=N_FORMULAS,
        n_jobs=n_jobs,
        bar_seconds=step,
        periods=bars_per_year(step),
        carry_position=True,
    )
    tables = _tables(result)
    tables["timing"]["wall_seconds"] = float(result["wall_seconds"])
    tables["timing"]["periods_per_year"] = int(result["periods"])
    tables["timing"]["bar_seconds"] = step
    tables["timing"]["window_bars"] = window
    tables["timing"]["timeframe"] = timeframe
    tables["timing"]["seed"] = SEED
    tables["timing"]["latest_open"] = int(latest)
    if tables["timing"]["serial_estimate_seconds"] and result["wall_seconds"] > 0:
        tables["timing"]["speedup"] = tables["timing"]["serial_estimate_seconds"] / result["wall_seconds"]
    note = (
        f"Bitfinex 公开 `trade:{timeframe}` candle，不是由本地 1m 重采样。"
        f"训练窗 {window} 根（{window * step // 60} 分钟）。"
        f"结算窗截至 {latest}（unix），共 {minutes} 根。"
        "种子 20261109，32 条公式，仓位 tanh。"
    )
    tables["timing"]["note"] = note
    _write(result, tables, folders=(out_dir, artifact_dir))
    text = render_report(tables, f"{timeframe} 窗 {window} 根")
    for folder in (out_dir, artifact_dir):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "REPORT.md").write_text(text, encoding="utf-8")
        (folder / "week_timing.json").write_text(json.dumps(tables["timing"], indent=2), encoding="utf-8")
    return {"timing": tables["timing"], "metrics": tables["metrics"], "daily": tables["daily"]}


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        metrics = row["metrics"].set_index("symbol").loc["ALL"]
        timing = row["timing"]
        out.append({
            "variant": row["name"],
            "periods_per_year": timing.get("periods_per_year"),
            "return_1bp": metrics["return_1bp"],
            "sharpe_1bp": metrics["sharpe_1bp"],
            "mdd_1bp": metrics["mdd_1bp"],
            "turnover_1bp": metrics["turnover_1bp"],
            "return_0cost": metrics["return_0cost"],
            "sharpe_0cost": metrics["sharpe_0cost"],
            "mdd_0cost": metrics["mdd_0cost"],
            "buyhold_return_0cost": metrics["buyhold_return_0cost"],
            "retrain_median_s": timing.get("retrain_seconds_median"),
            "retrain_p95_s": timing.get("retrain_seconds_p95"),
            "wall_seconds": timing.get("wall_seconds"),
            "valid_bars": metrics["valid_minutes"],
            "gap_bars": metrics["gap_minutes"],
        })
    return pd.DataFrame(out)


def _load_1m_all() -> dict:
    path = Path("/workspace/minute_alpha/portal_out_7d/week_metrics.csv")
    metrics = pd.read_csv(path).set_index("symbol").loc["ALL"]
    timing = json.loads(Path("/workspace/minute_alpha/portal_out_7d/week_timing.json").read_text())
    # The 1m run annualized with annualization() on a 4320-bar 60s grid.
    from minute_alpha.backtest import annualization
    periods = annualization(np.arange(4320, dtype=np.int64) * 60)
    return {
        "name": "1m / 4320 bars (3d)",
        "metrics": pd.read_csv(path),
        "timing": {
            **timing,
            "periods_per_year": periods,
            "retrain_seconds_median": timing.get("retrain_seconds_median"),
            "retrain_seconds_p95": timing.get("retrain_seconds_p95"),
        },
    }


def write_comparison(variants: list[dict]) -> None:
    frame = comparison_table([_load_1m_all(), *variants])
    text = ["# 1m / 5m / 15m 同一周对比", "", "等权组合。Sharpe 用各周期自己的年化系数。1m 系数来自 4320 根 60 秒网格的 `annualization()`。", ""]
    text.append("| 变体 | 年化系数 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有 | 重训中位秒 | P95 | 墙钟秒 |")
    text.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in frame.iterrows():
        text.append(
            f"| {row['variant']} | {int(row['periods_per_year'])} | {row['return_1bp']:.4%} | {row['sharpe_1bp']:.2f} | "
            f"{row['mdd_1bp']:.3%} | {row['turnover_1bp']:.4f} | {row['return_0cost']:.4%} | {row['sharpe_0cost']:.2f} | "
            f"{row['mdd_0cost']:.3%} | {row['buyhold_return_0cost']:.4%} | {row['retrain_median_s']:.3f} | "
            f"{row['retrain_p95_s']:.3f} | {row['wall_seconds']:.0f} |"
        )
    body = "\n".join(text) + "\n"
    for folder in (
        Path("/workspace/minute_alpha/portal_out_7d_5m"),
        Path("/workspace/minute_alpha/portal_out_7d_15m"),
        Path("/opt/cursor/artifacts/minute_alpha_5m"),
        Path("/opt/cursor/artifacts/minute_alpha_15m"),
    ):
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "timeframe_comparison.csv", index=False)
        (folder / "COMPARISON.md").write_text(body, encoding="utf-8")
    report = Path("/workspace/minute_alpha/REPORT.md")
    marker = "# 1m / 5m / 15m 同一周对比"
    previous = report.read_text(encoding="utf-8") if report.exists() else ""
    if marker in previous:
        previous = previous.split(marker)[0].rstrip() + "\n\n"
    report.write_text(previous + body, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["all", "5m", "15m288", "15m864", "compare"], default="all")
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()
    jobs = args.jobs or int(machine()["cpu_count"] or 1)
    if args.variant == "compare":
        five = {
            "name": "5m / 864 bars (3d)",
            "metrics": pd.read_csv("/workspace/minute_alpha/portal_out_7d_5m/week_metrics.csv"),
            "timing": json.loads(Path("/workspace/minute_alpha/portal_out_7d_5m/week_timing.json").read_text()),
            "daily": pd.read_csv("/workspace/minute_alpha/portal_out_7d_5m/daily_portfolio_returns.csv"),
        }
        slow = {
            "name": "15m / 288 bars (3d)",
            "metrics": pd.read_csv("/workspace/minute_alpha/portal_out_7d_15m/week_metrics.csv"),
            "timing": json.loads(Path("/workspace/minute_alpha/portal_out_7d_15m/week_timing.json").read_text()),
            "daily": pd.read_csv("/workspace/minute_alpha/portal_out_7d_15m/daily_portfolio_returns.csv"),
        }
        fast_long = {
            "name": "15m / 864 bars (9d)",
            "metrics": pd.read_csv("/workspace/minute_alpha/portal_out_7d_15m/window_864/week_metrics.csv"),
            "timing": json.loads(Path("/workspace/minute_alpha/portal_out_7d_15m/window_864/week_timing.json").read_text()),
            "daily": pd.read_csv("/workspace/minute_alpha/portal_out_7d_15m/window_864/daily_portfolio_returns.csv"),
        }
        write_comparison([five, slow, fast_long])
        return
    if args.variant == "5m":
        run_variant(
            "5m", 864,
            Path("/workspace/minute_alpha/portal_out_7d_5m"),
            Path("/opt/cursor/artifacts/minute_alpha_5m"),
            n_jobs=jobs,
        )
        return
    if args.variant == "15m288":
        run_variant(
            "15m", 288,
            Path("/workspace/minute_alpha/portal_out_7d_15m"),
            Path("/opt/cursor/artifacts/minute_alpha_15m"),
            n_jobs=jobs,
            refresh=False,
        )
        return
    if args.variant == "15m864":
        run_variant(
            "15m", 864,
            Path("/workspace/minute_alpha/portal_out_7d_15m/window_864"),
            Path("/opt/cursor/artifacts/minute_alpha_15m/window_864"),
            n_jobs=jobs,
            refresh=False,
        )
        return
    jobs = int(machine()["cpu_count"] or 1)
    five = run_variant(
        "5m", 864,
        Path("/workspace/minute_alpha/portal_out_7d_5m"),
        Path("/opt/cursor/artifacts/minute_alpha_5m"),
        n_jobs=jobs,
    )
    five["name"] = "5m / 864 bars (3d)"
    slow = run_variant(
        "15m", 288,
        Path("/workspace/minute_alpha/portal_out_7d_15m"),
        Path("/opt/cursor/artifacts/minute_alpha_15m"),
        n_jobs=jobs,
    )
    slow["name"] = "15m / 288 bars (3d)"
    fast_long = run_variant(
        "15m", 864,
        Path("/workspace/minute_alpha/portal_out_7d_15m/window_864"),
        Path("/opt/cursor/artifacts/minute_alpha_15m/window_864"),
        n_jobs=jobs,
    )
    fast_long["name"] = "15m / 864 bars (9d)"
    write_comparison([five, slow, fast_long])
    print("done", flush=True)


if __name__ == "__main__":
    main()
