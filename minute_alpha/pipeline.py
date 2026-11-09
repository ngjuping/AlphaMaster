"""Rolling 1-minute retrain, inference, and walk-forward backtest."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from minute_alpha.backtest import (
    annualization,
    bar_pnl,
    compound_equity,
    cost_1bp,
    cost_bitfinex,
    summarize,
)
from minute_alpha.config import (
    BARS,
    BFX_TAKER_FEE,
    N_FORMULAS,
    OOS_FRACTION,
    RETRAIN_EVERY,
    SEED,
    SLIPPAGE_1BP,
    TRAIN_WINDOW,
)
from minute_alpha.data import load_universe, to_matrices
from minute_alpha.diagnostics import (
    factor_target_frame,
    plot_equity,
    plot_factor_cumsum,
    plot_position_cumsum,
    position_history_frame,
    position_target_frame,
    time_utc,
)
from minute_alpha.portal import build_status, write_status
from minute_alpha.features import compute_features
from minute_alpha.position import target_positions
from minute_alpha.search import make_candidates, search
from minute_alpha.target import target_log_open
from minute_alpha.timing import StageClock, machine
from minute_alpha.vm import decode, execute

ARTIFACTS = Path("/opt/cursor/artifacts/minute_alpha")
REPO_OUT = Path("/workspace/minute_alpha/portal_out")


def _slice_raw(fields: dict, symbol_i: int, start: int, end: int) -> dict:
    return {k: fields[k][symbol_i : symbol_i + 1, start:end] for k in ("open", "high", "low", "close", "volume")}


def _fit(raw: dict, times: np.ndarray, candidates, n_jobs: int) -> tuple:
    t0 = time.perf_counter()
    feat = compute_features(raw)
    feature_s = time.perf_counter() - t0
    target = target_log_open(raw["open"])
    periods = annualization(times)
    t1 = time.perf_counter()
    best, ranked, _pool = search(
        feat,
        target,
        periods,
        n_formulas=len(candidates),
        n_jobs=n_jobs,
        seed=SEED,
        candidates=candidates,
    )
    search_s = time.perf_counter() - t1
    return best, feature_s, search_s, periods


def run(
    bars: int = BARS,
    n_formulas: int = N_FORMULAS,
    retrain_every: int = RETRAIN_EVERY,
    n_jobs: int = 0,
    train_window: int = TRAIN_WINDOW,
    oos_fraction: float = OOS_FRACTION,
    budget_seconds: float = 12 * 60,
) -> dict:
    """Download closed bars, roll the production score, write artifacts."""
    clock = StageClock()
    info = machine()
    if n_jobs <= 0:
        n_jobs = int(info["cpu_count"] or 1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with clock.stage("data_load"):
        aligned = load_universe(bars=bars)
        fields = to_matrices(aligned)
    symbols = [str(s) for s in fields["symbols"]]
    times = fields["time"]
    n, t = fields["close"].shape
    oos_start = int(t * oos_fraction)
    oos_start = min(max(oos_start, train_window), t - 3)
    true_target = target_log_open(fields["open"])
    candidates = make_candidates(n_formulas, SEED)

    # One window, two core counts, before the walk-forward.
    window_start = max(0, oos_start - train_window + 1)
    raw0 = _slice_raw(fields, 0, window_start, oos_start + 1)
    times0 = times[window_start : oos_start + 1]
    with clock.stage("retrain_1core"):
        best_1, feat_s_1, search_s_1, _ = _fit(raw0, times0, candidates, n_jobs=1)
    with clock.stage("retrain_multicore"):
        best_n, feat_s_n, search_s_n, _ = _fit(raw0, times0, candidates, n_jobs=n_jobs)
    one_core = feat_s_1 + search_s_1
    multi_core = feat_s_n + search_s_n
    if best_1 is None or best_n is None or best_1.formula != best_n.formula:
        raise RuntimeError("1-core and multi-core champions diverged")

    retrain_rows: list[dict] = []
    bench_finished = time.time()

    def _timing_row(kind: str, symbol: str, decision: int, start: int, feature_s: float,
                    search_s: float, inference_s: float, best, jobs: int) -> dict:
        total = feature_s + search_s + inference_s
        return {
            "kind": kind,
            "symbol": symbol,
            "decision_unix": int(times[decision]),
            "decision_utc": time_utc(int(times[decision])),
            "train_start_unix": int(times[start]),
            "train_end_unix": int(times[decision]),
            "n_bars": int(decision - start + 1),
            "cpu_count": int(info["cpu_count"] or 1),
            "n_jobs": int(jobs),
            "n_formulas": len(candidates),
            "feature_seconds": feature_s,
            "search_seconds": search_s,
            "inference_seconds": inference_s,
            "total_seconds": total,
            "finished_at_unix": bench_finished,
            "finished_at_utc": time_utc(int(bench_finished)),
            "champion": None if best is None else best.name,
            "formula": None if best is None else decode(best.formula),
            "train_score": None if best is None else best.train_score,
            "val_score": None if best is None else best.val_score,
        }

    retrain_rows.append(_timing_row(
        "benchmark_1core", symbols[0], oos_start, window_start, feat_s_1, search_s_1, 0.0, best_1, 1,
    ))
    retrain_rows.append(_timing_row(
        "benchmark_multicore", symbols[0], oos_start, window_start, feat_s_n, search_s_n, 0.0, best_n, n_jobs,
    ))

    decisions = list(range(oos_start, t, retrain_every))
    if not decisions:
        decisions = [oos_start]
    estimate = multi_core * n * len(decisions)
    if estimate > budget_seconds and multi_core > 0:
        scale = int(np.ceil(estimate / budget_seconds))
        retrain_every = int(retrain_every * scale)
        decisions = list(range(oos_start, t, retrain_every))

    positions = np.zeros((n, t), dtype=np.float64)
    factors = np.zeros((n, t), dtype=np.float64)
    names = [""] * n
    formulas: list[list[int]] = [[] for _ in range(n)]
    name_at = np.full((n, t), "", dtype=object)
    valid_from = np.full(n, t, dtype=int)

    for s in range(n):
        for di, decision in enumerate(decisions):
            hold_end = t - 1 if di + 1 == len(decisions) else decisions[di + 1] - 1
            start = max(0, decision - train_window + 1)
            raw = _slice_raw(fields, s, start, decision + 1)
            best, feature_s, search_s, _periods = _fit(raw, times[start : decision + 1], candidates, n_jobs)
            clock.add("feature", feature_s)
            clock.add("search", search_s)
            hold_feature_s = 0.0
            inference_s = 0.0
            if best is not None:
                names[s] = best.name
                formulas[s] = list(best.formula)
                hold = _slice_raw(fields, s, start, hold_end + 1)
                t_feat = time.perf_counter()
                feat_hold = compute_features(hold)
                hold_feature_s = time.perf_counter() - t_feat
                clock.add("feature", hold_feature_s)
                t_inf = time.perf_counter()
                factor = execute(best.formula, feat_hold)
                inference_s = time.perf_counter() - t_inf
                clock.add("inference", inference_s)
                if factor is not None:
                    series = np.asarray(factor[0], dtype=np.float64)
                    pos = target_positions(series)
                    lo = start if di == 0 else decision
                    local_lo = lo - start
                    local_hi = hold_end + 1 - start
                    factors[s, lo : hold_end + 1] = series[local_lo:local_hi]
                    positions[s, lo : hold_end + 1] = pos[local_lo:local_hi]
                    name_at[s, lo : hold_end + 1] = best.name
                    valid_from[s] = min(int(valid_from[s]), lo)
            finished = time.time()
            row = _timing_row(
                "walk_forward", symbols[s], decision, start,
                feature_s + hold_feature_s, search_s, inference_s, best, n_jobs,
            )
            row["finished_at_unix"] = finished
            row["finished_at_utc"] = time_utc(int(finished))
            row["hold_end_unix"] = int(times[hold_end])
            retrain_rows.append(row)

    # Live-style snapshot: retrain on everything up to the last closed bar.
    final_started = time.time()
    with clock.stage("final_retrain"):
        final_rows = []
        final_duration = 0.0
        for s, symbol in enumerate(symbols):
            start = max(0, t - train_window)
            raw = _slice_raw(fields, s, start, t)
            t_sym = time.perf_counter()
            best, feature_s, search_s, _p = _fit(raw, times[start:t], candidates, n_jobs)
            t_feat = time.perf_counter()
            feat = compute_features(raw)
            feature_s += time.perf_counter() - t_feat
            t_inf = time.perf_counter()
            factor = execute(best.formula, feat) if best is not None else None
            inference_s = time.perf_counter() - t_inf
            sym_total = time.perf_counter() - t_sym
            final_duration += sym_total
            pos = float("nan") if factor is None else float(target_positions(factor[0, -1:]).ravel()[0])
            finished = time.time()
            row = _timing_row(
                "latest_snapshot", symbol, t - 1, start, feature_s, search_s, inference_s, best, n_jobs,
            )
            row["finished_at_unix"] = finished
            row["finished_at_utc"] = time_utc(int(finished))
            retrain_rows.append(row)
            final_rows.append(
                {
                    "symbol": symbol,
                    "pair": f"t{symbol}F0:USTF0",
                    "asof_unix": int(times[-1]),
                    "asof_utc": time_utc(int(times[-1])),
                    "factor_name": None if best is None else best.name,
                    "formula": None if best is None else decode(best.formula),
                    "formula_tokens": None if best is None else list(best.formula),
                    "factor_value": None if factor is None else float(factor[0, -1]),
                    "target_position": pos,
                    "train_score": None if best is None else best.train_score,
                    "val_score": None if best is None else best.val_score,
                    "retrain_seconds": sym_total,
                }
            )
    final_finished = time.time()

    # Backtest uses bars whose label is inside the sample (drop the last two).
    usable = t - 2
    oos_slice = slice(oos_start, usable)
    periods = annualization(times[:usable])
    pnl_1, to_1 = bar_pnl(positions[:, :usable], true_target[:, :usable], cost_1bp())
    pnl_b, to_b = bar_pnl(positions[:, :usable], true_target[:, :usable], cost_bitfinex())
    port_1 = pnl_1.mean(axis=0)
    port_b = pnl_b.mean(axis=0)
    port_to = to_1.mean(axis=0)
    metrics_1 = summarize(port_1[oos_slice], port_to[oos_slice], periods)
    metrics_b = summarize(port_b[oos_slice], to_b.mean(axis=0)[oos_slice], periods)
    # Full path for the chart, including the in-sample segment left of the line.
    equity_1 = compound_equity(port_1)
    equity_b = compound_equity(port_b)

    factor_map = {symbol: factors[i, :usable] for i, symbol in enumerate(symbols)}
    name_map = {symbol: decode(formulas[i]) if formulas[i] else names[i] for i, symbol in enumerate(symbols)}
    factor_df = factor_target_frame(times[:usable], symbols, factor_map, true_target[:, :usable], oos_start, name_map)
    pos_df = position_target_frame(times[:usable], symbols, positions[:, :usable], true_target[:, :usable], oos_start)
    equity_df = pd.DataFrame(
        {
            "time": times[:usable],
            "segment": np.where(np.arange(usable) >= oos_start, "oos", "train"),
            "pnl_1bp": port_1,
            "pnl_bfx": port_b,
            "turnover": port_to,
            "equity_1bp": equity_1,
            "equity_bfx": equity_b,
        }
    )

    history_df = position_history_frame(
        times[:usable], symbols, positions[:, :usable], factors[:, :usable], name_at[:, :usable], oos_start, valid_from,
    )
    timing_df = pd.DataFrame(retrain_rows)
    _write_table(factor_df, "factor_target_cumsum.csv")
    _write_table(pos_df, "position_target_cumsum.csv")
    _write_table(equity_df, "equity.csv")
    _write_table(history_df, "position_history.csv")
    _write_table(timing_df, "retrain_timing_history.csv")
    _write_table(pd.DataFrame(final_rows), "next_minute_positions.csv")
    for folder in (ARTIFACTS, REPO_OUT):
        folder.mkdir(parents=True, exist_ok=True)
        plot_factor_cumsum(factor_df, times[:usable], oos_start, folder / "factor_target_cumsum.png")
        plot_position_cumsum(pos_df, times[:usable], oos_start, folder / "position_target_cumsum.png")
        plot_equity(times[:usable], equity_1, equity_b, oos_start, folder / "equity.png")

    speedup = one_core / multi_core if multi_core > 0 else float("nan")
    result = {
        "machine": info,
        "symbols": symbols,
        "bars": int(t),
        "oos_start_index": int(oos_start),
        "oos_start_unix": int(times[oos_start]),
        "retrain_every_min": int(retrain_every),
        "n_decisions": len(decisions),
        "n_formulas": len(candidates),
        "n_jobs": n_jobs,
        "train_window": train_window,
        "one_core_seconds": one_core,
        "one_core_feature_seconds": feat_s_1,
        "one_core_search_seconds": search_s_1,
        "multi_core_seconds": multi_core,
        "multi_core_feature_seconds": feat_s_n,
        "multi_core_search_seconds": search_s_n,
        "speedup": speedup,
        "fits_in_one_minute_1core": bool(one_core < 60),
        "fits_in_one_minute_multicore": bool(multi_core < 60),
        "metrics_1bp": metrics_1,
        "metrics_bfx": metrics_b,
        "bfx_taker_fee": BFX_TAKER_FEE,
        "slippage_1bp": SLIPPAGE_1BP,
        "next_positions": final_rows,
        "stages": clock.stages,
        "stage_rss_mb": clock.rss,
        "champion_1core": None if best_1 is None else best_1.name,
        "champion_formula": None if best_n is None else decode(best_n.formula),
    }
    last_retrain = {
        "started_at_unix": int(final_started),
        "started_at_utc": time_utc(int(final_started)),
        "finished_at_unix": int(final_finished),
        "finished_at_utc": time_utc(int(final_finished)),
        "duration_seconds": float(final_finished - final_started),
        "cpu_count": int(info["cpu_count"] or 1),
        "n_jobs": int(n_jobs),
        "n_formulas": len(candidates),
        "train_window_bars": int(train_window),
        "data_cutoff_unix": int(times[-1]),
        "data_cutoff_utc": time_utc(int(times[-1])),
    }
    status = build_status(
        cpu_count=int(info["cpu_count"] or 1),
        n_jobs=n_jobs,
        data_cutoff_unix=int(times[-1]),
        positions=final_rows,
        last_retrain=last_retrain,
        timing={
            "one_core_seconds": one_core,
            "one_core_feature_seconds": feat_s_1,
            "one_core_search_seconds": search_s_1,
            "multi_core_seconds": multi_core,
            "multi_core_feature_seconds": feat_s_n,
            "multi_core_search_seconds": search_s_n,
            "speedup": speedup,
            "fits_in_one_minute_1core": result["fits_in_one_minute_1core"],
            "fits_in_one_minute_multicore": result["fits_in_one_minute_multicore"],
        },
        symbols=symbols,
        retrain_every_minutes=int(retrain_every),
        bars=int(t),
        oos_start_unix=int(times[oos_start]),
    )
    result["status"] = status
    for folder in (ARTIFACTS, REPO_OUT):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "timing.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        write_status(folder / "status.json", status)
    report = _report(result)
    (ARTIFACTS / "REPORT.md").write_text(report, encoding="utf-8")
    (REPO_OUT / "REPORT.md").write_text(report, encoding="utf-8")
    Path("/workspace/minute_alpha/REPORT.md").write_text(report, encoding="utf-8")
    return result


def _write_table(frame: pd.DataFrame, name: str) -> None:
    for folder in (ARTIFACTS, REPO_OUT):
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / name, index=False)


def _report(result: dict) -> str:
    m1 = result["metrics_1bp"]
    mb = result["metrics_bfx"]
    lines = [
        "# 1 分钟滚动重训报告",
        "",
        "## 生产因子是哪一套",
        "",
        "仓库当前默认生产路径（2026-07-07 起）是**单品种** AlphaGPT，不是已经标成废弃的分组训练：",
        "",
        "- 挖掘：`train_file.py` / `AlphaEngine`，`REWARD_MODE=ftmo`，公式长度 8，5 折 walk-forward，gap 20。",
        "- 词表：`FEATURE_REGISTRY` 的 65 个特征 + `OPERATOR_REGISTRY` 的 62 个算子（仓库里没有 `active_features.json`，因此 65 个特征全部在生产词表里）。",
        "- 因子值：StackVM 执行后做因果标准化（单品种为 500 期滚动 z-score，裁剪到 [-3, 3]）。这就是进入 `tanh` 的值。",
        "- 组合：训练时用 top-25 因子池，只在训练切片上做 |corr|>0.85 的惩罚（×0.8）。实盘落地的是每个品种**一条**验证分最高的公式（`strategies/best_{symbol}.json`），不是池子加权。品种之间等权。",
        "- 仓位：`position = tanh(factor)`，绝对值小于 0.05 记为 0，范围 [-1, 1]。",
        "- 标签：`target[t] = log(open[t+2] / open[t+1])`，最后两根为 0。这不是下一分钟的简单收益率。信号在 t 收盘可知，赚的是下一根 bar 的 open-to-open 对数收益。",
        "- 生产交易簿在 `live_trade.py`：H1 的 XAUUSD 和美股指数，不是加密。本次以 Bitfinex USDt 永续 1 分钟为主跑通，品种为 BTC、ETH、SOL、XRP、ADA、DOGE、LTC（`t{BASE}F0:USTF0`）。",
        "",
        "重写在 `minute_alpha/`，不改原包。因子执行、标签、FTMO 打分、相关性惩罚和 tanh 仓位按上面的定义实现。",
        "采样不是 9000 步的 REINFORCE：那套策略网络无法在一分钟内跑完，而且跨进程不保证同一种子得到同一条公式。",
        "这里用固定种子的约束采样（含 `strategies/README.md` 里三条已发表公式作为锚点），并行打分后按固定顺序施加相关性惩罚。",
        "同一输入、同一候选公式，1 核和多核选出的冠军公式一致。",
        "",
        "## 数据",
        "",
        f"- 已闭合 1 分钟 bar：{result['bars']} 根，品种 {', '.join(result['symbols'])}。",
        "- 流动性差的永续不是每分钟都有成交。对齐用 60 秒网格上的向后 asof：每分钟只用该分钟或更早的最后一根已闭合 candle，没有新成交则价格保持、成交量记 0。不用未来 candle。",
        f"- 样本外起点 index {result['oos_start_index']}（unix {result['oos_start_unix']}）。",
        f"- 每次重训窗口 {result['train_window']} 根，只含决策 bar 及更早的数据。标签在窗口末尾两根被置 0，避免用到尚未发生的 open。",
        f"- 滚动间隔 {result['retrain_every_min']} 分钟，决策次数 {result['n_decisions']}，每窗公式数 {result['n_formulas']}。",
        "",
        "## 耗时",
        "",
        f"机器：{result['machine']['cpu_count']} CPU，内存约 {result['machine'].get('mem_total_mb')} MB，"
        f"运行时 RSS 见各阶段。",
        "",
        "下表是**同一训练窗、同一批公式**的重训（特征 + 搜索），不含下载：",
        "",
        "| 配置 | 特征秒 | 搜索秒 | 合计秒 |",
        "| --- | --- | --- | --- |",
        f"| 1 核 | {result['one_core_feature_seconds']:.3f} | {result['one_core_search_seconds']:.3f} | {result['one_core_seconds']:.3f} |",
        f"| {result['n_jobs']} 核 | {result['multi_core_feature_seconds']:.3f} | {result['multi_core_search_seconds']:.3f} | {result['multi_core_seconds']:.3f} |",
        "",
        f"加速比（1 核合计 / 多核合计）= {result['speedup']:.2f}x。",
        f"单次重训能否放进 1 分钟：1 核 {'能' if result['fits_in_one_minute_1core'] else '不能'}，"
        f"多核 {'能' if result['fits_in_one_minute_multicore'] else '不能'}。",
        "",
        "全流程阶段墙钟（秒）含全部品种的滚动重训：",
        "",
    ]
    for name, seconds in result["stages"].items():
        rss = result["stage_rss_mb"].get(name)
        lines.append(f"- {name}: {seconds:.3f}s，RSS {rss:.1f} MB" if isinstance(rss, float) else f"- {name}: {seconds:.3f}s")
    lines += [
        "",
        "原版 `AlphaEngine` 的逐步耗时和同公式的 StackVM 对比在 `bench` 的输出里（`timing_vs_original.json`）。",
        "原版默认 `TRAIN_STEPS=9000`、`BATCH_SIZE=192`。按单步耗时外推，完整预算不能按分钟滚动。",
        "",
        "## 下一分钟目标仓位",
        "",
        "映射：标准化因子 → `tanh` → 绝对值 < 0.05 置 0 → 裁剪到 [-1, 1]。",
        "数值是该币名义本金的目标比例，+1 满多，-1 满空。多币组合是这些比例的等权平均，不再做二次杠杆。",
        "下表是最后一根已闭合 bar 上、用该 bar 及之前的窗口重训得到的仓位，对应再下一分钟的持仓。",
        "",
        "| 币 | 公式 | 因子 | 目标仓位 |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["next_positions"]:
        lines.append(
            f"| {row['symbol']} | {row['factor_name']} | {row['factor_value']} | {row['target_position']} |"
        )
    lines += [
        "",
        "## 样本外回测",
        "",
        "收益对齐生产标签：`position[t] * log(open[t+2]/open[t+1]) - |Δposition| * cost`。",
        "Sharpe 用样本外组合逐 bar 收益，年化因子由时间戳跨度估计（与 `estimate_periods_per_year` 相同）。",
        "MDD 是 `exp(cumsum(pnl))` 财富指数的峰值回撤。换手是 `|Δposition|` 的样本外均值。",
        "",
        "| 成本 | Sharpe | MDD | 换手均值 | 期末权益 |",
        "| --- | --- | --- | --- | --- |",
        f"| 0 手续费 + 1bp 滑点 | {m1['sharpe']:.3f} | {m1['mdd']:.3%} | {m1['turnover_mean']:.5f} | {m1['equity_end']:.4f} |",
        f"| Bitfinex 衍生品吃单费率 {result['bfx_taker_fee']} | {mb['sharpe']:.3f} | {mb['mdd']:.3%} | {mb['turnover_mean']:.5f} | {mb['equity_end']:.4f} |",
        "",
        "Bitfinex 费率取自 2026-11-09 抓取的 https://www.bitfinex.com/fees/ ：",
        "Derivatives Trades 的 Maker fees 与 Taker Fees 均为 Zero。",
        "因此「实际费率」版本的交易成本是 0，曲线会高于 1bp 滑点版本。资金费（funding）不在该订单执行费率里，本回测没有扣。",
        "",
        "## 诊断图",
        "",
        "两组图都画在同一条时间轴上，竖线是训练 / 样本外分界。竖线右侧才是 walk-forward 样本外。",
        "CSV 里的 `oos_cumsum` 从分界处重新累加。",
        "",
        "1. `factor_target_cumsum.csv`：每个品种冠军公式的**标准化因子**乘训练标签的累加，另有一行 `ALL` 等权平均。",
        "   这是信号诊断，说明因子和标签是否同向。它没有换手成本，**不是策略收益**。",
        "2. `position_target_cumsum.csv`：目标仓位乘同一标签的累加（每币和等权组合）。同样不含手续费和滑点。",
        "   策略收益以 `equity.csv` / `equity.png` 为准，那里扣了成本。",
        "",
        "## 门户「AlphaMaster 在线学习」对接",
        "",
        "页面读 `minute_alpha/portal_out/status.json`（同一份也在 `/opt/cursor/artifacts/minute_alpha/`）。",
        "`schema_version` 为 `1.0.0`。时间字段成对出现：`*_unix` 是 UTC 秒，`*_utc` 是 `YYYY-MM-DDTHH:MM:SSZ`。",
        "仓位是连续比例，范围 [-1, 1]，不是手数。",
        "",
        "### status.json",
        "",
        "| 字段 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| page | string | 固定 `alphamaster_online_learning` |",
        "| schema_version | string | 本文件结构版本 |",
        "| package_version | string | `minute_alpha` 包版本 |",
        "| git_commit | string | 跑这次结果时的 `git rev-parse HEAD` |",
        "| git_branch | string | 当时的分支名 |",
        "| generated_at_unix / generated_at_utc | int / string | 写出 status 的时刻 |",
        "| cpu_count | int | 机器逻辑核数 |",
        "| n_jobs | int | 这次重训实际用的进程数 |",
        "| symbols | string[] | 币种顺序，与 CSV 的 symbol 一致 |",
        "| bars | int | 对齐后的已闭合 1 分钟根数 |",
        "| retrain_every_minutes | int | 滚动重训间隔（分钟） |",
        "| data_cutoff_unix / data_cutoff_utc | int / string | 最后一根已闭合 bar 的开盘时间。重训没用这根之后的数据 |",
        "| oos_start_unix / oos_start_utc | int / string | walk-forward 样本外第一根 bar |",
        "| last_retrain.started_at_unix / started_at_utc | int / string | 最近一次全币快照重训开始 |",
        "| last_retrain.finished_at_unix / finished_at_utc | int / string | 该次重训结束 |",
        "| last_retrain.duration_seconds | float | 该次墙钟秒数（含全部币，串行） |",
        "| last_retrain.cpu_count / n_jobs / n_formulas / train_window_bars | int | 核数、进程数、公式数、训练窗根数 |",
        "| timing.one_core_seconds / timing.multi_core_seconds | float | 同一窗口 1 核与多核重训（特征+搜索） |",
        "| timing.speedup | float | one_core_seconds / multi_core_seconds |",
        "| timing.fits_in_one_minute_1core / fits_in_one_minute_multicore | bool | 单次重训是否短于 60 秒 |",
        "| positions[] | object[] | 当前各币目标仓位，见下表 |",
        "| files | object | 相对本目录的 CSV 文件名 |",
        "",
        "`positions[]` 每个元素：",
        "",
        "| 字段 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| symbol | string | BTC / ETH / SOL / XRP / ADA / DOGE / LTC |",
        "| pair | string | Bitfinex 永续，如 `tBTCF0:USTF0` |",
        "| asof_unix / asof_utc | int / string | 与 data_cutoff 相同，这根 bar 的开盘时间 |",
        "| factor_name | string | 冠军公式名称（锚点名或 sample_N） |",
        "| formula | string | 用 `→` 连接的 token 名 |",
        "| formula_tokens | int[] | 与生产词表一致的 token id |",
        "| factor_value | float | 进入 tanh 的标准化因子 |",
        "| target_position | float | 下一分钟目标仓位，[-1, 1] |",
        "| train_score / val_score | float | 训练窗内的 FTMO 训练分 / 验证分 |",
        "| retrain_seconds | float | 该币这一次快照重训的秒数 |",
        "",
        "### position_history.csv（目标仓位历史）",
        "",
        "| 列 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| time | int | bar 开盘 unix 秒 |",
        "| time_utc | string | 同上，UTC |",
        "| segment | string | `train` 或 `oos`。`oos` 从 oos_start 起 |",
        "| symbol | string | 币种，或 `ALL`（等权平均仓位） |",
        "| target_position | float | 该 bar 收盘时的目标仓位 [-1, 1] |",
        "| factor_value | float | 标准化因子。`ALL` 行为空 |",
        "| factor_name | string | 产生该段仓位的公式。滚动换公式后，后续行换成新公式 |",
        "",
        "### factor_target_cumsum.csv（因子 × target 累加）",
        "",
        "信号诊断，不是扣费后的策略收益。`target` 是生产训练标签 `log(open[t+2]/open[t+1])`。",
        "",
        "| 列 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| time / time_utc | int / string | bar 开盘时间 |",
        "| symbol | string | 币种或 `ALL` |",
        "| factor_name | string | 公式；`ALL` 行为 `equal_weight_mean` |",
        "| segment | string | `train` 或 `oos` |",
        "| factor | float | 标准化因子。`ALL` 行为空 |",
        "| target | float | 训练标签。`ALL` 行为空 |",
        "| product | float | 单币是 factor×target；`ALL` 是各币 product 的等权平均 |",
        "| product_sum | float | 仅 `ALL` 行：各币 product 之和 |",
        "| cumsum | float | 从序列起点累加 product |",
        "| oos_cumsum | float | 只在样本外从 0 重新累加；训练段为空 |",
        "",
        "### position_target_cumsum.csv（仓位 × target 累加）",
        "",
        "同样不含手续费和滑点。扣费后的曲线在 `equity.csv`。",
        "",
        "| 列 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| time / time_utc | int / string | bar 开盘时间 |",
        "| symbol | string | 币种或 `ALL` |",
        "| segment | string | `train` 或 `oos` |",
        "| position | float | 目标仓位。`ALL` 是等权平均 |",
        "| target | float | 训练标签。`ALL` 行为空 |",
        "| product | float | position×target；`ALL` 为等权平均 |",
        "| cumsum / oos_cumsum | float | 与因子诊断相同的两种累加 |",
        "",
        "### retrain_timing_history.csv（重训耗时历史）",
        "",
        "| 列 | 类型 | 含义 |",
        "| --- | --- | --- |",
        "| kind | string | `benchmark_1core`、`benchmark_multicore`、`walk_forward`、`latest_snapshot` |",
        "| symbol | string | 这一行测的币 |",
        "| decision_unix / decision_utc | int / string | 决策 bar（训练窗最后一根） |",
        "| train_start_unix / train_end_unix | int | 训练窗起止，含端点。train_end 等于 decision |",
        "| n_bars | int | 训练窗根数 |",
        "| cpu_count | int | 机器核数 |",
        "| n_jobs | int | 该行使用的进程数。1core 基准为 1 |",
        "| n_formulas | int | 候选公式数 |",
        "| feature_seconds | float | 特征计算 |",
        "| search_seconds | float | 公式搜索与打分 |",
        "| inference_seconds | float | 冻结公式后的因子与仓位 |",
        "| total_seconds | float | 以上三项之和 |",
        "| finished_at_unix / finished_at_utc | int / string | 该行完成时刻 |",
        "| champion | string | 冠军公式名 |",
        "| formula | string | token 名序列 |",
        "| train_score / val_score | float | 训练分 / 验证分 |",
        "| hold_end_unix | int | 仅 walk_forward：该公式一直用到的最后一根 bar |",
        "",
        "## 已知局限",
        "",
        "- 重写搜索是单次种子采样，不是 Transformer 策略梯度。冠军公式不必与某次原版长训相同。",
        "- 单品种生产路径下，横截面算子退化为恒等（与原版 N=1 一致）。",
        "- `WILLR_14` 在原版里被裁到 [-1, 0]，正常 OHLC 上会变成 0。重写保持这一行为。",
        "- 部分特征先做 200 期 median/MAD。float32 上一个 ulp 的差别会换掉下中位数，少数特征点会和 torch 差到 1e-2 量级。",
        "  同一特征张量上的算子，以及 N=1 的锚点公式因子值，对照测试要求误差在 1e-4 以内。",
        "- N>1 时，横截面 z-score 会把接近 0 的残差放大。生产推理是 N=1，不走这条路径。",
        "- 训练打分里的换手质量项对连续仓位几乎恒为 -2（原版 `int(tanh)` 把 (-1,1) 截成 0），不改变排序。",
        "- 没有连接实盘、没有交易密钥、没有合并 main。",
        "",
    ]
    return "\n".join(lines) + "\n"
