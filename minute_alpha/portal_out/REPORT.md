# 1 分钟滚动重训报告

## 生产因子是哪一套

仓库当前默认生产路径（2026-07-07 起）是**单品种** AlphaGPT，不是已经标成废弃的分组训练：

- 挖掘：`train_file.py` / `AlphaEngine`，`REWARD_MODE=ftmo`，公式长度 8，5 折 walk-forward，gap 20。
- 词表：`FEATURE_REGISTRY` 的 65 个特征 + `OPERATOR_REGISTRY` 的 62 个算子（仓库里没有 `active_features.json`，因此 65 个特征全部在生产词表里）。
- 因子值：StackVM 执行后做因果标准化（单品种为 500 期滚动 z-score，裁剪到 [-3, 3]）。这就是进入 `tanh` 的值。
- 组合：训练时用 top-25 因子池，只在训练切片上做 |corr|>0.85 的惩罚（×0.8）。实盘落地的是每个品种**一条**验证分最高的公式（`strategies/best_{symbol}.json`），不是池子加权。品种之间等权。
- 仓位：`position = tanh(factor)`，绝对值小于 0.05 记为 0，范围 [-1, 1]。
- 标签：`target[t] = log(open[t+2] / open[t+1])`，最后两根为 0。这不是下一分钟的简单收益率。信号在 t 收盘可知，赚的是下一根 bar 的 open-to-open 对数收益。
- 生产交易簿在 `live_trade.py`：H1 的 XAUUSD 和美股指数，不是加密。本次以 Bitfinex USDt 永续 1 分钟为主跑通，品种为 BTC、ETH、SOL、XRP、ADA、DOGE、LTC（`t{BASE}F0:USTF0`）。

重写在 `minute_alpha/`，不改原包。因子执行、标签、FTMO 打分、相关性惩罚和 tanh 仓位按上面的定义实现。
采样不是 9000 步的 REINFORCE：那套策略网络无法在一分钟内跑完，而且跨进程不保证同一种子得到同一条公式。
这里用固定种子的约束采样（含 `strategies/README.md` 里三条已发表公式作为锚点），并行打分后按固定顺序施加相关性惩罚。
同一输入、同一候选公式，1 核和多核选出的冠军公式一致。

## 数据

- 已闭合 1 分钟 bar：9430 根，品种 BTC, ETH, SOL, XRP, ADA, DOGE, LTC。
- 流动性差的永续不是每分钟都有成交。对齐用 60 秒网格上的向后 asof：每分钟只用该分钟或更早的最后一根已闭合 candle，没有新成交则价格保持、成交量记 0。不用未来 candle。
- 样本外起点 index 5658（unix 1793967060）。
- 每次重训窗口 4320 根，只含决策 bar 及更早的数据。标签在窗口末尾两根被置 0，避免用到尚未发生的 open。
- 滚动间隔 60 分钟，决策次数 63，每窗公式数 32。

## 耗时

机器：4 CPU，内存约 16003.25 MB，运行时 RSS 见各阶段。

下表是**同一训练窗、同一批公式**的重训（特征 + 搜索），不含下载：

| 配置 | 特征秒 | 搜索秒 | 合计秒 |
| --- | --- | --- | --- |
| 1 核 | 0.292 | 0.216 | 0.508 |
| 4 核 | 0.184 | 0.081 | 0.265 |

加速比（1 核合计 / 多核合计）= 1.92x。
单次重训能否放进 1 分钟：1 核 能，多核 能。

全流程阶段墙钟（秒）含全部品种的滚动重训：

- data_load: 0.087s，RSS 198.3 MB
- retrain_1core: 0.509s，RSS 210.6 MB
- retrain_multicore: 0.265s，RSS 213.2 MB
- feature: 151.197s，RSS 221.1 MB
- search: 31.813s，RSS 221.1 MB
- inference: 1.622s，RSS 221.1 MB
- final_retrain: 2.889s，RSS 221.1 MB

原版 `AlphaEngine` 的逐步耗时和同公式的 StackVM 对比在 `bench` 的输出里（`timing_vs_original.json`）。
原版默认 `TRAIN_STEPS=9000`、`BATCH_SIZE=192`。按单步耗时外推，完整预算不能按分钟滚动。

## 下一分钟目标仓位

映射：标准化因子 → `tanh` → 绝对值 < 0.05 置 0 → 裁剪到 [-1, 1]。
数值是该币名义本金的目标比例，+1 满多，-1 满空。多币组合是这些比例的等权平均，不再做二次杠杆。
下表是最后一根已闭合 bar 上、用该 bar 及之前的窗口重训得到的仓位，对应再下一分钟的持仓。

| 币 | 公式 | 因子 | 目标仓位 |
| --- | --- | --- | --- |
| BTC | sample_12 | -0.3852687478065491 | -0.3672743067206746 |
| ETH | sample_12 | -0.42724689841270447 | -0.4030179369048538 |
| SOL | sample_12 | -0.12351183593273163 | -0.12288758007522244 |
| XRP | forex_v2 | 0.03145222365856171 | 0.0 |
| ADA | sample_12 | -0.1740226000547409 | -0.17228692982012955 |
| DOGE | sample_17 | -3.0 | -0.9950547536867305 |
| LTC | sample_25 | 0.3256857991218567 | 0.3146389652111312 |

## 样本外回测

收益对齐生产标签：`position[t] * log(open[t+2]/open[t+1]) - |Δposition| * cost`。
Sharpe 用样本外组合逐 bar 收益，年化因子由时间戳跨度估计（与 `estimate_periods_per_year` 相同）。
MDD 是 `exp(cumsum(pnl))` 财富指数的峰值回撤。换手是 `|Δposition|` 的样本外均值。

| 成本 | Sharpe | MDD | 换手均值 | 期末权益 |
| --- | --- | --- | --- | --- |
| 0 手续费 + 1bp 滑点 | -8.776 | 2.382% | 0.04469 | 0.9873 |
| Bitfinex 衍生品吃单费率 0.0 | 2.771 | 1.833% | 0.04469 | 1.0040 |

Bitfinex 费率取自 2026-11-09 抓取的 https://www.bitfinex.com/fees/ ：
Derivatives Trades 的 Maker fees 与 Taker Fees 均为 Zero。
因此「实际费率」版本的交易成本是 0，曲线会高于 1bp 滑点版本。资金费（funding）不在该订单执行费率里，本回测没有扣。

## 诊断图

两组图都画在同一条时间轴上，竖线是训练 / 样本外分界。竖线右侧才是 walk-forward 样本外。
CSV 里的 `oos_cumsum` 从分界处重新累加。

1. `factor_target_cumsum.csv`：每个品种冠军公式的**标准化因子**乘训练标签的累加，另有一行 `ALL` 等权平均。
   这是信号诊断，说明因子和标签是否同向。它没有换手成本，**不是策略收益**。
2. `position_target_cumsum.csv`：目标仓位乘同一标签的累加（每币和等权组合）。同样不含手续费和滑点。
   策略收益以 `equity.csv` / `equity.png` 为准，那里扣了成本。

## 门户「AlphaMaster 在线学习」对接

页面读 `minute_alpha/portal_out/status.json`（同一份也在 `/opt/cursor/artifacts/minute_alpha/`）。
`schema_version` 为 `1.0.0`。时间字段成对出现：`*_unix` 是 UTC 秒，`*_utc` 是 `YYYY-MM-DDTHH:MM:SSZ`。
仓位是连续比例，范围 [-1, 1]，不是手数。

### status.json

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| page | string | 固定 `alphamaster_online_learning` |
| schema_version | string | 本文件结构版本 |
| package_version | string | `minute_alpha` 包版本 |
| git_commit | string | 跑这次结果时的 `git rev-parse HEAD` |
| git_branch | string | 当时的分支名 |
| generated_at_unix / generated_at_utc | int / string | 写出 status 的时刻 |
| cpu_count | int | 机器逻辑核数 |
| n_jobs | int | 这次重训实际用的进程数 |
| symbols | string[] | 币种顺序，与 CSV 的 symbol 一致 |
| bars | int | 对齐后的已闭合 1 分钟根数 |
| retrain_every_minutes | int | 滚动重训间隔（分钟） |
| data_cutoff_unix / data_cutoff_utc | int / string | 最后一根已闭合 bar 的开盘时间。重训没用这根之后的数据 |
| oos_start_unix / oos_start_utc | int / string | walk-forward 样本外第一根 bar |
| last_retrain.started_at_unix / started_at_utc | int / string | 最近一次全币快照重训开始 |
| last_retrain.finished_at_unix / finished_at_utc | int / string | 该次重训结束 |
| last_retrain.duration_seconds | float | 该次墙钟秒数（含全部币，串行） |
| last_retrain.cpu_count / n_jobs / n_formulas / train_window_bars | int | 核数、进程数、公式数、训练窗根数 |
| timing.one_core_seconds / timing.multi_core_seconds | float | 同一窗口 1 核与多核重训（特征+搜索） |
| timing.speedup | float | one_core_seconds / multi_core_seconds |
| timing.fits_in_one_minute_1core / fits_in_one_minute_multicore | bool | 单次重训是否短于 60 秒 |
| positions[] | object[] | 当前各币目标仓位，见下表 |
| files | object | 相对本目录的 CSV 文件名 |

`positions[]` 每个元素：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| symbol | string | BTC / ETH / SOL / XRP / ADA / DOGE / LTC |
| pair | string | Bitfinex 永续，如 `tBTCF0:USTF0` |
| asof_unix / asof_utc | int / string | 与 data_cutoff 相同，这根 bar 的开盘时间 |
| factor_name | string | 冠军公式名称（锚点名或 sample_N） |
| formula | string | 用 `→` 连接的 token 名 |
| formula_tokens | int[] | 与生产词表一致的 token id |
| factor_value | float | 进入 tanh 的标准化因子 |
| target_position | float | 下一分钟目标仓位，[-1, 1] |
| train_score / val_score | float | 训练窗内的 FTMO 训练分 / 验证分 |
| retrain_seconds | float | 该币这一次快照重训的秒数 |

### position_history.csv（目标仓位历史）

| 列 | 类型 | 含义 |
| --- | --- | --- |
| time | int | bar 开盘 unix 秒 |
| time_utc | string | 同上，UTC |
| segment | string | `train` 或 `oos`。`oos` 从 oos_start 起 |
| symbol | string | 币种，或 `ALL`（等权平均仓位） |
| target_position | float | 该 bar 收盘时的目标仓位 [-1, 1] |
| factor_value | float | 标准化因子。`ALL` 行为空 |
| factor_name | string | 产生该段仓位的公式。滚动换公式后，后续行换成新公式 |

### factor_target_cumsum.csv（因子 × target 累加）

信号诊断，不是扣费后的策略收益。`target` 是生产训练标签 `log(open[t+2]/open[t+1])`。

| 列 | 类型 | 含义 |
| --- | --- | --- |
| time / time_utc | int / string | bar 开盘时间 |
| symbol | string | 币种或 `ALL` |
| factor_name | string | 公式；`ALL` 行为 `equal_weight_mean` |
| segment | string | `train` 或 `oos` |
| factor | float | 标准化因子。`ALL` 行为空 |
| target | float | 训练标签。`ALL` 行为空 |
| product | float | 单币是 factor×target；`ALL` 是各币 product 的等权平均 |
| product_sum | float | 仅 `ALL` 行：各币 product 之和 |
| cumsum | float | 从序列起点累加 product |
| oos_cumsum | float | 只在样本外从 0 重新累加；训练段为空 |

### position_target_cumsum.csv（仓位 × target 累加）

同样不含手续费和滑点。扣费后的曲线在 `equity.csv`。

| 列 | 类型 | 含义 |
| --- | --- | --- |
| time / time_utc | int / string | bar 开盘时间 |
| symbol | string | 币种或 `ALL` |
| segment | string | `train` 或 `oos` |
| position | float | 目标仓位。`ALL` 是等权平均 |
| target | float | 训练标签。`ALL` 行为空 |
| product | float | position×target；`ALL` 为等权平均 |
| cumsum / oos_cumsum | float | 与因子诊断相同的两种累加 |

### retrain_timing_history.csv（重训耗时历史）

| 列 | 类型 | 含义 |
| --- | --- | --- |
| kind | string | `benchmark_1core`、`benchmark_multicore`、`walk_forward`、`latest_snapshot` |
| symbol | string | 这一行测的币 |
| decision_unix / decision_utc | int / string | 决策 bar（训练窗最后一根） |
| train_start_unix / train_end_unix | int | 训练窗起止，含端点。train_end 等于 decision |
| n_bars | int | 训练窗根数 |
| cpu_count | int | 机器核数 |
| n_jobs | int | 该行使用的进程数。1core 基准为 1 |
| n_formulas | int | 候选公式数 |
| feature_seconds | float | 特征计算 |
| search_seconds | float | 公式搜索与打分 |
| inference_seconds | float | 冻结公式后的因子与仓位 |
| total_seconds | float | 以上三项之和 |
| finished_at_unix / finished_at_utc | int / string | 该行完成时刻 |
| champion | string | 冠军公式名 |
| formula | string | token 名序列 |
| train_score / val_score | float | 训练分 / 验证分 |
| hold_end_unix | int | 仅 walk_forward：该公式一直用到的最后一根 bar |

## 已知局限

- 重写搜索是单次种子采样，不是 Transformer 策略梯度。冠军公式不必与某次原版长训相同。
- 单品种生产路径下，横截面算子退化为恒等（与原版 N=1 一致）。
- `WILLR_14` 在原版里被裁到 [-1, 0]，正常 OHLC 上会变成 0。重写保持这一行为。
- 部分特征先做 200 期 median/MAD。float32 上一个 ulp 的差别会换掉下中位数，少数特征点会和 torch 差到 1e-2 量级。
  同一特征张量上的算子，以及 N=1 的锚点公式因子值，对照测试要求误差在 1e-4 以内。
- N>1 时，横截面 z-score 会把接近 0 的残差放大。生产推理是 N=1，不走这条路径。
- 训练打分里的换手质量项对连续仓位几乎恒为 -2（原版 `int(tanh)` 把 (-1,1) 截成 0），不改变排序。
- 没有连接实盘、没有交易密钥、没有合并 main。

## 与原版重训的耗时对比

同一批 32 条公式、1500 根合成 1 分钟 bar，重写版对原版 StackVM（`MT5FeatureEngineer` + `StackVM.execute` + 一折 `evaluate_fold`）。原版多核是 PyTorch intra-op 线程，不是进程池（`ModelConfig.PARALLEL_EVAL` 在仓库里是关闭的）。重写版多核是公式级多进程，每个进程 1 个 BLAS 线程。

| 实现 | 核/进程 | 特征秒 | 搜索或评估秒 | 合计秒 |
| --- | --- | --- | --- | --- |
| 重写 | 1 进程 | 0.081 | 0.081 | 0.162 |
| 重写 | 4 进程 | 0.080 | 0.040 | 0.120 |
| 原版 StackVM | 1 线程 | 0.267 | 0.252 | 0.519 |
| 原版 StackVM | 4 线程 | 0.106 | 0.162 | 0.268 |

重写版 1 进程相对原版 1 线程约 3.21 倍。重写版 4 进程相对自身 1 进程约 1.34 倍（这段数据上特征占了一半，搜索加速约 2 倍）。原版 4 线程相对 1 线程约 1.93 倍。

原版 `AlphaEngine.train` 一步（batch 8，T=480，临时目录，不写仓库策略）：1 线程 0.303 秒，4 线程 0.236 秒。按线性外推到生产配置 `TRAIN_STEPS=9000`、`BATCH_SIZE=192`，约 65459 秒（18 小时）才完成一个品种，而且真实 H1 历史更长。这个预算不能每分钟或每几分钟重跑。重写版把单次重训压到 0.5 秒以内，60 秒的 1 分钟 bar 足够滚动。

冠军公式在 1 进程和 4 进程上相同，验证分都是 3.6407。

## 过去一天逐分钟回测

回测段是截至最新已闭合 1 分钟 bar 的最近 1440 根结算 bar。
每一分钟的目标仓位只在前一根收盘时重训：训练窗是此前 4320 根已闭合 bar，候选公式由固定种子生成。
结算收益是随后 1 分钟的简单收益 `close[t+1]/close[t]-1`。
挖掘器内部的打分标签仍是生产定义 `log(open[t+2]/open[t+1])`，这个标签不参与结算。
逐分钟全串行大约要一小时，所以按币种多进程跑，每个进程里的单次重训是单线程。

数据截止 2026-11-09T03:58:00Z（下载时刻能看到的最新已闭合 bar）。进程数 4，机器核数 4。
7 个币 × 1440 分钟 = 10080 次重训。单次中位数 0.347 秒，P95 0.360 秒，最大 0.439 秒。
全串行估计 3502 秒（约 58 分钟）。4 个进程按币并行的墙钟是 1000 秒，加速比 3.50。
单次重训远小于 60 秒；摊到每个结算分钟，7 个币一起约 0.69 秒，赶得上 1 分钟 bar。

收益是复利期末权益减 1。Sharpe 用 1 分钟收益年化（与时间戳估计的每年 bar 数一致）。
MDD 是复利权益的峰值回撤。换手是 `|Δ仓位|` 的均值。买入持有从空仓进入满仓，1bp 版本在第一分钟扣一次滑点。零仓位权益恒为 1。

| 币 | 收益 1bp | Sharpe 1bp | MDD 1bp | 换手 | 收益 0 成本 | Sharpe 0 成本 | MDD 0 成本 | 买入持有收益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | -1.2484% | -10.48 | 2.471% | 0.0811 | -0.0891% | -0.54 | 2.425% | -1.3184% |
| ETH | -0.2070% | -0.81 | 2.863% | 0.0919 | 1.1230% | 6.51 | 2.374% | -4.3327% |
| SOL | -1.1265% | -6.80 | 2.791% | 0.0670 | -0.1687% | -0.76 | 2.572% | -1.9833% |
| XRP | -0.8090% | -5.45 | 2.501% | 0.0610 | 0.0658% | 0.72 | 2.023% | -4.2745% |
| ADA | -5.7245% | -33.40 | 8.239% | 0.0686 | -4.7890% | -27.79 | 7.753% | -2.3111% |
| DOGE | -0.0893% | -0.30 | 4.125% | 0.0765 | 1.0164% | 6.87 | 3.851% | -2.3354% |
| LTC | 2.1731% | 13.61 | 2.165% | 0.0407 | 2.7737% | 17.24 | 2.164% | -2.2733% |
| ALL | -0.9919% | -15.77 | 1.772% | 0.0695 | 0.0042% | 0.18 | 1.324% | -2.6698% |

因子 × target、仓位 × target 的 target 是上面的简单收益，累加不含手续费。那是信号诊断，不是扣滑点后的策略收益。
策略权益在 `equity.csv` / `equity.png`。
