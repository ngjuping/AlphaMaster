"""Runtime configuration for the 1-minute rewrite.

Production defaults this mirrors (see ``model_core/config.py``, ``config.py``,
``train_file.py``):

- single-symbol mining (Config comment, 2026-07-07), not the deprecated groups
- reward mode ``ftmo`` (set by ``train_file.py``)
- formula length 8, factor-pool top-25, correlation 0.85 / penalty 0.8
- walk-forward gap 20, 5 folds
- position = tanh(standardized factor), |position| < 0.05 forced to 0
- training turnover cost = Config.COST_RATE = 0.0003
"""

from __future__ import annotations

# Deterministic seed for sampling. Evaluation order does not depend on workers.
SEED = 20261109

# JEV-style Bitfinex USDt perpetuals. Production live book is H1 XAUUSD + US
# indices; these pairs are the crypto universe requested for the 1-minute run.
PAIRS = {
    "BTC": "tBTCF0:USTF0",
    "ETH": "tETHF0:USTF0",
    "SOL": "tSOLF0:USTF0",
    "XRP": "tXRPF0:USTF0",
    "ADA": "tADAF0:USTF0",
    "DOGE": "tDOGEF0:USTF0",
    "LTC": "tLTCF0:USTF0",
}

# ~6 days of closed 1-minute bars. Long enough for a 3-day train window and a
# multi-day walk-forward, short enough to download from the public candles API.
BARS = 6 * 24 * 60
# Trailing window used by each retrain (includes the decision bar).
TRAIN_WINDOW = 3 * 24 * 60
# First bar index fraction where walk-forward OOS positions begin.
OOS_FRACTION = 0.60
# Default cadence. ``run.py`` may widen this only so the full walk-forward
# finishes; the per-retrain timing is always reported against a 60-second bar.
RETRAIN_EVERY = 60

N_FORMULAS = 32
MAX_FORMULA_LEN = 8
N_FOLDS = 5
WF_GAP = 20

FACTOR_TOP_K = 25
CORR_THRESHOLD = 0.85
CORR_PENALTY = 0.8
MIN_TRADE_EXPOSURE = 0.05

# Production training cost: 0.02% commission + 0.01% slippage.
TRAIN_COST_RATE = 0.0003
# User-facing backtests.
SLIPPAGE_1BP = 0.0001
# Bitfinex public fee page (https://www.bitfinex.com/fees/, fetched 2026-11-09):
# "DERIVATIVES ORDER EXECUTION / Derivatives Trades / Maker fees Zero / Taker Fees Zero".
# Funding is a separate borrow charge and is not part of that order-execution line.
BFX_MAKER_FEE = 0.0
BFX_TAKER_FEE = 0.0

# FTMO single-symbol weights (model_core/backtest.py, REWARD_MODE == "ftmo", N=1).
REWARD_MODE = "ftmo"
IC_GATE_THRESH = 0.01
IC_GATE_MULT = 1.15
IC_NEG_MULT = 0.75
REWARD_ALPHA = 1.0

# Published strategy README formulas, addressed by token name (ids drifted).
ANCHOR_FORMULAS = (
    (
        "forex_v1",
        (
            "AROON_OSC_25",
            "TS_MIN_10",
            "EMA_20",
            "CS_SCALE",
            "SIGN",
            "CS_SCALE",
            "EMA_5",
            "MOMENTUM_5",
        ),
    ),
    (
        "forex_v2",
        (
            "DMI_DIFF_14",
            "HL_RANGE",
            "TS_MIN_20",
            "SUB",
            "TS_MIN_20",
            "TS_MEAN_20",
            "CS_SCALE",
            "WMA",
        ),
    ),
    (
        "metals_comm_v1",
        (
            "CS_ZSCORE_RET20",
            "EMA_20",
            "TS_MAX_10",
            "TANH_SQUASH",
            "EMA_20",
            "AMIHUD_ILLIQ",
            "MAX",
            "MOMENTUM_5",
        ),
    ),
)

SECONDS_PER_YEAR = 365.25 * 86400.0
H1_PERIODS_FALLBACK = 6240
