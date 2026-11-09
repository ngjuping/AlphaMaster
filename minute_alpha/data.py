"""Closed 1-minute Bitfinex perpetual candles.

Only bars whose start timestamp is at least one minute in the past are kept.
The public payload is ``[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]``.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from minute_alpha.config import BARS, PAIRS

_API = "https://api-pub.bitfinex.com/v2/candles/trade:1m:{pair}/hist"
_CACHE = Path(__file__).resolve().parent / "cache"

# Exchange candle size in seconds. These are Bitfinex trade candles, not
# aggregates built from the 1-minute prints.
TIMEFRAMES = {"1m": 60, "5m": 300, "15m": 900}


def _get(url: str, timeout: float = 30.0) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "minute-alpha/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_pair(pair: str, bars: int, pause: float = 0.35) -> pd.DataFrame:
    """Download newest-first pages and return ascending closed bars."""
    rows: list[list] = []
    end: int | None = None
    now_ms = int(time.time() * 1000)
    while len(rows) < bars + 5:
        url = f"{_API.format(pair=pair)}?limit=10000&sort=-1"
        if end is not None:
            url += f"&end={end}"
        page = _get(url)
        if not page:
            break
        rows.extend(page)
        oldest = int(page[-1][0])
        if end is not None and oldest >= end:
            break
        end = oldest - 1
        if len(page) < 10000:
            break
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"no candles for {pair}")
    frame = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
    frame = frame.drop_duplicates("mts").sort_values("mts")
    closed = frame["mts"] + 60_000 <= now_ms
    frame = frame.loc[closed].tail(bars).reset_index(drop=True)
    frame["time"] = (frame["mts"] // 1000).astype(np.int64)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(np.float64)
    return frame


def align_closed_panel(
    frames: dict[str, pd.DataFrame],
    grid_end: str = "min",
) -> dict[str, pd.DataFrame]:
    """Put every symbol on one 60-second grid without looking ahead.

    Illiquid perpetuals do not print a candle every minute. Each grid minute
    uses the last candle whose open time is at or before that minute. Minutes
    without a new print keep the last close and set volume to 0, so the
    close-to-close simple return is 0 until the next real print.

    ``grid_end="min"`` stops at the earliest symbol's last print.
    ``grid_end="max"`` continues to the newest print in the universe; symbols
    that have already stopped repeat their last close.
    """
    start = max(int(frame["time"].min()) for frame in frames.values())
    lasts = [int(frame["time"].max()) for frame in frames.values()]
    if grid_end == "max":
        end = max(lasts)
    elif grid_end == "min":
        end = min(lasts)
    else:
        raise ValueError(f"grid_end must be 'min' or 'max', got {grid_end!r}")
    if start % 60:
        start += 60 - (start % 60)
    if end <= start:
        raise RuntimeError("no overlapping closed-bar window")
    grid = pd.DataFrame({"time": np.arange(start, end + 1, 60, dtype=np.int64)})
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        src = frame.sort_values("time").drop_duplicates("time").rename(columns={"time": "print_time"})
        merged = pd.merge_asof(grid, src, left_on="time", right_on="print_time", direction="backward")
        held = merged["time"].to_numpy() != merged["print_time"].to_numpy()
        for col in ("open", "high", "low"):
            merged.loc[held, col] = merged.loc[held, "close"]
        merged.loc[held, "volume"] = 0.0
        merged = merged.dropna(subset=["close"])
        aligned[symbol] = merged
    common = None
    for frame in aligned.values():
        stamps = set(frame["time"].tolist())
        common = stamps if common is None else common & stamps
    if not common:
        raise RuntimeError("asof alignment produced no common minutes")
    times = sorted(common)
    return {
        symbol: frame.set_index("time").loc[times].reset_index()
        for symbol, frame in aligned.items()
    }


def load_raw_frames(
    bars: int = BARS,
    cache: bool = True,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return each symbol's own closed prints. Missing minutes are absent, not filled."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    for symbol, pair in PAIRS.items():
        path = _CACHE / f"{symbol}_{bars}.parquet"
        if cache and path.exists() and not refresh:
            frame = pd.read_parquet(path)
        else:
            frame = fetch_pair(pair, bars)
            if cache:
                frame.to_parquet(path, index=False)
        frames[symbol] = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return frames


def load_universe(
    bars: int = BARS,
    cache: bool = True,
    grid_end: str = "min",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load cached or public candles and align them on a closed-bar clock."""
    frames = load_raw_frames(bars=bars, cache=cache, refresh=refresh)
    return align_closed_panel(frames, grid_end=grid_end)


def fetch_candles(pair: str, timeframe: str, bars: int, pause: float = 0.35) -> pd.DataFrame:
    """Download closed Bitfinex trade candles for ``1m``, ``5m`` or ``15m``.

    The public payload is ``[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]``. MTS is
    the open time of that candle. A candle is kept only after it has closed
    (``open + bar <= now``). Nothing here is built by resampling 1-minute bars.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    bar_ms = TIMEFRAMES[timeframe] * 1000
    rows: list[list] = []
    end: int | None = None
    now_ms = int(time.time() * 1000)
    while len(rows) < bars + 5:
        url = (
            f"https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{pair}/hist"
            f"?limit=10000&sort=-1"
        )
        if end is not None:
            url += f"&end={end}"
        page = None
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                page = _get(url)
                break
            except Exception as exc:  # network / rate limit; the payload is public
                last_err = exc
                time.sleep(pause * (attempt + 1))
        if page is None:
            raise RuntimeError(f"no candles for {pair} {timeframe}: {last_err}")
        if not page:
            break
        rows.extend(page)
        oldest = int(page[-1][0])
        if end is not None and oldest >= end:
            break
        end = oldest - 1
        if len(page) < 10000:
            break
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"no candles for {pair} {timeframe}")
    frame = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
    frame = frame.drop_duplicates("mts").sort_values("mts")
    closed = frame["mts"] + bar_ms <= now_ms
    frame = frame.loc[closed].tail(bars).reset_index(drop=True)
    frame["time"] = (frame["mts"] // 1000).astype(np.int64)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(np.float64)
    return frame


def load_timeframe_frames(
    timeframe: str,
    bars: int = 10000,
    cache: bool = True,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Each symbol's own closed prints at ``timeframe``. Gaps are absent, not filled."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    _CACHE.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, pair in PAIRS.items():
        path = _CACHE / f"{symbol}_{timeframe}_{bars}.parquet"
        if cache and path.exists() and not refresh:
            frame = pd.read_parquet(path)
        else:
            frame = fetch_candles(pair, timeframe, bars)
            if cache:
                frame.to_parquet(path, index=False)
        frames[symbol] = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return frames


def to_matrices(aligned: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    symbols = list(aligned)
    fields = {}
    for col in ("open", "high", "low", "close", "volume"):
        fields[col] = np.stack([aligned[s][col].to_numpy(dtype=np.float64) for s in symbols], axis=0)
    fields["time"] = aligned[symbols[0]]["time"].to_numpy(dtype=np.int64)
    fields["symbols"] = np.array(symbols)
    return fields
