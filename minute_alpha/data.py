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


def align_closed_panel(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Put every symbol on one 60-second grid without looking ahead.

    Illiquid perpetuals do not print a candle every minute. Each grid minute
    uses the last candle whose open time is at or before that minute. Minutes
    without a new print keep the last close and set volume to 0, so the
    open-to-open label is 0 until the next real print.
    """
    start = max(int(frame["time"].min()) for frame in frames.values())
    end = min(int(frame["time"].max()) for frame in frames.values())
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


def load_universe(bars: int = BARS, cache: bool = True) -> dict[str, pd.DataFrame]:
    """Load cached or public candles and align them on a closed-bar clock."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    for symbol, pair in PAIRS.items():
        path = _CACHE / f"{symbol}_{bars}.parquet"
        if cache and path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = fetch_pair(pair, bars)
            if cache:
                frame.to_parquet(path, index=False)
        frames[symbol] = frame
    return align_closed_panel(frames)


def to_matrices(aligned: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    symbols = list(aligned)
    fields = {}
    for col in ("open", "high", "low", "close", "volume"):
        fields[col] = np.stack([aligned[s][col].to_numpy(dtype=np.float64) for s in symbols], axis=0)
    fields["time"] = aligned[symbols[0]]["time"].to_numpy(dtype=np.int64)
    fields["symbols"] = np.array(symbols)
    return fields
