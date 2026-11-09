"""Wall-clock stages and process memory."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return float("nan")
    return float("nan")


def machine() -> dict:
    mem_total = mem_avail = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            info = {}
            for line in handle:
                key, val = line.split(":", 1)
                info[key] = int(val.strip().split()[0]) / 1024.0
        mem_total = info.get("MemTotal")
        mem_avail = info.get("MemAvailable")
    except OSError:
        pass
    return {
        "cpu_count": os.cpu_count(),
        "rss_mb": rss_mb(),
        "mem_total_mb": mem_total,
        "mem_available_mb": mem_avail,
    }


class StageClock:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.rss: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)
            self.rss[name] = rss_mb()

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds
        self.rss[name] = rss_mb()
