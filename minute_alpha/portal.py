"""Files consumed by the AlphaMaster online-learning page.

``status.json`` is the snapshot. The CSVs are the histories. Field meanings
are also written into ``REPORT.md`` so the page can bind without reading code.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from minute_alpha import __version__
from minute_alpha.diagnostics import time_utc

SCHEMA_VERSION = "1.0.0"
PAGE_ID = "alphamaster_online_learning"


def git_meta(repo: str = "/workspace") -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def build_status(
    *,
    cpu_count: int,
    n_jobs: int,
    data_cutoff_unix: int,
    positions: list[dict],
    last_retrain: dict,
    timing: dict,
    symbols: list[str],
    retrain_every_minutes: int,
    bars: int,
    oos_start_unix: int,
) -> dict:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    meta = git_meta()
    return {
        "page": PAGE_ID,
        "schema_version": SCHEMA_VERSION,
        "package_version": meta["package_version"],
        "git_commit": meta["git_commit"],
        "git_branch": meta["git_branch"],
        "generated_at_unix": now,
        "generated_at_utc": time_utc(now),
        "cpu_count": int(cpu_count),
        "n_jobs": int(n_jobs),
        "symbols": list(symbols),
        "bars": int(bars),
        "retrain_every_minutes": int(retrain_every_minutes),
        "data_cutoff_unix": int(data_cutoff_unix),
        "data_cutoff_utc": time_utc(data_cutoff_unix),
        "oos_start_unix": int(oos_start_unix),
        "oos_start_utc": time_utc(oos_start_unix),
        "last_retrain": last_retrain,
        "timing": timing,
        "positions": positions,
        "files": {
            "position_history": "position_history.csv",
            "factor_target_cumsum": "factor_target_cumsum.csv",
            "position_target_cumsum": "position_target_cumsum.csv",
            "retrain_timing_history": "retrain_timing_history.csv",
            "equity": "equity.csv",
            "next_minute_positions": "next_minute_positions.csv",
        },
    }


def write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
