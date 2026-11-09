"""status.json has the fields the online-learning page binds to."""

from __future__ import annotations

import pandas as pd

from minute_alpha.portal import SCHEMA_VERSION, build_status


def test_status_has_page_fields():
    payload = build_status(
        cpu_count=4,
        n_jobs=4,
        data_cutoff_unix=1_700_000_000,
        positions=[
            {
                "symbol": "BTC",
                "pair": "tBTCF0:USTF0",
                "target_position": 0.2,
                "factor_value": 0.3,
            }
        ],
        last_retrain={"duration_seconds": 1.5, "finished_at_unix": 1_700_000_060},
        timing={"one_core_seconds": 2.0, "multi_core_seconds": 0.8, "speedup": 2.5},
        symbols=["BTC"],
        retrain_every_minutes=60,
        bars=100,
        oos_start_unix=1_699_000_000,
    )
    for key in (
        "page",
        "schema_version",
        "package_version",
        "git_commit",
        "cpu_count",
        "n_jobs",
        "data_cutoff_unix",
        "data_cutoff_utc",
        "last_retrain",
        "positions",
        "timing",
        "files",
    ):
        assert key in payload
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["page"] == "alphamaster_online_learning"
    assert payload["positions"][0]["target_position"] == 0.2
    assert payload["files"]["position_history"] == "position_history.csv"
    assert payload["files"]["factor_target_cumsum"] == "factor_target_cumsum.csv"
    assert payload["files"]["position_target_cumsum"] == "position_target_cumsum.csv"
    assert payload["files"]["retrain_timing_history"] == "retrain_timing_history.csv"
    assert payload["data_cutoff_utc"].endswith("Z")


def test_asof_alignment_does_not_use_future_prints():
    from minute_alpha.data import align_closed_panel

    btc = pd.DataFrame(
        {
            "time": [120, 180, 240],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [5, 5, 5],
        }
    )
    ltc = pd.DataFrame(
        {
            "time": [120, 240],
            "open": [10, 30],
            "high": [10, 30],
            "low": [10, 30],
            "close": [10, 30],
            "volume": [7, 7],
        }
    )
    panel = align_closed_panel({"BTC": btc, "LTC": ltc})
    ltc_out = panel["LTC"].set_index("time")
    # Minute 180 has no LTC print. The close must stay 10, not the later 30.
    assert ltc_out.loc[180, "close"] == 10
    assert ltc_out.loc[180, "volume"] == 0
    assert ltc_out.loc[240, "close"] == 30
