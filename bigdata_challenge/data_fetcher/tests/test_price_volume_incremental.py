from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "01_price_volume.py"


def load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("price_volume", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_per_stock_incremental_start_dates_handle_new_constituents(tmp_path: Path) -> None:
    csv_path = tmp_path / "daily_price_volume.csv"
    pd.DataFrame(
        [
            {"date": "2026-05-29", "code": "sh.600000"},
            {"date": "2026-06-01", "code": "sh.600000"},
            {"date": "2026-05-30", "code": "sz.000001"},
        ]
    ).to_csv(csv_path, index=False)

    module = load_module()
    last_dates = module.get_stock_last_dates(csv_path)

    assert last_dates == {"sh.600000": "2026-06-01", "sz.000001": "2026-05-30"}
    assert module.get_update_start_for_code("sh.600000", last_dates, "2023-01-01") == "2026-06-02"
    assert module.get_update_start_for_code("sz.000657", last_dates, "2023-01-01") == "2023-01-01"


def test_run_with_timeout_interrupts_slow_calls() -> None:
    module = load_module()

    assert module.run_with_timeout(lambda: "ok", timeout_seconds=1) == "ok"

    try:
        module.run_with_timeout(lambda: time.sleep(2), timeout_seconds=1)
    except module.QueryTimeout:
        return

    raise AssertionError("Expected QueryTimeout")


def test_skip_only_codes_already_at_mixed_local_latest() -> None:
    module = load_module()

    mixed_last_dates = {
        "sh.600000": "2026-06-15",
        "sh.688981": "2026-06-01",
        "sz.000657": "2026-06-15",
    }
    uniform_last_dates = {
        "sh.600000": "2026-06-15",
        "sz.000657": "2026-06-15",
    }

    assert module.should_skip_local_latest("sh.600000", mixed_last_dates) is True
    assert module.should_skip_local_latest("sh.688981", mixed_last_dates) is False
    assert module.should_skip_local_latest("sz.000988", mixed_last_dates) is False
    assert module.should_skip_local_latest("sh.600000", uniform_last_dates) is False


def test_empty_update_result_is_not_reported_as_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()

    class FakeBaostock:
        def login(self):
            return object()

        def logout(self):
            return None

    monkeypatch.setattr(module, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(module, "END_DATE", "2026-06-16")
    monkeypatch.setattr(
        module,
        "get_hs300_stocks",
        lambda: pd.DataFrame([{"code": "sh.600000"}]),
    )
    monkeypatch.setattr(
        module,
        "get_stock_daily",
        lambda code, start, end: pd.DataFrame(),
    )
    monkeypatch.setattr(module, "bs", FakeBaostock())

    result = module.fetch_all_stocks()
    output = capsys.readouterr().out

    assert result.empty
    assert "无新增数据" in output
    assert "失败股票" not in output
