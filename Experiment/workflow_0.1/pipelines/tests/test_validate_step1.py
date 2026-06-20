from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PIPELINES_DIR = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PIPELINES_DIR / "validate_step1.py"
BUILDER_PATH = PIPELINES_DIR / "build_step1_outputs.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def make_raw_fixture(raw_dir: Path, duplicate_daily: bool = False, missing_daily_code: str | None = None) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stocks = [
        {"updateDate": "2026-06-15", "code": "sh.600000", "code_name": "浦发银行"},
        {"updateDate": "2026-06-15", "code": "sz.000001", "code_name": "平安银行"},
        {"updateDate": "2026-06-15", "code": "sz.000657", "code_name": "中钨高新"},
    ]
    write_csv(raw_dir / "hs300_stocks.csv", stocks)
    write_csv(
        raw_dir / "stock_industry.csv",
        [
            {
                "updateDate": "2026-06-01",
                "code": row["code"],
                "code_name": row["code_name"],
                "industry": "J66货币金融服务" if row["code"] != "sz.000657" else "C33金属制品业",
                "industryClassification": "证监会行业分类",
            }
            for row in stocks
        ],
    )

    daily_rows = []
    for code, start in [("sh.600000", 10.0), ("sz.000001", 20.0), ("sz.000657", 30.0)]:
        if code == missing_daily_code:
            continue
        for day in range(13, 16):
            close = start + day
            daily_rows.append(
                {
                    "date": f"2026-06-{day}",
                    "code": code,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "preclose": close - 1,
                    "volume": 1000 * day,
                    "amount": 10000 * day,
                    "turn": day / 10,
                    "tradestatus": 1,
                    "pctChg": 1.0,
                    "peTTM": 1,
                    "pbMRQ": 1,
                    "psTTM": 1,
                    "pcfNcfTTM": 1,
                    "isST": 0,
                }
            )
    if duplicate_daily:
        daily_rows.append(daily_rows[0].copy())
    write_csv(raw_dir / "daily_price_volume.csv", daily_rows)


def test_validate_raw_data_accepts_complete_current_stock_pool(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_raw_fixture(raw_dir)
    module = load_module(VALIDATOR_PATH, "validate_step1")

    metrics = module.validate_raw_data(raw_dir, expected_stock_count=3)

    assert metrics["hs300_count"] == 3
    assert metrics["daily_current_code_count"] == 3
    assert metrics["daily_latest_T"] == "2026-06-15"
    assert metrics["daily_duplicates"] == 0


def test_validate_raw_data_rejects_missing_current_stock_daily_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_raw_fixture(raw_dir, missing_daily_code="sz.000657")
    module = load_module(VALIDATOR_PATH, "validate_step1")

    with pytest.raises(module.Step1ValidationError, match="missing daily data"):
        module.validate_raw_data(raw_dir, expected_stock_count=3)


def test_validate_raw_data_rejects_duplicate_stock_date_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_raw_fixture(raw_dir, duplicate_daily=True)
    module = load_module(VALIDATOR_PATH, "validate_step1")

    with pytest.raises(module.Step1ValidationError, match="duplicate"):
        module.validate_raw_data(raw_dir, expected_stock_count=3)


def test_validate_outputs_accepts_generated_step1_csvs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs" / "step1"
    make_raw_fixture(raw_dir)
    builder = load_module(BUILDER_PATH, "build_step1_outputs")
    module = load_module(VALIDATOR_PATH, "validate_step1")

    builder.build_step1_outputs(raw_dir=raw_dir, output_dir=output_dir, note="test run")
    metrics = module.validate_outputs(output_dir, expected_stock_count=3)

    assert metrics["output_stock_count"] == 3
    assert metrics["output_unmatched_sector_count"] == 0
    assert metrics["output_latest_T"] == "2026-06-15"


def test_validate_outputs_rejects_bad_header_and_unmatched_sector(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs" / "step1"
    make_raw_fixture(raw_dir)
    builder = load_module(BUILDER_PATH, "build_step1_outputs")
    module = load_module(VALIDATOR_PATH, "validate_step1")

    builder.build_step1_outputs(raw_dir=raw_dir, output_dir=output_dir, note="test run")
    daily_path = output_dir / "step1_daily_raw_data.csv"
    daily = pd.read_csv(daily_path, dtype={"股票代码": str})
    daily = daily.rename(columns={"开盘": "错误开盘"})
    daily.to_csv(daily_path, index=False)

    stock_path = output_dir / "step1_stock_summary.csv"
    stock = pd.read_csv(stock_path, dtype={"股票代码": str})
    stock.loc[0, "板块划分"] = "未匹配"
    stock.to_csv(stock_path, index=False)

    with pytest.raises(module.Step1ValidationError) as exc_info:
        module.validate_outputs(output_dir, expected_stock_count=3)

    message = str(exc_info.value)
    assert "step1_daily_raw_data.csv columns mismatch" in message
    assert "unmatched sector" in message
