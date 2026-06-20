from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step1_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_step1_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_step1_outputs_generates_workflow_csvs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs" / "step1"
    raw_dir.mkdir()

    write_csv(
        raw_dir / "hs300_stocks.csv",
        [
            {"updateDate": "2026-06-15", "code": "sh.600000", "code_name": "浦发银行"},
            {"updateDate": "2026-06-15", "code": "sz.000001", "code_name": "平安银行"},
            {"updateDate": "2026-06-15", "code": "sz.000657", "code_name": "中钨高新"},
        ],
    )
    write_csv(
        raw_dir / "stock_industry.csv",
        [
            {
                "updateDate": "2026-06-01",
                "code": "sh.600000",
                "code_name": "浦发银行",
                "industry": "J66货币金融服务",
                "industryClassification": "证监会行业分类",
            },
            {
                "updateDate": "2026-06-01",
                "code": "sz.000001",
                "code_name": "平安银行",
                "industry": "J66货币金融服务",
                "industryClassification": "证监会行业分类",
            },
            {
                "updateDate": "2026-06-01",
                "code": "sz.000657",
                "code_name": "中钨高新",
                "industry": "C33金属制品业",
                "industryClassification": "证监会行业分类",
            },
        ],
    )

    daily_rows = []
    for code, start in [("sh.600000", 10.0), ("sz.000001", 20.0), ("sz.000657", 30.0)]:
        for day in range(1, 7):
            close = start + day
            daily_rows.append(
                {
                    "date": f"2026-06-0{day}",
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
    write_csv(raw_dir / "daily_price_volume.csv", daily_rows)

    module = load_module()
    module.build_step1_outputs(raw_dir=raw_dir, output_dir=output_dir)

    daily = pd.read_csv(output_dir / "step1_daily_raw_data.csv", dtype={"股票代码": str})
    stock = pd.read_csv(output_dir / "step1_stock_summary.csv", dtype={"股票代码": str})
    sector = pd.read_csv(output_dir / "step1_sector_summary.csv")
    manifest = pd.read_csv(output_dir / "step1_data_manifest.csv")

    assert list(daily.columns) == module.STEP1_DAILY_COLUMNS
    assert list(stock.columns) == module.STEP1_STOCK_COLUMNS
    assert list(sector.columns) == module.STEP1_SECTOR_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]

    assert len(daily) == 18
    assert set(stock["股票代码"]) == {"600000", "000001", "000657"}
    assert set(stock["板块划分"]) == {"金融地产", "制造"}
    sector_counts = dict(zip(sector["板块划分"], sector["股票数量"], strict=False))
    assert sector_counts == {"制造": 1, "金融地产": 2}
    assert "workflow_0.1_csv_v1" in set(manifest["说明"])
