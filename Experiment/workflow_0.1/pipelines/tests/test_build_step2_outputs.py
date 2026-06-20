from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step2_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_step2_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_step1_output(step1_output_dir: Path, days: int = 25) -> None:
    dates = pd.bdate_range("2026-05-01", periods=days).strftime("%Y-%m-%d").tolist()
    stocks = [
        ("600000", "浦发银行", "J66货币金融服务", "金融地产", 10.0),
        ("000001", "平安银行", "J66货币金融服务", "金融地产", 12.0),
        ("600010", "包钢股份", "C31黑色金属冶炼和压延加工业", "周期", 2.0),
    ]

    daily_rows = []
    stock_rows = []
    for code, name, industry, sector, base in stocks:
        for idx, date in enumerate(dates):
            close = base + idx * 0.12 + (0.04 if code == "600010" else 0)
            daily_rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "开盘": close - 0.03,
                    "收盘": close,
                    "最高": close + 0.08,
                    "最低": close - 0.08,
                    "成交量": 1_000_000 + idx * 10_000,
                    "成交额": (120_000_000 if sector == "金融地产" else 80_000_000) + idx * 1_000_000,
                    "振幅": 1.0,
                    "涨跌额": 0.12,
                    "换手率": 0.5,
                    "涨跌幅": 1.0,
                }
            )
        stock_rows.append(
            {
                "股票代码": code,
                "股票名称": name,
                "最新日期": dates[-1],
                "原始行业": industry,
                "行业分类口径": "证监会行业分类",
                "板块划分": sector,
            }
        )

    write_csv(step1_output_dir / "step1_daily_raw_data.csv", daily_rows)
    write_csv(step1_output_dir / "step1_stock_summary.csv", stock_rows)
    write_csv(
        step1_output_dir / "step1_sector_summary.csv",
        [
            {"板块划分": "金融地产", "股票数量": 2},
            {"板块划分": "周期", "股票数量": 1},
        ],
    )
    write_csv(
        step1_output_dir / "step1_data_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "date_start", "说明": dates[0]},
            {"项目": "date_end", "说明": dates[-1]},
            {"项目": "latest_T", "说明": dates[-1]},
            {"项目": "raw_交易日数", "说明": len(dates)},
        ],
    )


def test_build_step2_outputs_generates_standard_csvs(tmp_path: Path) -> None:
    module = load_module()
    step1_output_dir = tmp_path / "step1_exp" / "outputs" / "step1"
    output_dir = tmp_path / "step2_exp" / "outputs" / "step2"
    make_step1_output(step1_output_dir)

    outputs = module.build_step2_outputs(
        step1_output_dir=step1_output_dir,
        output_dir=output_dir,
        input_step1_experiment="step1_exp",
    )

    assert set(outputs) == {"feature", "sector", "latest", "metadata", "manifest", "sector_latest", "risk"}
    feature = pd.read_csv(output_dir / "step2_feature_table_daily.csv", dtype={"股票代码": str})
    sector = pd.read_csv(output_dir / "step2_sector_feature_table.csv")
    latest = pd.read_csv(output_dir / "step2_latest_t_screen.csv", dtype={"股票代码": str})
    metadata = pd.read_csv(output_dir / "step2_feature_metadata.csv")
    manifest = pd.read_csv(output_dir / "step2_data_manifest.csv")
    sector_latest = pd.read_csv(output_dir / "step2_sector_score_latest.csv")
    risk = pd.read_csv(output_dir / "step2_risk_feature_table.csv", dtype={"股票代码": str})

    assert list(feature.columns) == module.STEP2_FEATURE_COLUMNS
    assert list(sector.columns) == module.STEP2_SECTOR_COLUMNS
    assert list(latest.columns) == module.STEP2_LATEST_SCREEN_COLUMNS
    assert list(metadata.columns) == module.STEP2_METADATA_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]
    assert list(sector_latest.columns) == module.STEP2_SECTOR_COLUMNS
    assert list(risk.columns) == module.STEP2_RISK_COLUMNS

    latest_t = manifest.loc[manifest["项目"] == "latest_T", "说明"].iloc[0]
    assert latest_t == "2026-06-04"
    assert set(latest["日期"]) == {latest_t}
    assert len(latest) == 3
    assert len(feature) == 75
    assert set(metadata["特征名"]) >= set(module.GENERATED_FEATURES_FOR_METADATA)
