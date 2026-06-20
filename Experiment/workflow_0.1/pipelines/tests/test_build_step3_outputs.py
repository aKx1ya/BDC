from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step3_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_step3_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_step2_output(step2_output_dir: Path, days: int = 70, stock_count: int = 3) -> list[str]:
    dates = pd.bdate_range("2026-01-01", periods=days).strftime("%Y-%m-%d").tolist()
    stocks = [
        ("600000", "浦发银行", "金融地产", 10.0),
        ("000001", "平安银行", "金融地产", 12.0),
        ("600010", "包钢股份", "周期", 2.0),
        ("600011", "华能国际", "基建与公用", 8.0),
        ("600030", "中信证券", "金融地产", 20.0),
        ("600519", "贵州茅台", "消费", 100.0),
    ][:stock_count]

    feature_rows = []
    risk_rows = []
    for code, name, sector, base in stocks:
        for idx, date in enumerate(dates):
            open_price = base + idx * (0.05 + len(code) * 0.001)
            row = {
                "股票代码": code,
                "日期": date,
                "股票名称": name,
                "原始行业": "测试行业",
                "行业分类口径": "测试口径",
                "开盘": open_price,
                "收盘": open_price + 0.02,
                "最高": open_price + 0.05,
                "最低": open_price - 0.05,
                "成交量": 1_000_000 + idx,
                "成交额": 100_000_000 + idx * 100_000,
                "振幅": 1.0,
                "涨跌额": 0.02,
                "换手率": 0.5,
                "涨跌幅": 0.8,
                "ret_5": idx / 100,
                "trend_slope_5": idx / 1000,
                "amount_ratio_5_20": 1.0 + idx / 1000,
                "volume_ratio_5_20": 1.0 + idx / 1000,
                "risk_any_flag": 0,
                "low_liquidity_flag": 0,
                "no_trade_or_abnormal_flag": 0,
                "板块划分": sector,
            }
            feature_rows.append(row)
            risk_rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "股票名称": name,
                    "max_drawdown_20": -1,
                    "extreme_drop_20_flag": 0,
                    "low_liquidity_flag": 0,
                    "no_trade_or_abnormal_flag": 0,
                    "risk_any_flag": 0,
                    "板块划分": sector,
                }
            )

    write_csv(step2_output_dir / "step2_feature_table_daily.csv", feature_rows)
    write_csv(step2_output_dir / "step2_sector_feature_table.csv", [{"日期": dates[-1], "板块划分": "金融地产"}])
    write_csv(step2_output_dir / "step2_latest_t_screen.csv", [])
    write_csv(
        step2_output_dir / "step2_feature_metadata.csv",
        [
            {
                "特征名": "ret_5",
                "特征来源": "测试",
                "计算窗口": "5日",
                "是否用于模型": "是",
                "是否用于精排": "是",
                "防泄漏说明": "只使用T及以前数据。",
            }
        ],
    )
    write_csv(
        step2_output_dir / "step2_data_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "feature_set_id", "说明": "feature_set_v1_momentum_volume_risk"},
            {"项目": "date_start", "说明": dates[0]},
            {"项目": "date_end", "说明": dates[-1]},
            {"项目": "latest_T", "说明": dates[-1]},
        ],
    )
    write_csv(step2_output_dir / "step2_sector_score_latest.csv", [{"日期": dates[-1], "板块划分": "金融地产"}])
    write_csv(step2_output_dir / "step2_risk_feature_table.csv", risk_rows)
    return dates


def test_build_step3_outputs_generates_sample_assets(tmp_path: Path) -> None:
    module = load_module()
    step2_output_dir = tmp_path / "step2_exp" / "outputs" / "step2"
    output_dir = tmp_path / "step3_exp" / "outputs" / "step3"
    dates = make_step2_output(step2_output_dir)

    outputs = module.build_step3_outputs(
        step2_output_dir=step2_output_dir,
        output_dir=output_dir,
        input_step2_experiment="step2_exp",
    )

    assert set(outputs) == {"sample", "window", "group", "rank", "manifest", "label_distribution", "quality"}
    sample = pd.read_csv(output_dir / "step3_sample_table.csv", dtype={"股票代码": str})
    window = pd.read_csv(output_dir / "step3_window_index.csv", dtype={"股票代码": str})
    group = pd.read_csv(output_dir / "step3_group_info.csv")
    rank = pd.read_csv(output_dir / "step3_rank_label_table.csv", dtype={"股票代码": str})
    manifest = pd.read_csv(output_dir / "step3_sample_manifest.csv")
    label_distribution = pd.read_csv(output_dir / "step3_label_distribution.csv")
    quality = pd.read_csv(output_dir / "step3_sample_quality_summary.csv")

    assert list(sample.columns) == module.STEP3_SAMPLE_COLUMNS
    assert list(window.columns) == module.STEP3_WINDOW_INDEX_COLUMNS
    assert list(group.columns) == module.STEP3_GROUP_COLUMNS
    assert list(rank.columns) == module.STEP3_RANK_LABEL_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]
    assert list(label_distribution.columns) == module.STEP3_LABEL_DISTRIBUTION_COLUMNS
    assert list(quality.columns) == module.STEP3_QUALITY_COLUMNS

    assert sample["样本日期T"].min() == dates[59]
    assert sample["样本日期T"].max() == dates[-6]
    assert len(sample) == 18
    assert len(group) == 6
    assert set(sample["样本可用标记"]) == {"是"}
    assert set(window["窗口完整标记"]) == {"是"}
    assert manifest.loc[manifest["项目"] == "sample_set_id", "说明"].iloc[0] == module.SAMPLE_SET_ID
