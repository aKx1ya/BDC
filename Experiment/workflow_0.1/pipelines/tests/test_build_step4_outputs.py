from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step4_outputs.py"
STEP3_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step3_outputs.py"


def load_module(name: str = "build_step4_outputs", path: Path = MODULE_PATH):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def load_step3_module():
    spec = importlib.util.spec_from_file_location("build_step3_outputs_for_step4_tests", STEP3_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_step3_output(step3_output_dir: Path, days: int = 40, stock_count: int = 3) -> list[str]:
    step3 = load_step3_module()
    dates = pd.bdate_range("2026-01-01", periods=days).strftime("%Y-%m-%d").tolist()
    stocks = [
        ("600000", "浦发银行"),
        ("000001", "平安银行"),
        ("600010", "包钢股份"),
        ("600011", "华能国际"),
        ("600030", "中信证券"),
    ][:stock_count]

    sample_rows = []
    window_rows = []
    rank_rows = []
    group_rows = []
    for group_id, sample_date in enumerate(dates):
        group_start = len(sample_rows)
        for rank, (code, name) in enumerate(stocks, start=1):
            sample_id = f"{sample_date}_{code}"
            label = (stock_count - rank + 1) / 100
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "样本日期T": sample_date,
                    "股票代码": code,
                    "股票名称": name,
                    "板块划分": "金融地产",
                    "原始行业": "测试行业",
                    "window_start": sample_date,
                    "window_end": sample_date,
                    "window_length": 60,
                    "feature_count": 3,
                    "label_open_t1_date": sample_date,
                    "label_open_t5_date": sample_date,
                    "label_open_t1": 10 + rank,
                    "label_open_t5": 10.5 + rank,
                    "label_ret_5d_open_to_open": label,
                    "label_rank_desc": rank,
                    "label_pct_rank": rank / stock_count,
                    "label_top5_flag": 1 if rank <= 5 else 0,
                    "label_top10_flag": 1 if rank <= 10 else 0,
                    "label_top30_flag": 1,
                    "risk_any_flag": 0,
                    "low_liquidity_flag": 0,
                    "no_trade_or_abnormal_flag": 0,
                    "样本可用标记": "是",
                    "样本过滤原因": "",
                }
            )
            window_rows.append(
                {
                    "sample_id": sample_id,
                    "样本日期T": sample_date,
                    "股票代码": code,
                    "window_start": sample_date,
                    "window_end": sample_date,
                    "window_length": 60,
                    "window_row_count": 60,
                    "source_feature_table": "fixture",
                    "window_start_row_number": 0,
                    "window_end_row_number": 59,
                    "窗口完整标记": "是",
                    "窗口过滤原因": "",
                }
            )
            rank_rows.append(
                {
                    "样本日期T": sample_date,
                    "股票代码": code,
                    "股票名称": name,
                    "label_ret_5d_open_to_open": label,
                    "label_rank_desc": rank,
                    "label_pct_rank": rank / stock_count,
                    "label_top5_flag": 1 if rank <= 5 else 0,
                    "label_top10_flag": 1 if rank <= 10 else 0,
                    "label_top30_flag": 1,
                    "label_available_flag": 1,
                    "label_filter_reason": "",
                }
            )
        group_rows.append(
            {
                "样本日期T": sample_date,
                "group_id": group_id,
                "group_start_row": group_start,
                "group_end_row": group_start + stock_count - 1,
                "group_stock_count": stock_count,
                "可用样本数": stock_count,
                "不可用样本数": 0,
                "label_mean": 0.01,
                "label_std": 0.001,
                "label_min": 0.001,
                "label_max": 0.03,
                "top5_label_mean": 0.02,
                "bottom5_label_mean": 0.001,
            }
        )

    write_csv(step3_output_dir / "step3_sample_table.csv", sample_rows, step3.STEP3_SAMPLE_COLUMNS)
    write_csv(step3_output_dir / "step3_window_index.csv", window_rows, step3.STEP3_WINDOW_INDEX_COLUMNS)
    write_csv(step3_output_dir / "step3_group_info.csv", group_rows, step3.STEP3_GROUP_COLUMNS)
    write_csv(step3_output_dir / "step3_rank_label_table.csv", rank_rows, step3.STEP3_RANK_LABEL_COLUMNS)
    write_csv(
        step3_output_dir / "step3_sample_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "sample_set_id", "说明": "sample_set_v1_60d_5d_open_to_open"},
            {"项目": "sample_date_start", "说明": dates[0]},
            {"项目": "sample_date_end", "说明": dates[-1]},
            {"项目": "sample_date_count", "说明": len(dates)},
            {"项目": "sample_row_count", "说明": len(sample_rows)},
            {"项目": "leakage_control_note", "说明": "fixture"},
        ],
        ["项目", "说明"],
    )
    write_csv(
        step3_output_dir / "step3_label_distribution.csv",
        [],
        step3.STEP3_LABEL_DISTRIBUTION_COLUMNS,
    )
    write_csv(
        step3_output_dir / "step3_sample_quality_summary.csv",
        [{"项目": "sample_date_count", "说明": len(dates)}],
        step3.STEP3_QUALITY_COLUMNS,
    )
    return dates


def test_build_step4_outputs_generates_split_assets(tmp_path: Path) -> None:
    module = load_module()
    step3_output_dir = tmp_path / "step3_exp" / "outputs" / "step3"
    output_dir = tmp_path / "step4_exp" / "outputs" / "step4"
    dates = make_step3_output(step3_output_dir, days=40, stock_count=3)

    outputs = module.build_step4_outputs(
        step3_output_dir=step3_output_dir,
        output_dir=output_dir,
        input_step3_experiment="step3_exp",
        train_window=10,
        gap_days=2,
        eval_days=3,
        walk_forward_step=3,
        train_ratio=0.75,
        final_test_days=4,
    )

    assert set(outputs) == {
        "split_detail",
        "split_summary",
        "walk_forward",
        "final_retrain",
        "manifest",
        "leakage_check",
    }
    detail = pd.read_csv(output_dir / "step4_split_detail.csv")
    summary = pd.read_csv(output_dir / "step4_split_summary.csv")
    walk_forward = pd.read_csv(output_dir / "step4_walk_forward_plan.csv")
    final_retrain = pd.read_csv(output_dir / "step4_final_retrain_plan.csv")
    manifest = pd.read_csv(output_dir / "step4_split_manifest.csv")
    leakage = pd.read_csv(output_dir / "step4_leakage_check.csv")

    assert list(detail.columns) == module.STEP4_SPLIT_DETAIL_COLUMNS
    assert list(summary.columns) == module.STEP4_SPLIT_SUMMARY_COLUMNS
    assert list(walk_forward.columns) == module.STEP4_WALK_FORWARD_COLUMNS
    assert list(final_retrain.columns) == module.STEP4_FINAL_RETRAIN_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]
    assert list(leakage.columns) == module.STEP4_LEAKAGE_CHECK_COLUMNS

    assert detail["split_role"].value_counts().to_dict() == {
        "inner_train": 27,
        "validation": 7,
        "final_test": 4,
        "gap": 2,
    }
    assert detail.tail(4)["样本日期T"].astype(str).tolist() == dates[-4:]
    assert set(detail.tail(4)["split_role"]) == {"final_test"}
    assert len(walk_forward) == 9
    assert set(final_retrain.loc[final_retrain["source_split_role"].eq("final_test"), "final_retrain_allowed"]) == {0}
    assert manifest.loc[manifest["项目"].eq("split_set_id"), "说明"].iloc[0] == module.SPLIT_SET_ID
    assert set(leakage["状态"]) == {"PASS"}
