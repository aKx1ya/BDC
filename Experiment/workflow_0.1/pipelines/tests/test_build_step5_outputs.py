from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from test_build_step4_outputs import make_step3_output, write_csv


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step5_outputs.py"
STEP4_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step4_outputs.py"


STOCKS = [
    ("600000", "浦发银行", 5.0),
    ("000001", "平安银行", 4.0),
    ("600010", "包钢股份", 3.0),
    ("600011", "华能国际", 2.0),
    ("600030", "中信证券", 1.0),
]


def load_module(name: str = "build_step5_outputs", path: Path = MODULE_PATH):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_step4_module():
    spec = importlib.util.spec_from_file_location("build_step4_outputs_for_step5_tests", STEP4_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def append_manifest_item(path: Path, item: str, value: object) -> None:
    manifest = pd.read_csv(path)
    manifest = pd.concat([manifest, pd.DataFrame([{"项目": item, "说明": value}])], ignore_index=True)
    manifest.to_csv(path, index=False)


def make_step2_output(step2_output_dir: Path, sample_dates: list[str], latest_t: str) -> None:
    rows = []
    all_dates = [*sample_dates, latest_t]
    for date_idx, date in enumerate(all_dates):
        for code, name, strength in STOCKS:
            rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "股票名称": name,
                    "原始行业": "测试行业",
                    "行业分类口径": "测试口径",
                    "板块划分": "金融地产",
                    "ret_5": strength + date_idx * 0.01,
                    "amount_ratio_5_20": 1.0 + strength / 10,
                    "risk_any_flag": 0,
                }
            )
    write_csv(step2_output_dir / "step2_feature_table_daily.csv", rows)
    write_csv(
        step2_output_dir / "step2_feature_metadata.csv",
        [
            {
                "特征名": "ret_5",
                "特征来源": "fixture",
                "计算窗口": "5日",
                "是否用于模型": "是",
                "是否用于精排": "是",
                "防泄漏说明": "只使用T及以前数据。",
            },
            {
                "特征名": "amount_ratio_5_20",
                "特征来源": "fixture",
                "计算窗口": "20日",
                "是否用于模型": "是",
                "是否用于精排": "是",
                "防泄漏说明": "只使用T及以前数据。",
            },
            {
                "特征名": "label_ret_5d_open_to_open",
                "特征来源": "forbidden",
                "计算窗口": "future",
                "是否用于模型": "是",
                "是否用于精排": "否",
                "防泄漏说明": "测试禁用字段。",
            },
        ],
    )
    write_csv(
        step2_output_dir / "step2_data_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "feature_set_id", "说明": "feature_set_v1_momentum_volume_risk"},
            {"项目": "latest_T", "说明": latest_t},
            {"项目": "date_start", "说明": all_dates[0]},
            {"项目": "date_end", "说明": latest_t},
        ],
        ["项目", "说明"],
    )


def make_step5_input_chain(root: Path) -> tuple[Path, Path, Path, str]:
    workflow_root = root / "Experiment" / "workflow_0.1"
    step2_dir = workflow_root / "experiments" / "exp_step2_fixture"
    step3_dir = workflow_root / "experiments" / "exp_step3_fixture"
    step4_dir = workflow_root / "experiments" / "exp_step4_fixture"

    sample_dates = make_step3_output(step3_dir / "outputs" / "step3", days=40, stock_count=len(STOCKS))
    latest_t = "2026-03-02"
    make_step2_output(step2_dir / "outputs" / "step2", sample_dates, latest_t)
    append_manifest_item(step3_dir / "outputs" / "step3" / "step3_sample_manifest.csv", "input_step2_experiment", step2_dir.name)

    step4_module = load_step4_module()
    step4_module.build_step4_outputs(
        step3_output_dir=step3_dir / "outputs" / "step3",
        output_dir=step4_dir / "outputs" / "step4",
        input_step3_experiment=step3_dir.name,
        train_window=10,
        gap_days=2,
        eval_days=3,
        walk_forward_step=3,
        train_ratio=0.75,
        final_test_days=4,
    )

    for step_dir, step_name in [(step2_dir, "step2"), (step3_dir, "step3"), (step4_dir, "step4")]:
        report = step_dir / "notes" / f"{step_name}_run_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"""# {step_name} report

## Status

SUCCESS
""",
            encoding="utf-8",
        )

    return step2_dir, step3_dir, step4_dir, latest_t


def test_build_step5_outputs_generates_model_assets(tmp_path: Path) -> None:
    module = load_module()
    step2_dir, step3_dir, step4_dir, latest_t = make_step5_input_chain(tmp_path)
    experiment_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step5_fixture"
    output_dir = experiment_dir / "outputs" / "step5"
    model_dir = experiment_dir / "models" / "step5"

    outputs = module.build_step5_outputs(
        step2_output_dir=step2_dir / "outputs" / "step2",
        step3_output_dir=step3_dir / "outputs" / "step3",
        step4_output_dir=step4_dir / "outputs" / "step4",
        output_dir=output_dir,
        model_dir=model_dir,
        input_step2_experiment=step2_dir.name,
        input_step3_experiment=step3_dir.name,
        input_step4_experiment=step4_dir.name,
        candidate_size=3,
        random_seed=7,
    )

    assert set(outputs) == {
        "model_registry",
        "feature_set",
        "walk_forward_predictions",
        "walk_forward_metrics",
        "feature_importance",
        "candidate_top30",
        "manifest",
        "leakage_check",
    }
    registry = pd.read_csv(output_dir / "step5_model_registry.csv")
    feature_set = pd.read_csv(output_dir / "step5_feature_set_used.csv")
    predictions = pd.read_csv(output_dir / "step5_walk_forward_predictions.csv", dtype={"股票代码": str})
    metrics = pd.read_csv(output_dir / "step5_walk_forward_metrics.csv")
    candidate = pd.read_csv(output_dir / "step5_candidate_top30.csv", dtype={"股票代码": str})
    manifest = pd.read_csv(output_dir / "step5_model_manifest.csv")
    leakage = pd.read_csv(output_dir / "step5_leakage_check.csv")

    assert list(registry.columns) == module.STEP5_MODEL_REGISTRY_COLUMNS
    assert list(feature_set.columns) == module.STEP5_FEATURE_SET_COLUMNS
    assert list(predictions.columns) == module.STEP5_WALK_FORWARD_PREDICTIONS_COLUMNS
    assert list(metrics.columns) == module.STEP5_WALK_FORWARD_METRICS_COLUMNS
    assert list(candidate.columns) == module.STEP5_CANDIDATE_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]
    assert list(leakage.columns) == module.STEP5_LEAKAGE_CHECK_COLUMNS

    assert len(candidate) == 3
    assert set(candidate["candidate_date"]) == {latest_t}
    assert "label_ret_5d_open_to_open" not in set(candidate.columns)
    assert set(feature_set["feature_name"]) == {"ret_5", "amount_ratio_5_20"}
    assert len(metrics) == 8
    assert set(leakage["状态"]) == {"PASS"}
    assert all(Path(path).exists() for path in registry["model_artifact_path"].astype(str))
