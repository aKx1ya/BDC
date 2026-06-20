from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step6_outputs.py"


CANDIDATES = [
    ("600000", "浦发银行", "金融地产", 1.20),
    ("000001", "平安银行", "金融地产", 1.10),
    ("600010", "包钢股份", "周期", 1.00),
    ("600011", "华能国际", "制造", 0.90),
    ("600030", "中信证券", "金融地产", 0.80),
    ("300001", "特锐德", "科技（TMT）", 0.70),
]


def load_module(name: str = "build_step6_outputs", path: Path = MODULE_PATH):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def write_success_report(experiment_dir: Path, step_slug: str) -> None:
    report = experiment_dir / "notes" / f"{step_slug}_run_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        """# report

## Status

SUCCESS
""",
        encoding="utf-8",
    )


def make_step6_input_chain(root: Path) -> tuple[Path, Path, str]:
    workflow_root = root / "Experiment" / "workflow_0.1"
    step2_dir = workflow_root / "experiments" / "exp_step2_fixture"
    step5_dir = workflow_root / "experiments" / "exp_step5_fixture"
    latest_t = "2026-06-15"

    step2_rows = []
    for idx, (code, name, sector, _) in enumerate(CANDIDATES, start=1):
        step2_rows.append(
            {
                "股票代码": code,
                "日期": latest_t,
                "股票名称": name,
                "板块划分": sector,
                "成交量": 10_000_000 * idx,
                "成交额": 500_000_000 - idx * 10_000_000,
                "换手率": 1.0 + idx / 10,
                "ret_5": 6.0 - idx,
                "trend_slope_5": 0.5 - idx / 100,
                "sector_ret_5": 3.0 + idx / 10,
                "sector_short_score": 60.0 - idx,
                "max_drawdown_20": -5.0 * idx,
                "extreme_drop_20_flag": 1 if idx == 2 else 0,
                "low_liquidity_flag": 1 if code == "600030" else 0,
                "no_trade_or_abnormal_flag": 0,
                "risk_any_flag": 1 if idx in {2, 5} else 0,
            }
        )
    write_csv(step2_dir / "outputs" / "step2" / "step2_feature_table_daily.csv", step2_rows)
    write_csv(
        step2_dir / "outputs" / "step2" / "step2_data_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "feature_set_id", "说明": "feature_set_v1_test"},
            {"项目": "latest_T", "说明": latest_t},
        ],
        ["项目", "说明"],
    )

    candidate_rows = []
    for rank, (code, name, sector, score) in enumerate(CANDIDATES, start=1):
        candidate_rows.append(
            {
                "candidate_date": latest_t,
                "股票代码": code,
                "股票名称": name,
                "板块划分": sector,
                "model_score": score,
                "model_rank": rank,
                "fusion_score": score,
                "fusion_rank": rank,
                "model_source": "baseline_corr_final",
                "fusion_method": "single_model_rank_v1",
                "candidate_size": len(CANDIDATES),
                "generated_at": "2026-06-17 00:00:00",
            }
        )
    write_csv(step5_dir / "outputs" / "step5" / "step5_candidate_top30.csv", candidate_rows)
    write_csv(
        step5_dir / "outputs" / "step5" / "step5_model_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "model_set_id", "说明": "model_set_v1_test"},
            {"项目": "input_step2_experiment", "说明": step2_dir.name},
            {"项目": "candidate_size", "说明": len(CANDIDATES)},
            {"项目": "prediction_date", "说明": latest_t},
        ],
        ["项目", "说明"],
    )
    write_csv(
        step5_dir / "outputs" / "step5" / "step5_leakage_check.csv",
        [{"检查项": "candidate_top30_no_future_labels", "状态": "PASS", "说明": "fixture"}],
        ["检查项", "状态", "说明"],
    )
    write_success_report(step2_dir, "step2")
    write_success_report(step5_dir, "step5")
    return step2_dir, step5_dir, latest_t


def test_build_step6_outputs_generates_refined_result_assets(tmp_path: Path) -> None:
    module = load_module()
    step2_dir, step5_dir, latest_t = make_step6_input_chain(tmp_path)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step6_fixture" / "outputs" / "step6"

    outputs = module.build_step6_outputs(
        step5_output_dir=step5_dir / "outputs" / "step5",
        step2_output_dir=step2_dir / "outputs" / "step2",
        output_dir=output_dir,
        input_step5_experiment=step5_dir.name,
        input_step2_experiment=step2_dir.name,
    )

    assert set(outputs) == {
        "ranking_log",
        "final_top5",
        "result",
        "weight_plan",
        "manifest",
        "leakage_check",
    }

    ranking = pd.read_csv(output_dir / "step6_ranking_log.csv", dtype={"股票代码": str})
    final_top5 = pd.read_csv(output_dir / "step6_final_top5.csv", dtype={"股票代码": str})
    result = pd.read_csv(output_dir / "step6_result.csv", dtype={"stock_id": str})
    weight_plan = pd.read_csv(output_dir / "step6_weight_plan.csv")
    manifest = pd.read_csv(output_dir / "step6_refine_manifest.csv")
    leakage = pd.read_csv(output_dir / "step6_leakage_check.csv")

    assert list(ranking.columns) == module.STEP6_RANKING_LOG_COLUMNS
    assert list(final_top5.columns) == module.STEP6_FINAL_TOP5_COLUMNS
    assert list(result.columns) == module.STEP6_RESULT_COLUMNS
    assert list(weight_plan.columns) == module.STEP6_WEIGHT_PLAN_COLUMNS
    assert list(manifest.columns) == ["项目", "说明"]
    assert list(leakage.columns) == module.STEP6_LEAKAGE_CHECK_COLUMNS

    assert len(ranking) == len(CANDIDATES)
    assert len(result) <= 5
    assert set(result["stock_id"]).issubset({code for code, *_ in CANDIDATES})
    assert float(result["weight"].sum()) <= 1
    assert set(final_top5["股票代码"]) == set(result["stock_id"])
    assert set(leakage["状态"]) == {"PASS"}
    assert set(ranking["candidate_date"]) == {latest_t}
    assert ranking.loc[ranking["股票代码"].eq("600030"), "gate_status"].iloc[0] == "removed"
    assert dict(zip(manifest["项目"], manifest["说明"]))["refine_set_id"] == module.REFINE_SET_ID
