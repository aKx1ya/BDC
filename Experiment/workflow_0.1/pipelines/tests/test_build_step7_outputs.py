from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step7_outputs.py"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_SCRIPT = PROJECT_ROOT / "THU-BDC2026-main" / "test" / "score_self.py"


def load_module(name: str = "build_step7_outputs", path: Path = MODULE_PATH):
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


def make_step7_input_chain(root: Path, *, candidate_date: str = "2026-06-15") -> Path:
    workflow_root = root / "Experiment" / "workflow_0.1"
    step6_dir = workflow_root / "experiments" / "exp_step6_fixture"
    step6_output = step6_dir / "outputs" / "step6"
    result_rows = [
        {"stock_id": "000001", "weight": 0.6},
        {"stock_id": "000002", "weight": 0.4},
    ]
    final_rows = [
        {
            "trade_date": candidate_date,
            "股票代码": "000001",
            "股票名称": "平安银行",
            "板块划分": "金融地产",
            "final_rank": 1,
            "weight": 0.6,
            "refine_score": 0.8,
            "model_rank": 1,
            "selection_reason": "fixture",
        },
        {
            "trade_date": candidate_date,
            "股票代码": "000002",
            "股票名称": "万科A",
            "板块划分": "金融地产",
            "final_rank": 2,
            "weight": 0.4,
            "refine_score": 0.7,
            "model_rank": 2,
            "selection_reason": "fixture",
        },
    ]
    write_csv(step6_output / "step6_result.csv", result_rows, ["stock_id", "weight"])
    write_csv(step6_output / "step6_final_top5.csv", final_rows)
    write_csv(step6_output / "step6_ranking_log.csv", [{"股票代码": "000001"}, {"股票代码": "000002"}])
    write_csv(
        step6_output / "step6_refine_manifest.csv",
        [
            {"项目": "schema_version", "说明": "workflow_0.1_csv_v1"},
            {"项目": "input_candidate_date", "说明": candidate_date},
        ],
        ["项目", "说明"],
    )
    write_csv(
        step6_output / "step6_leakage_check.csv",
        [{"检查项": "fixture", "状态": "PASS", "说明": "ok"}],
        ["检查项", "状态", "说明"],
    )
    write_success_report(step6_dir, "step6")
    return step6_dir


def make_test_data(path: Path, *, start: str = "2026-06-16") -> Path:
    dates = pd.bdate_range(start, periods=5).strftime("%Y-%m-%d").tolist()
    rows = []
    for code, base in [("000001", 10.0), ("000002", 20.0)]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "开盘": base + idx,
                    "收盘": base + idx + 0.5,
                }
            )
    write_csv(path, rows)
    return path


def test_build_step7_outputs_freeze_only(tmp_path: Path) -> None:
    module = load_module()
    step6_dir = make_step7_input_chain(tmp_path)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step7_fixture" / "outputs" / "step7"
    workspace_dir = tmp_path / "workspace"

    outputs = module.build_step7_outputs(
        step6_output_dir=step6_dir / "outputs" / "step6",
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        mode="freeze-only",
        input_step6_experiment=step6_dir.name,
    )

    assert set(outputs) == {"frozen_result", "score_summary", "stock_contribution", "manifest", "leakage_check"}
    frozen = pd.read_csv(output_dir / "step7_frozen_result.csv", dtype={"stock_id": str})
    summary = pd.read_csv(output_dir / "step7_score_summary.csv")
    contribution = pd.read_csv(output_dir / "step7_stock_contribution.csv")
    leakage = pd.read_csv(output_dir / "step7_leakage_check.csv")

    assert list(frozen.columns) == module.STEP7_FROZEN_RESULT_COLUMNS
    assert list(summary.columns) == module.STEP7_SCORE_SUMMARY_COLUMNS
    assert list(contribution.columns) == module.STEP7_STOCK_CONTRIBUTION_COLUMNS
    assert set(leakage["状态"]) == {"PASS"}
    assert summary["score_mode"].iloc[0] == "freeze-only"
    assert summary["result_status"].iloc[0] == "FREEZE_ONLY_SUCCESS"
    assert contribution.empty


def test_build_step7_outputs_local_score(tmp_path: Path) -> None:
    module = load_module()
    step6_dir = make_step7_input_chain(tmp_path)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step7_fixture" / "outputs" / "step7"
    workspace_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step7_fixture" / "official_scoring_workspace"
    test_path = make_test_data(tmp_path / "test.csv", start="2026-06-16")

    module.build_step7_outputs(
        step6_output_dir=step6_dir / "outputs" / "step6",
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        mode="local-score",
        input_step6_experiment=step6_dir.name,
        official_script_path=OFFICIAL_SCRIPT,
        test_data_path=test_path,
    )

    summary = pd.read_csv(output_dir / "step7_score_summary.csv")
    contribution = pd.read_csv(output_dir / "step7_stock_contribution.csv", dtype={"stock_id": str})
    leakage = pd.read_csv(output_dir / "step7_leakage_check.csv")

    assert summary["score_mode"].iloc[0] == "local-score"
    assert summary["result_status"].iloc[0] == "SCORE_SUCCESS"
    assert float(summary["final_score"].iloc[0]) != -999
    assert len(contribution) == 2
    assert abs(contribution["score_contribution"].sum() - float(summary["final_score"].iloc[0])) < 1e-8
    assert set(leakage["状态"]) == {"PASS"}
    assert (workspace_dir / "temp" / "tmp.csv").exists()


def test_build_step7_outputs_local_score_marks_nonfuture_test_failed(tmp_path: Path) -> None:
    module = load_module()
    step6_dir = make_step7_input_chain(tmp_path, candidate_date="2026-06-15")
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step7_fixture" / "outputs" / "step7"
    workspace_dir = tmp_path / "workspace"
    test_path = make_test_data(tmp_path / "test.csv", start="2026-06-01")

    module.build_step7_outputs(
        step6_output_dir=step6_dir / "outputs" / "step6",
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        mode="local-score",
        input_step6_experiment=step6_dir.name,
        official_script_path=OFFICIAL_SCRIPT,
        test_data_path=test_path,
    )

    summary = pd.read_csv(output_dir / "step7_score_summary.csv")
    leakage = pd.read_csv(output_dir / "step7_leakage_check.csv")
    assert summary["result_status"].iloc[0] == "FAILED"
    assert "FAIL" in set(leakage["状态"])
    failed_checks = set(leakage.loc[leakage["状态"].eq("FAIL"), "检查项"])
    assert "test_data_is_future_of_candidate_date" in failed_checks
