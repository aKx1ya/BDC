from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step4_outputs import make_step3_output


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step4_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step4.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_successful_step3_experiment(root: Path, days: int = 40, stock_count: int = 3) -> Path:
    experiment_dir = root / "Experiment" / "workflow_0.1" / "experiments" / "exp_step3_fixture"
    make_step3_output(experiment_dir / "outputs" / "step3", days=days, stock_count=stock_count)
    report_path = experiment_dir / "notes" / "step3_run_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """# Step-3 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return experiment_dir


def build_good_step4(tmp_path: Path) -> tuple[object, Path, Path]:
    build_module = load_module("build_step4_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step4", VALIDATE_MODULE_PATH)
    step3_experiment_dir = make_successful_step3_experiment(tmp_path)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step4_fixture" / "outputs" / "step4"
    build_module.build_step4_outputs(
        step3_output_dir=step3_experiment_dir / "outputs" / "step3",
        output_dir=output_dir,
        input_step3_experiment=step3_experiment_dir.name,
        train_window=10,
        gap_days=2,
        eval_days=3,
        walk_forward_step=3,
        train_ratio=0.75,
        final_test_days=4,
    )
    return validate_module, step3_experiment_dir, output_dir


def test_validate_step4_accepts_good_outputs(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)

    metrics = module.validate_step4(
        step3_experiment_dir,
        output_dir,
        train_window=10,
        gap_days=2,
        eval_days=3,
        walk_forward_step=3,
        train_ratio=0.75,
        final_test_days=4,
    )

    assert metrics["input_step3_sample_date_count"] == 40
    assert metrics["output_inner_train_dates"] == 27
    assert metrics["output_gap_dates"] == 2
    assert metrics["output_validation_dates"] == 7
    assert metrics["output_final_test_dates"] == 4
    assert metrics["output_walk_forward_rounds"] == 9


def test_validate_step4_rejects_unsuccessful_step3_report(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    (step3_experiment_dir / "notes" / "step3_run_report.md").write_text(
        """# Step-3 正式健康版运行报告

## Status

FAILED
""",
        encoding="utf-8",
    )

    with pytest.raises(module.Step4ValidationError, match="Step-3 report"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_duplicate_split_date(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_split_detail.csv"
    detail = pd.read_csv(path)
    detail = pd.concat([detail, detail.head(1)], ignore_index=True)
    detail.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="duplicate"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_gap_count_mismatch(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_split_detail.csv"
    detail = pd.read_csv(path)
    first_gap = detail.index[detail["split_role"].eq("gap")][0]
    detail.loc[first_gap, "split_role"] = "validation"
    detail.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="gap date count"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_bad_walk_forward_train_count(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_walk_forward_plan.csv"
    walk_forward = pd.read_csv(path)
    walk_forward.loc[0, "train_date_count"] = 9
    walk_forward.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="train_date_count"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_final_test_train_allowed(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_split_detail.csv"
    detail = pd.read_csv(path)
    final_idx = detail.index[detail["split_role"].eq("final_test")][0]
    detail.loc[final_idx, "is_train_allowed"] = 1
    detail.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="is_train_allowed"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_non_pass_leakage_check(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_leakage_check.csv"
    leakage = pd.read_csv(path)
    leakage.loc[0, "状态"] = "FAIL"
    leakage.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="non-PASS"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )


def test_validate_step4_rejects_missing_manifest_leakage_note(tmp_path: Path) -> None:
    module, step3_experiment_dir, output_dir = build_good_step4(tmp_path)
    path = output_dir / "step4_split_manifest.csv"
    manifest = pd.read_csv(path)
    manifest = manifest[~manifest["项目"].eq("leakage_control_note")]
    manifest.to_csv(path, index=False)

    with pytest.raises(module.Step4ValidationError, match="leakage_control_note"):
        module.validate_step4(
            step3_experiment_dir,
            output_dir,
            train_window=10,
            gap_days=2,
            eval_days=3,
            walk_forward_step=3,
            train_ratio=0.75,
            final_test_days=4,
        )
