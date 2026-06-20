from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step5_outputs import make_step5_input_chain


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step5_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step5.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_good_step5(tmp_path: Path) -> tuple[object, Path, Path, Path, Path, Path]:
    build_module = load_module("build_step5_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step5", VALIDATE_MODULE_PATH)
    step2_dir, step3_dir, step4_dir, _ = make_step5_input_chain(tmp_path)
    experiment_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step5_fixture"
    output_dir = experiment_dir / "outputs" / "step5"
    model_dir = experiment_dir / "models" / "step5"
    build_module.build_step5_outputs(
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
    return validate_module, step2_dir, step3_dir, step4_dir, output_dir, model_dir


def test_validate_step5_accepts_good_outputs(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)

    metrics = module.validate_step5(
        step2_dir,
        step3_dir,
        step4_dir,
        output_dir,
        model_dir,
        candidate_size=3,
        random_seed=7,
    )

    assert metrics["input_step2_experiment"] == "exp_step2_fixture"
    assert metrics["output_candidate_rows"] == 3
    assert metrics["output_walk_forward_rounds_used"] == 8
    assert metrics["output_feature_count"] == 2


def test_validate_step5_rejects_failed_step4_report(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    (step4_dir / "notes" / "step4_run_report.md").write_text(
        """# Step-4 report

## Status

FAILED
""",
        encoding="utf-8",
    )

    with pytest.raises(module.Step5ValidationError, match="Step-4 report"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)


def test_validate_step5_rejects_forbidden_feature(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    path = output_dir / "step5_feature_set_used.csv"
    feature_set = pd.read_csv(path)
    bad = feature_set.iloc[0].copy()
    bad["feature_name"] = "label_ret_5d_open_to_open"
    feature_set = pd.concat([feature_set, pd.DataFrame([bad])], ignore_index=True)
    feature_set.to_csv(path, index=False)

    with pytest.raises(module.Step5ValidationError, match="forbidden"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)


def test_validate_step5_rejects_duplicate_candidate(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    path = output_dir / "step5_candidate_top30.csv"
    candidate = pd.read_csv(path, dtype={"股票代码": str})
    candidate.loc[1, "股票代码"] = candidate.loc[0, "股票代码"]
    candidate.to_csv(path, index=False)

    with pytest.raises(module.Step5ValidationError, match="duplicate"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)


def test_validate_step5_rejects_candidate_weight_column(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    path = output_dir / "step5_candidate_top30.csv"
    candidate = pd.read_csv(path, dtype={"股票代码": str})
    candidate["weight"] = 0.2
    candidate.to_csv(path, index=False)

    with pytest.raises(module.Step5ValidationError, match="columns mismatch"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)


def test_validate_step5_rejects_non_pass_leakage_check(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    path = output_dir / "step5_leakage_check.csv"
    leakage = pd.read_csv(path)
    leakage.loc[0, "状态"] = "FAIL"
    leakage.to_csv(path, index=False)

    with pytest.raises(module.Step5ValidationError, match="non-PASS"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)


def test_validate_step5_rejects_missing_model_artifact(tmp_path: Path) -> None:
    module, step2_dir, step3_dir, step4_dir, output_dir, model_dir = build_good_step5(tmp_path)
    path = output_dir / "step5_model_registry.csv"
    registry = pd.read_csv(path)
    artifact = Path(str(registry.loc[0, "model_artifact_path"]))
    artifact.unlink()

    with pytest.raises(module.Step5ValidationError, match="model artifacts missing"):
        module.validate_step5(step2_dir, step3_dir, step4_dir, output_dir, model_dir, candidate_size=3, random_seed=7)
