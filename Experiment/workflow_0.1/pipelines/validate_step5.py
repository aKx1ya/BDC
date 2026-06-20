#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step3_outputs import SAMPLE_SET_ID  # noqa: E402
from build_step4_outputs import SPLIT_SET_ID  # noqa: E402
from build_step5_outputs import (  # noqa: E402
    DEFAULT_CANDIDATE_SIZE,
    DEFAULT_RANDOM_SEED,
    FEATURE_SET_ID,
    MODEL_SET_ID,
    SCHEMA_VERSION,
    STEP5_CANDIDATE_COLUMNS,
    STEP5_FEATURE_IMPORTANCE_COLUMNS,
    STEP5_FEATURE_SET_COLUMNS,
    STEP5_LEAKAGE_CHECK_COLUMNS,
    STEP5_MODEL_REGISTRY_COLUMNS,
    STEP5_WALK_FORWARD_METRICS_COLUMNS,
    STEP5_WALK_FORWARD_PREDICTIONS_COLUMNS,
    is_forbidden_feature,
)


STEP2_OUTPUT_FILES = {
    "feature": "step2_feature_table_daily.csv",
    "metadata": "step2_feature_metadata.csv",
    "manifest": "step2_data_manifest.csv",
}


STEP3_OUTPUT_FILES = {
    "sample": "step3_sample_table.csv",
    "manifest": "step3_sample_manifest.csv",
}


STEP4_OUTPUT_FILES = {
    "split_detail": "step4_split_detail.csv",
    "walk_forward": "step4_walk_forward_plan.csv",
    "final_retrain": "step4_final_retrain_plan.csv",
    "manifest": "step4_split_manifest.csv",
    "leakage_check": "step4_leakage_check.csv",
}


STEP5_OUTPUT_FILES = {
    "model_registry": "step5_model_registry.csv",
    "feature_set": "step5_feature_set_used.csv",
    "walk_forward_predictions": "step5_walk_forward_predictions.csv",
    "walk_forward_metrics": "step5_walk_forward_metrics.csv",
    "feature_importance": "step5_feature_importance.csv",
    "candidate_top30": "step5_candidate_top30.csv",
    "manifest": "step5_model_manifest.csv",
    "leakage_check": "step5_leakage_check.csv",
}


REQUIRED_STEP5_MANIFEST_ITEMS = {
    "schema_version",
    "model_set_id",
    "feature_set_id",
    "input_step2_path",
    "input_step3_path",
    "input_step4_path",
    "input_step2_experiment",
    "input_step3_experiment",
    "input_step4_experiment",
    "input_step2_latest_T",
    "input_step3_sample_set_id",
    "input_step4_split_set_id",
    "feature_count",
    "model_count",
    "candidate_size",
    "prediction_date",
    "random_seed",
    "training_policy",
    "model_family",
    "fusion_method",
    "walk_forward_rounds_used",
    "generated_at",
    "data_window_note",
    "leakage_control_note",
}


FORBIDDEN_CANDIDATE_COLUMNS = {
    "weight",
    "final_selected",
    "result",
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
}


class Step5ValidationError(Exception):
    """Step-5 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step5ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step5ValidationError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> list[str]:
    actual = list(df.columns)
    if actual != expected:
        return [f"{file_name} columns mismatch: expected {expected}, got {actual}"]
    return []


def manifest_value(manifest: pd.DataFrame, item: str, default: str = "") -> str:
    if {"项目", "说明"} - set(manifest.columns):
        return default
    matched = manifest.loc[manifest["项目"].astype(str) == item, "说明"]
    if matched.empty:
        return default
    return str(matched.iloc[0])


def require_manifest_items(manifest: pd.DataFrame, required: set[str], file_name: str) -> list[str]:
    if list(manifest.columns) != ["项目", "说明"]:
        return [f"{file_name} columns mismatch: expected ['项目', '说明'], got {list(manifest.columns)}"]
    items = set(manifest["项目"].astype(str))
    missing = sorted(required - items)
    if missing:
        return [f"{file_name} missing items: {missing}"]
    return []


def report_is_success(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def expected_eval_dates_without_final_test(walk_forward: pd.DataFrame, split_detail: pd.DataFrame) -> dict[int, set[str]]:
    sample_dates = split_detail["样本日期T"].astype(str).tolist()
    final_test_dates = set(split_detail.loc[split_detail["split_role"].eq("final_test"), "样本日期T"].astype(str))
    out: dict[int, set[str]] = {}
    for _, row in walk_forward.iterrows():
        eval_dates = {date for date in sample_dates if str(row["eval_start"]) <= date <= str(row["eval_end"])}
        if eval_dates & final_test_dates:
            continue
        out[int(row["wf_round"])] = eval_dates
    return out


def validate_inputs(
    step2_experiment_dir: Path,
    step3_experiment_dir: Path,
    step4_experiment_dir: Path,
) -> dict[str, object]:
    step2_experiment_dir = Path(step2_experiment_dir)
    step3_experiment_dir = Path(step3_experiment_dir)
    step4_experiment_dir = Path(step4_experiment_dir)
    step2_output_dir = step2_experiment_dir / "outputs" / "step2"
    step3_output_dir = step3_experiment_dir / "outputs" / "step3"
    step4_output_dir = step4_experiment_dir / "outputs" / "step4"
    errors: list[str] = []

    if not report_is_success(step2_experiment_dir / "notes" / "step2_run_report.md"):
        errors.append(f"Step-2 report is not SUCCESS: {step2_experiment_dir}")
    if not report_is_success(step3_experiment_dir / "notes" / "step3_run_report.md"):
        errors.append(f"Step-3 report is not SUCCESS: {step3_experiment_dir}")
    if not report_is_success(step4_experiment_dir / "notes" / "step4_run_report.md"):
        errors.append(f"Step-4 report is not SUCCESS: {step4_experiment_dir}")

    step2_feature = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str})
    step2_metadata = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["metadata"])
    step2_manifest = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["manifest"])
    step3_sample = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["sample"], dtype={"股票代码": str})
    step3_manifest = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["manifest"])
    step4_split = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["split_detail"])
    step4_walk_forward = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["walk_forward"])
    step4_manifest = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["manifest"])
    step4_leakage = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["leakage_check"])

    if manifest_value(step3_manifest, "input_step2_experiment") != step2_experiment_dir.name:
        errors.append("Step-3 manifest input_step2_experiment does not match selected Step-2")
    if manifest_value(step4_manifest, "input_step3_experiment") != step3_experiment_dir.name:
        errors.append("Step-4 manifest input_step3_experiment does not match selected Step-3")
    if manifest_value(step3_manifest, "sample_set_id") != SAMPLE_SET_ID:
        errors.append(f"Step-3 sample_set_id expected {SAMPLE_SET_ID}")
    if manifest_value(step4_manifest, "split_set_id") != SPLIT_SET_ID:
        errors.append(f"Step-4 split_set_id expected {SPLIT_SET_ID}")

    latest_t = manifest_value(step2_manifest, "latest_T")
    if not latest_t:
        errors.append("Step-2 latest_T is empty")
    elif "日期" in step2_feature.columns and latest_t not in set(step2_feature["日期"].astype(str)):
        errors.append(f"Step-2 feature table does not contain latest_T={latest_t}")

    if {"样本日期T"} <= set(step3_sample.columns) and {"样本日期T"} <= set(step4_split.columns):
        step3_dates = set(step3_sample["样本日期T"].astype(str))
        step4_dates = set(step4_split["样本日期T"].astype(str))
        if step3_dates != step4_dates:
            errors.append("Step-3 sample dates do not match Step-4 split_detail dates")

    if "状态" not in step4_leakage.columns or not step4_leakage["状态"].astype(str).eq("PASS").all():
        errors.append("Step-4 leakage_check must be all PASS")
    if step4_walk_forward.empty:
        errors.append("Step-4 walk_forward_plan is empty")

    if "防泄漏说明" not in step2_metadata.columns:
        errors.append("Step-2 feature_metadata missing 防泄漏说明")
    else:
        model_features = step2_metadata[step2_metadata["是否用于模型"].astype(str).eq("是")]
        if model_features["防泄漏说明"].isna().any() or model_features["防泄漏说明"].astype(str).str.strip().eq("").any():
            errors.append("Step-2 model feature metadata has empty 防泄漏说明")

    raise_if_errors(errors)

    return {
        "input_step2_experiment": step2_experiment_dir.name,
        "input_step3_experiment": step3_experiment_dir.name,
        "input_step4_experiment": step4_experiment_dir.name,
        "input_step2_latest_T": latest_t,
        "input_step2_rows": int(len(step2_feature)),
        "input_step3_sample_rows": int(len(step3_sample)),
        "input_step4_split_dates": int(step4_split["样本日期T"].nunique()),
        "input_step4_walk_forward_rounds": int(len(step4_walk_forward)),
        "input_step4_final_test_dates": int(step4_split["split_role"].astype(str).eq("final_test").sum()),
    }


def validate_feature_set(feature_set: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(feature_set, STEP5_FEATURE_SET_COLUMNS, STEP5_OUTPUT_FILES["feature_set"])
    if errors:
        return errors
    if feature_set.empty:
        errors.append("step5_feature_set_used.csv is empty")
    duplicate_features = int(feature_set.duplicated(["feature_name"]).sum())
    if duplicate_features:
        errors.append(f"step5_feature_set_used.csv duplicate feature_name rows: {duplicate_features}")
    bad_features = [name for name in feature_set["feature_name"].astype(str) if is_forbidden_feature(name)]
    if bad_features:
        errors.append(f"feature_set_used contains forbidden features: {bad_features}")
    if not pd.to_numeric(feature_set["used_for_model"], errors="coerce").fillna(0).astype(int).eq(1).all():
        errors.append("feature_set_used used_for_model must be 1 for all rows")
    if not feature_set["fit_scope"].astype(str).eq("train_only").all():
        errors.append("feature_set_used fit_scope must be train_only")
    if feature_set["leakage_guard_note"].isna().any() or feature_set["leakage_guard_note"].astype(str).str.strip().eq("").any():
        errors.append("feature_set_used leakage_guard_note must be non-empty")
    return errors


def validate_model_registry(registry: pd.DataFrame, model_dir: Path) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(registry, STEP5_MODEL_REGISTRY_COLUMNS, STEP5_OUTPUT_FILES["model_registry"])
    if errors:
        return errors
    if registry.empty:
        errors.append("step5_model_registry.csv is empty")
    duplicate_models = int(registry.duplicated(["model_id"]).sum())
    if duplicate_models:
        errors.append(f"step5_model_registry.csv duplicate model_id rows: {duplicate_models}")
    if not registry["status"].astype(str).eq("ready").all():
        errors.append("all model_registry status values must be ready")
    if registry["random_seed"].isna().any() or registry["random_seed"].astype(str).str.strip().eq("").any():
        errors.append("model_registry random_seed must be non-empty")
    missing_artifacts = []
    for path_text in registry["model_artifact_path"].astype(str):
        path = Path(path_text)
        if not path.exists():
            alt = model_dir / path.name
            if not alt.exists():
                missing_artifacts.append(path_text)
    if missing_artifacts:
        errors.append(f"model artifacts missing: {missing_artifacts[:5]}")
    return errors


def validate_predictions(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    step4_output_dir: Path | None,
    *,
    candidate_size: int,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(
        predictions,
        STEP5_WALK_FORWARD_PREDICTIONS_COLUMNS,
        STEP5_OUTPUT_FILES["walk_forward_predictions"],
    )
    errors += validate_columns(metrics, STEP5_WALK_FORWARD_METRICS_COLUMNS, STEP5_OUTPUT_FILES["walk_forward_metrics"])
    if errors:
        return errors
    if predictions.empty:
        errors.append("step5_walk_forward_predictions.csv is empty")
    if metrics.empty:
        errors.append("step5_walk_forward_metrics.csv is empty")
    duplicate_predictions = int(predictions.duplicated(["wf_round", "预测日期T", "股票代码", "model_id"]).sum())
    if duplicate_predictions:
        errors.append(f"walk_forward_predictions duplicate prediction rows: {duplicate_predictions}")
    duplicate_metrics = int(metrics.duplicated(["wf_round"]).sum())
    if duplicate_metrics:
        errors.append(f"walk_forward_metrics duplicate wf_round rows: {duplicate_metrics}")
    pred_rounds = set(pd.to_numeric(predictions["wf_round"], errors="coerce").dropna().astype(int))
    metric_rounds = set(pd.to_numeric(metrics["wf_round"], errors="coerce").dropna().astype(int))
    if pred_rounds != metric_rounds:
        errors.append("prediction wf_rounds must match metric wf_rounds")
    if not predictions["prediction_scope"].astype(str).eq("walk_forward_eval").all():
        errors.append("walk_forward prediction_scope must be walk_forward_eval")
    rank_check = pd.to_numeric(predictions["fusion_rank"], errors="coerce")
    flags = pd.to_numeric(predictions["candidate_top30_flag"], errors="coerce").fillna(0).astype(int)
    if not (flags.eq(1) == rank_check.le(candidate_size)).all():
        errors.append("candidate_top30_flag must equal fusion_rank <= candidate_size")
    if not pd.to_numeric(metrics["candidate_size"], errors="coerce").fillna(-1).astype(int).eq(candidate_size).all():
        errors.append("walk_forward_metrics candidate_size mismatch")

    if step4_output_dir is not None:
        split = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["split_detail"])
        walk_forward = read_csv(step4_output_dir / STEP4_OUTPUT_FILES["walk_forward"])
        expected_eval = expected_eval_dates_without_final_test(walk_forward, split)
        final_test_dates = set(split.loc[split["split_role"].eq("final_test"), "样本日期T"].astype(str))
        prediction_dates = set(predictions["预测日期T"].astype(str))
        if prediction_dates & final_test_dates:
            errors.append("walk_forward_predictions must not include final_test dates")
        for wf_round, group in predictions.groupby("wf_round"):
            round_id = int(wf_round)
            dates = set(group["预测日期T"].astype(str))
            if round_id not in expected_eval:
                errors.append(f"walk_forward_predictions include ineligible wf_round={round_id}")
                continue
            if dates != expected_eval[round_id]:
                errors.append(f"walk_forward_predictions eval dates mismatch for wf_round={round_id}")
                break
    return errors


def validate_candidate(candidate: pd.DataFrame, *, candidate_size: int, prediction_date: str) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(candidate, STEP5_CANDIDATE_COLUMNS, STEP5_OUTPUT_FILES["candidate_top30"])
    if errors:
        return errors
    if len(candidate) != candidate_size:
        errors.append(f"step5_candidate_top30.csv row count expected {candidate_size}, got {len(candidate)}")
    duplicate_codes = int(candidate.duplicated(["股票代码"]).sum())
    if duplicate_codes:
        errors.append(f"step5_candidate_top30.csv duplicate 股票代码 rows: {duplicate_codes}")
    if set(candidate["candidate_date"].astype(str)) != {str(prediction_date)}:
        errors.append("candidate_date must equal manifest prediction_date")
    expected_ranks = list(range(1, len(candidate) + 1))
    actual_model_rank = pd.to_numeric(candidate["model_rank"], errors="coerce").fillna(-1).astype(int).tolist()
    actual_fusion_rank = pd.to_numeric(candidate["fusion_rank"], errors="coerce").fillna(-1).astype(int).tolist()
    if sorted(actual_model_rank) != expected_ranks:
        errors.append("candidate model_rank must be 1..candidate_size")
    if sorted(actual_fusion_rank) != expected_ranks:
        errors.append("candidate fusion_rank must be 1..candidate_size")
    forbidden = sorted(FORBIDDEN_CANDIDATE_COLUMNS & set(candidate.columns))
    if forbidden:
        errors.append(f"candidate_top30 contains forbidden columns: {forbidden}")
    return errors


def validate_feature_importance(importance: pd.DataFrame, feature_set: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(importance, STEP5_FEATURE_IMPORTANCE_COLUMNS, STEP5_OUTPUT_FILES["feature_importance"])
    if errors:
        return errors
    if importance.empty:
        errors.append("step5_feature_importance.csv is empty")
    feature_names = set(feature_set["feature_name"].astype(str))
    importance_names = set(importance["feature_name"].astype(str))
    missing = sorted(feature_names - importance_names)
    if missing:
        errors.append(f"feature_importance missing features: {missing[:5]}")
    return errors


def validate_manifest(
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    feature_set: pd.DataFrame,
    metrics: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_size: int,
    random_seed: int,
    input_metrics: dict[str, object] | None,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(manifest, ["项目", "说明"], STEP5_OUTPUT_FILES["manifest"])
    errors += require_manifest_items(manifest, REQUIRED_STEP5_MANIFEST_ITEMS, STEP5_OUTPUT_FILES["manifest"])
    if errors:
        return errors
    if manifest_value(manifest, "schema_version") != SCHEMA_VERSION:
        errors.append(f"step5_model_manifest.csv schema_version expected {SCHEMA_VERSION}")
    if manifest_value(manifest, "model_set_id") != MODEL_SET_ID:
        errors.append(f"step5_model_manifest.csv model_set_id expected {MODEL_SET_ID}")
    if manifest_value(manifest, "feature_set_id") != FEATURE_SET_ID:
        errors.append(f"step5_model_manifest.csv feature_set_id expected {FEATURE_SET_ID}")
    if manifest_value(manifest, "input_step3_sample_set_id") != SAMPLE_SET_ID:
        errors.append(f"input_step3_sample_set_id expected {SAMPLE_SET_ID}")
    if manifest_value(manifest, "input_step4_split_set_id") != SPLIT_SET_ID:
        errors.append(f"input_step4_split_set_id expected {SPLIT_SET_ID}")
    if int_value(manifest_value(manifest, "feature_count")) != len(feature_set):
        errors.append("manifest feature_count mismatch")
    if int_value(manifest_value(manifest, "model_count")) != len(registry):
        errors.append("manifest model_count mismatch")
    if int_value(manifest_value(manifest, "candidate_size")) != candidate_size:
        errors.append("manifest candidate_size mismatch")
    if int_value(manifest_value(manifest, "random_seed")) != random_seed:
        errors.append("manifest random_seed mismatch")
    if int_value(manifest_value(manifest, "walk_forward_rounds_used")) != len(metrics):
        errors.append("manifest walk_forward_rounds_used mismatch")
    if manifest_value(manifest, "prediction_date") != str(candidate["candidate_date"].iloc[0]):
        errors.append("manifest prediction_date mismatch candidate_date")
    if not manifest_value(manifest, "generated_at").strip():
        errors.append("manifest generated_at is empty")
    if not manifest_value(manifest, "leakage_control_note").strip():
        errors.append("manifest leakage_control_note is empty")

    if input_metrics:
        for key in ["input_step2_experiment", "input_step3_experiment", "input_step4_experiment"]:
            if manifest_value(manifest, key) != str(input_metrics.get(key, "")):
                errors.append(f"manifest {key} mismatch selected input")
        if manifest_value(manifest, "input_step2_latest_T") != str(input_metrics.get("input_step2_latest_T", "")):
            errors.append("manifest input_step2_latest_T mismatch")
    return errors


def validate_leakage_check(leakage_check: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(leakage_check, STEP5_LEAKAGE_CHECK_COLUMNS, STEP5_OUTPUT_FILES["leakage_check"])
    if errors:
        return errors
    if leakage_check.empty:
        errors.append("step5_leakage_check.csv is empty")
    bad = leakage_check[~leakage_check["状态"].astype(str).eq("PASS")]
    if not bad.empty:
        errors.append(f"step5_leakage_check.csv has non-PASS rows: {bad['检查项'].astype(str).tolist()}")
    return errors


def validate_outputs(
    output_dir: Path,
    model_dir: Path,
    *,
    step4_output_dir: Path | None = None,
    input_metrics: dict[str, object] | None = None,
    candidate_size: int = DEFAULT_CANDIDATE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    errors: list[str] = []

    registry = read_csv(output_dir / STEP5_OUTPUT_FILES["model_registry"])
    feature_set = read_csv(output_dir / STEP5_OUTPUT_FILES["feature_set"])
    predictions = read_csv(output_dir / STEP5_OUTPUT_FILES["walk_forward_predictions"], dtype={"股票代码": str})
    metrics = read_csv(output_dir / STEP5_OUTPUT_FILES["walk_forward_metrics"])
    importance = read_csv(output_dir / STEP5_OUTPUT_FILES["feature_importance"])
    candidate = read_csv(output_dir / STEP5_OUTPUT_FILES["candidate_top30"], dtype={"股票代码": str})
    manifest = read_csv(output_dir / STEP5_OUTPUT_FILES["manifest"])
    leakage_check = read_csv(output_dir / STEP5_OUTPUT_FILES["leakage_check"])

    prediction_date = manifest_value(manifest, "prediction_date", "")
    errors += validate_feature_set(feature_set)
    errors += validate_model_registry(registry, model_dir)
    errors += validate_predictions(
        predictions,
        metrics,
        step4_output_dir,
        candidate_size=candidate_size,
    )
    errors += validate_candidate(candidate, candidate_size=candidate_size, prediction_date=prediction_date)
    errors += validate_feature_importance(importance, feature_set)
    errors += validate_manifest(
        manifest,
        registry,
        feature_set,
        metrics,
        candidate,
        candidate_size=candidate_size,
        random_seed=random_seed,
        input_metrics=input_metrics,
    )
    errors += validate_leakage_check(leakage_check)

    if (output_dir / "result.csv").exists() or (output_dir.parent / "result.csv").exists():
        errors.append("Step-5 must not generate result.csv")

    raise_if_errors(errors)

    return {
        "output_model_count": int(len(registry)),
        "output_feature_count": int(len(feature_set)),
        "output_walk_forward_prediction_rows": int(len(predictions)),
        "output_walk_forward_rounds_used": int(len(metrics)),
        "output_candidate_rows": int(len(candidate)),
        "output_candidate_date": "" if candidate.empty else str(candidate["candidate_date"].iloc[0]),
        "output_top_candidate_code": "" if candidate.empty else str(candidate.sort_values("fusion_rank")["股票代码"].iloc[0]),
        "output_top_candidate_name": "" if candidate.empty else str(candidate.sort_values("fusion_rank")["股票名称"].iloc[0]),
        "output_leakage_check_rows": int(len(leakage_check)),
    }


def validate_step5(
    step2_experiment_dir: Path,
    step3_experiment_dir: Path,
    step4_experiment_dir: Path,
    output_dir: Path,
    model_dir: Path,
    *,
    candidate_size: int = DEFAULT_CANDIDATE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    input_metrics = validate_inputs(step2_experiment_dir, step3_experiment_dir, step4_experiment_dir)
    output_metrics = validate_outputs(
        output_dir,
        model_dir,
        step4_output_dir=Path(step4_experiment_dir) / "outputs" / "step4",
        input_metrics=input_metrics,
        candidate_size=candidate_size,
        random_seed=random_seed,
    )
    input_metrics.update(output_metrics)
    return input_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-5 model and candidate outputs.")
    parser.add_argument("--step2-experiment-dir", type=Path, required=True)
    parser.add_argument("--step3-experiment-dir", type=Path, required=True)
    parser.add_argument("--step4-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidate-size", type=int, default=DEFAULT_CANDIDATE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step5(
            args.step2_experiment_dir,
            args.step3_experiment_dir,
            args.step4_experiment_dir,
            args.output_dir,
            args.model_dir,
            candidate_size=args.candidate_size,
            random_seed=args.random_seed,
        )
    except Step5ValidationError as exc:
        print(f"Step-5 validation failed: {exc}")
        return 1

    print("Step-5 validation passed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
