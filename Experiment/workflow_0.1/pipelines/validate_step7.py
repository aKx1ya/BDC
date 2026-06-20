#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step7_outputs import (  # noqa: E402
    EVALUATION_SET_ID,
    SCHEMA_VERSION,
    STEP6_OUTPUT_FILES,
    STEP7_FROZEN_RESULT_COLUMNS,
    STEP7_LEAKAGE_CHECK_COLUMNS,
    STEP7_SCORE_SUMMARY_COLUMNS,
    STEP7_STOCK_CONTRIBUTION_COLUMNS,
    VALID_SCORE_MODES,
    manifest_value,
    normalize_code,
)


STEP7_OUTPUT_FILES = {
    "frozen_result": "step7_frozen_result.csv",
    "score_summary": "step7_score_summary.csv",
    "stock_contribution": "step7_stock_contribution.csv",
    "manifest": "step7_score_manifest.csv",
    "leakage_check": "step7_leakage_check.csv",
}


REQUIRED_STEP7_MANIFEST_ITEMS = {
    "schema_version",
    "evaluation_set_id",
    "input_step6_experiment",
    "input_step6_result_path",
    "frozen_result_path",
    "score_mode",
    "official_script_path",
    "test_data_path",
    "selected_count",
    "total_weight",
    "final_score",
    "generated_at",
    "output_dir",
    "data_window_note",
    "leakage_control_note",
}


COMMON_STEP7_LEAKAGE_CHECKS = {
    "input_step6_success",
    "input_step6_leakage_pass",
    "result_frozen_before_test_read",
    "frozen_result_matches_step6_result",
    "result_schema_valid",
    "result_stock_count_lte_5",
    "result_stock_id_unique",
    "result_weight_non_negative",
    "result_weight_sum_lte_1",
    "test_data_read_after_freeze",
    "official_score_not_used_to_modify_step6",
    "manifest_leakage_note",
}


LOCAL_SCORE_LEAKAGE_CHECKS = {
    "test_data_available",
    "selected_stocks_covered_by_test",
    "each_selected_stock_has_5_test_rows",
    "test_data_is_future_of_candidate_date",
    "official_script_completed",
    "final_score_not_negative_999",
    "stock_contribution_matches_final_score",
}


class Step7ValidationError(Exception):
    """Step-7 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step7ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step7ValidationError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> list[str]:
    actual = list(df.columns)
    if actual != expected:
        return [f"{file_name} columns mismatch: expected {expected}, got {actual}"]
    return []


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


def float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def normalize_result(result: pd.DataFrame) -> pd.DataFrame:
    out = result.copy()
    out["stock_id"] = out["stock_id"].map(normalize_code)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    return out[STEP7_FROZEN_RESULT_COLUMNS].sort_values("stock_id").reset_index(drop=True)


def validate_result_frame(result: pd.DataFrame, file_name: str) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(result, STEP7_FROZEN_RESULT_COLUMNS, file_name)
    if errors:
        return errors
    if len(result) > 5:
        errors.append(f"{file_name} row count must be <= 5")
    stock_id = result["stock_id"].map(normalize_code)
    if stock_id.duplicated().any():
        errors.append(f"{file_name} stock_id values must be unique")
    weights = pd.to_numeric(result["weight"], errors="coerce")
    if weights.isna().any():
        errors.append(f"{file_name} weight must be numeric")
    if weights.lt(0).any():
        errors.append(f"{file_name} weight must be non-negative")
    if float(weights.sum()) > 1.0000001:
        errors.append(f"{file_name} weight sum must be <= 1")
    return errors


def validate_inputs(step6_experiment_dir: Path) -> dict[str, object]:
    step6_experiment_dir = Path(step6_experiment_dir)
    step6_output_dir = step6_experiment_dir / "outputs" / "step6"
    errors: list[str] = []

    if not report_is_success(step6_experiment_dir / "notes" / "step6_run_report.md"):
        errors.append(f"Step-6 report is not SUCCESS: {step6_experiment_dir}")

    result = read_csv(step6_output_dir / STEP6_OUTPUT_FILES["result"], dtype={"stock_id": str})
    manifest = read_csv(step6_output_dir / STEP6_OUTPUT_FILES["manifest"])
    leakage = read_csv(step6_output_dir / STEP6_OUTPUT_FILES["leakage_check"])
    errors += validate_result_frame(result, STEP6_OUTPUT_FILES["result"])

    if "状态" not in leakage.columns or not leakage["状态"].astype(str).eq("PASS").all():
        errors.append("Step-6 leakage_check must be all PASS")

    candidate_date = manifest_value(manifest, "input_candidate_date")
    raise_if_errors(errors)

    return {
        "input_step6_experiment": step6_experiment_dir.name,
        "input_step6_result_path": str(step6_output_dir / STEP6_OUTPUT_FILES["result"]),
        "input_candidate_date": candidate_date,
        "input_selected_count": int(len(result)),
        "input_total_weight": float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0,
        "input_result_codes": sorted(set(result["stock_id"].map(normalize_code))),
    }


def validate_frozen_result(
    frozen_result: pd.DataFrame,
    step6_result: pd.DataFrame | None,
    *,
    input_metrics: dict[str, object] | None,
) -> list[str]:
    errors: list[str] = []
    errors += validate_result_frame(frozen_result, STEP7_OUTPUT_FILES["frozen_result"])
    if errors:
        return errors
    if input_metrics:
        codes = sorted(set(frozen_result["stock_id"].map(normalize_code)))
        if codes != sorted(input_metrics.get("input_result_codes", [])):
            errors.append("frozen_result stock_id set mismatch Step-6 result")
        if len(frozen_result) != int(input_metrics.get("input_selected_count", -1)):
            errors.append("frozen_result row count mismatch Step-6 result")
    if step6_result is not None:
        if not normalize_result(frozen_result).equals(normalize_result(step6_result)):
            errors.append("frozen_result must exactly match Step-6 result")
    return errors


def validate_score_summary(
    summary: pd.DataFrame,
    frozen_result: pd.DataFrame,
    contribution: pd.DataFrame,
    *,
    mode: str,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(summary, STEP7_SCORE_SUMMARY_COLUMNS, STEP7_OUTPUT_FILES["score_summary"])
    if errors:
        return errors
    if len(summary) != 1:
        errors.append("step7_score_summary.csv must have exactly one row")
        return errors
    row = summary.iloc[0]
    actual_mode = str(row["score_mode"])
    if actual_mode not in VALID_SCORE_MODES:
        errors.append(f"score_summary score_mode invalid: {actual_mode}")
    if actual_mode != mode:
        errors.append(f"score_summary score_mode expected {mode}, got {actual_mode}")
    if int_value(row["selected_count"]) != len(frozen_result):
        errors.append("score_summary selected_count mismatch frozen_result")
    total_weight = float(pd.to_numeric(frozen_result["weight"], errors="coerce").sum()) if not frozen_result.empty else 0.0
    if abs(float_value(row["total_weight"]) - total_weight) > 1e-8:
        errors.append("score_summary total_weight mismatch frozen_result")

    status = str(row["result_status"])
    if mode == "freeze-only":
        if status != "FREEZE_ONLY_SUCCESS":
            errors.append("freeze-only result_status must be FREEZE_ONLY_SUCCESS")
        if str(row["final_score"]).strip() not in {"", "nan", "NaN"}:
            errors.append("freeze-only final_score must be empty")
    else:
        if status != "SCORE_SUCCESS":
            errors.append("local-score result_status must be SCORE_SUCCESS")
        final_score = pd.to_numeric(pd.Series([row["final_score"]]), errors="coerce").iloc[0]
        if pd.isna(final_score):
            errors.append("local-score final_score must be numeric")
        elif float(final_score) == -999:
            errors.append("local-score final_score must not be -999")
        contribution_sum = float(pd.to_numeric(contribution["score_contribution"], errors="coerce").sum()) if not contribution.empty else 0.0
        if pd.notna(final_score) and abs(contribution_sum - float(final_score)) > 1e-8:
            errors.append("stock_contribution score_contribution sum must equal final_score")
    return errors


def validate_stock_contribution(
    contribution: pd.DataFrame,
    frozen_result: pd.DataFrame,
    *,
    mode: str,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(contribution, STEP7_STOCK_CONTRIBUTION_COLUMNS, STEP7_OUTPUT_FILES["stock_contribution"])
    if errors:
        return errors
    if mode == "freeze-only":
        if not contribution.empty:
            errors.append("freeze-only stock_contribution must be empty")
        return errors

    result_codes = set(frozen_result["stock_id"].map(normalize_code))
    contribution_codes = set(contribution["stock_id"].map(normalize_code))
    if contribution_codes != result_codes:
        errors.append("stock_contribution must cover exactly all frozen_result stocks in local-score mode")
    if pd.to_numeric(contribution["matched_test_rows"], errors="coerce").lt(5).any():
        errors.append("stock_contribution matched_test_rows must be >= 5")
    numeric_cols = ["weight", "open_first", "open_last", "return_5d_open_to_open", "score_contribution"]
    for col in numeric_cols:
        if pd.to_numeric(contribution[col], errors="coerce").isna().any():
            errors.append(f"stock_contribution {col} must be numeric")
    return errors


def validate_manifest(
    manifest: pd.DataFrame,
    frozen_result: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    input_metrics: dict[str, object] | None,
    mode: str,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(manifest, ["项目", "说明"], STEP7_OUTPUT_FILES["manifest"])
    errors += require_manifest_items(manifest, REQUIRED_STEP7_MANIFEST_ITEMS, STEP7_OUTPUT_FILES["manifest"])
    if errors:
        return errors
    if manifest_value(manifest, "schema_version") != SCHEMA_VERSION:
        errors.append(f"step7_score_manifest.csv schema_version expected {SCHEMA_VERSION}")
    if manifest_value(manifest, "evaluation_set_id") != EVALUATION_SET_ID:
        errors.append(f"step7_score_manifest.csv evaluation_set_id expected {EVALUATION_SET_ID}")
    if manifest_value(manifest, "score_mode") != mode:
        errors.append("manifest score_mode mismatch")
    if int_value(manifest_value(manifest, "selected_count")) != len(frozen_result):
        errors.append("manifest selected_count mismatch frozen_result")
    total_weight = float(pd.to_numeric(frozen_result["weight"], errors="coerce").sum()) if not frozen_result.empty else 0.0
    if abs(float_value(manifest_value(manifest, "total_weight")) - total_weight) > 1e-8:
        errors.append("manifest total_weight mismatch frozen_result")
    if not manifest_value(manifest, "generated_at").strip():
        errors.append("manifest generated_at is empty")
    if not manifest_value(manifest, "leakage_control_note").strip():
        errors.append("manifest leakage_control_note is empty")
    if input_metrics and manifest_value(manifest, "input_step6_experiment") != str(input_metrics.get("input_step6_experiment", "")):
        errors.append("manifest input_step6_experiment mismatch")
    summary_final = "" if summary.empty else str(summary["final_score"].iloc[0])
    manifest_final = manifest_value(manifest, "final_score")
    if mode == "local-score" and summary_final.strip() and abs(float_value(manifest_final) - float_value(summary_final)) > 1e-8:
        errors.append("manifest final_score mismatch score_summary")
    return errors


def validate_leakage_check(leakage: pd.DataFrame, *, mode: str) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(leakage, STEP7_LEAKAGE_CHECK_COLUMNS, STEP7_OUTPUT_FILES["leakage_check"])
    if errors:
        return errors
    checks = set(leakage["检查项"].astype(str))
    required = set(COMMON_STEP7_LEAKAGE_CHECKS)
    if mode == "local-score":
        required |= LOCAL_SCORE_LEAKAGE_CHECKS
    missing = sorted(required - checks)
    if missing:
        errors.append(f"step7_leakage_check.csv missing checks: {missing}")
    bad = leakage[~leakage["状态"].astype(str).eq("PASS")]
    if not bad.empty:
        errors.append(f"step7_leakage_check.csv has non-PASS rows: {bad['检查项'].astype(str).tolist()}")
    return errors


def validate_outputs(
    output_dir: Path,
    *,
    mode: str,
    step6_result_path: Path | None = None,
    input_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if mode not in VALID_SCORE_MODES:
        raise Step7ValidationError(f"mode must be one of {sorted(VALID_SCORE_MODES)}, got {mode!r}")
    errors: list[str] = []

    frozen = read_csv(output_dir / STEP7_OUTPUT_FILES["frozen_result"], dtype={"stock_id": str})
    summary = read_csv(output_dir / STEP7_OUTPUT_FILES["score_summary"], dtype={"experiment_id": str})
    contribution = read_csv(output_dir / STEP7_OUTPUT_FILES["stock_contribution"], dtype={"stock_id": str})
    manifest = read_csv(output_dir / STEP7_OUTPUT_FILES["manifest"])
    leakage = read_csv(output_dir / STEP7_OUTPUT_FILES["leakage_check"])
    step6_result = read_csv(step6_result_path, dtype={"stock_id": str}) if step6_result_path else None

    errors += validate_frozen_result(frozen, step6_result, input_metrics=input_metrics)
    errors += validate_stock_contribution(contribution, frozen, mode=mode)
    errors += validate_score_summary(summary, frozen, contribution, mode=mode)
    errors += validate_manifest(manifest, frozen, summary, input_metrics=input_metrics, mode=mode)
    errors += validate_leakage_check(leakage, mode=mode)

    raise_if_errors(errors)

    final_score_raw = "" if summary.empty else str(summary["final_score"].iloc[0])
    final_score = "" if final_score_raw in {"", "nan", "NaN"} else float_value(final_score_raw)
    return {
        "output_score_mode": mode,
        "output_selected_count": int(len(frozen)),
        "output_total_weight": float(pd.to_numeric(frozen["weight"], errors="coerce").sum()) if not frozen.empty else 0.0,
        "output_result_status": "" if summary.empty else str(summary["result_status"].iloc[0]),
        "output_final_score": final_score,
        "output_test_date_start": "" if summary.empty else str(summary["test_date_start"].iloc[0]),
        "output_test_date_end": "" if summary.empty else str(summary["test_date_end"].iloc[0]),
    }


def validate_step7(step6_experiment_dir: Path, output_dir: Path, *, mode: str) -> dict[str, object]:
    input_metrics = validate_inputs(step6_experiment_dir)
    output_metrics = validate_outputs(
        output_dir,
        mode=mode,
        step6_result_path=Path(input_metrics["input_step6_result_path"]),
        input_metrics=input_metrics,
    )
    input_metrics.update(output_metrics)
    return input_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-7 frozen score outputs.")
    parser.add_argument("--step6-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(VALID_SCORE_MODES), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step7(args.step6_experiment_dir, args.output_dir, mode=args.mode)
    except Step7ValidationError as exc:
        print(f"Step-7 validation failed: {exc}")
        return 1

    print("Step-7 validation passed")
    for key, value in metrics.items():
        if key == "input_result_codes":
            print(f"{key}: {len(value)} codes")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
