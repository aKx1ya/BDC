#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step6_outputs import (  # noqa: E402
    DEFAULT_MAX_STOCK_COUNT,
    DEFAULT_MAX_PER_SECTOR,
    DEFAULT_MIN_TURNOVER,
    FORBIDDEN_CANDIDATE_COLUMNS,
    REFINE_SET_ID,
    SCHEMA_VERSION,
    STEP2_OUTPUT_FILES,
    STEP5_OUTPUT_FILES,
    STEP6_FINAL_TOP5_COLUMNS,
    STEP6_LEAKAGE_CHECK_COLUMNS,
    STEP6_RANKING_LOG_COLUMNS,
    STEP6_RESULT_COLUMNS,
    STEP6_WEIGHT_PLAN_COLUMNS,
    WEIGHTING_METHOD,
    manifest_value,
    normalize_code,
)


STEP6_OUTPUT_FILES = {
    "ranking_log": "step6_ranking_log.csv",
    "final_top5": "step6_final_top5.csv",
    "result": "step6_result.csv",
    "weight_plan": "step6_weight_plan.csv",
    "manifest": "step6_refine_manifest.csv",
    "leakage_check": "step6_leakage_check.csv",
}


REQUIRED_STEP6_MANIFEST_ITEMS = {
    "schema_version",
    "refine_set_id",
    "input_step5_path",
    "input_step2_path",
    "input_step5_experiment",
    "input_step2_experiment",
    "input_candidate_date",
    "input_candidate_size",
    "selected_count",
    "weighting_method",
    "total_weight",
    "cash_weight",
    "max_stock_count",
    "max_per_sector",
    "min_turnover",
    "generated_at",
    "output_dir",
    "data_window_note",
    "leakage_control_note",
}


REQUIRED_STEP6_LEAKAGE_CHECKS = {
    "input_step5_success",
    "candidate_date_matches_step2_latest_T",
    "all_selected_from_top30",
    "no_future_label_columns_used",
    "no_step7_score_used",
    "result_schema_valid",
    "result_stock_count_lte_5",
    "result_weight_non_negative",
    "result_weight_sum_lte_1",
    "ranking_log_covers_all_candidates",
    "manifest_leakage_note",
}


ALLOWED_GATE_STATUS = {"pass", "removed", "relaxed", "selected"}
ALLOWED_SECTOR_STATUS = {"pass", "blocked", "relaxed", "not_applicable"}


class Step6ValidationError(Exception):
    """Step-6 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step6ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step6ValidationError(f"missing file: {path}")
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


def validate_inputs(step5_experiment_dir: Path, step2_experiment_dir: Path) -> dict[str, object]:
    step5_experiment_dir = Path(step5_experiment_dir)
    step2_experiment_dir = Path(step2_experiment_dir)
    step5_output_dir = step5_experiment_dir / "outputs" / "step5"
    step2_output_dir = step2_experiment_dir / "outputs" / "step2"
    errors: list[str] = []

    if not report_is_success(step5_experiment_dir / "notes" / "step5_run_report.md"):
        errors.append(f"Step-5 report is not SUCCESS: {step5_experiment_dir}")
    if not report_is_success(step2_experiment_dir / "notes" / "step2_run_report.md"):
        errors.append(f"Step-2 report is not SUCCESS: {step2_experiment_dir}")

    candidate = read_csv(step5_output_dir / STEP5_OUTPUT_FILES["candidate"], dtype={"股票代码": str})
    step5_manifest = read_csv(step5_output_dir / STEP5_OUTPUT_FILES["manifest"])
    step5_leakage = read_csv(step5_output_dir / STEP5_OUTPUT_FILES["leakage_check"])
    step2_feature = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str})
    step2_manifest = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["manifest"])

    if "状态" not in step5_leakage.columns or not step5_leakage["状态"].astype(str).eq("PASS").all():
        errors.append("Step-5 leakage_check must be all PASS")

    errors += validate_columns(
        candidate,
        [
            "candidate_date",
            "股票代码",
            "股票名称",
            "板块划分",
            "model_score",
            "model_rank",
            "fusion_score",
            "fusion_rank",
            "model_source",
            "fusion_method",
            "candidate_size",
            "generated_at",
        ],
        STEP5_OUTPUT_FILES["candidate"],
    )
    forbidden = sorted(FORBIDDEN_CANDIDATE_COLUMNS & set(candidate.columns))
    if forbidden:
        errors.append(f"Step-5 candidate_top30 contains forbidden columns: {forbidden}")

    candidate_size = int_value(manifest_value(step5_manifest, "candidate_size"), len(candidate))
    if len(candidate) != candidate_size:
        errors.append(f"Step-5 candidate_top30 row count expected {candidate_size}, got {len(candidate)}")
    if candidate["股票代码"].map(normalize_code).duplicated().any():
        errors.append("Step-5 candidate_top30 contains duplicate 股票代码")

    inferred_step2 = manifest_value(step5_manifest, "input_step2_experiment")
    if inferred_step2 and inferred_step2 != step2_experiment_dir.name:
        errors.append(f"Step-5 manifest input_step2_experiment={inferred_step2} does not match {step2_experiment_dir.name}")

    candidate_dates = set(pd.to_datetime(candidate["candidate_date"], errors="coerce").dt.strftime("%Y-%m-%d"))
    if len(candidate_dates) != 1:
        errors.append(f"Step-5 candidate_top30 must contain one candidate_date, got {sorted(candidate_dates)}")
        candidate_date = ""
    else:
        candidate_date = next(iter(candidate_dates))

    latest_t = manifest_value(step2_manifest, "latest_T")
    if candidate_date and latest_t and candidate_date != latest_t:
        errors.append(f"Step-5 candidate_date {candidate_date} does not match Step-2 latest_T {latest_t}")

    if "日期" not in step2_feature.columns or "股票代码" not in step2_feature.columns:
        errors.append("Step-2 feature table must contain 股票代码 and 日期")
        latest_feature = step2_feature.iloc[0:0].copy()
    else:
        feature = step2_feature.copy()
        feature["股票代码"] = feature["股票代码"].map(normalize_code)
        feature["日期"] = pd.to_datetime(feature["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        latest_feature = feature[feature["日期"].astype(str).eq(str(latest_t))].copy()
        latest_feature = latest_feature.drop_duplicates(["股票代码"], keep="last")
        candidate_codes = set(candidate["股票代码"].map(normalize_code))
        feature_codes = set(latest_feature["股票代码"].astype(str))
        missing_codes = sorted(candidate_codes - feature_codes)
        if missing_codes:
            errors.append(f"Step-2 latest_T features missing Top30 stocks: {missing_codes[:10]}")

    raise_if_errors(errors)

    return {
        "input_step5_experiment": step5_experiment_dir.name,
        "input_step2_experiment": step2_experiment_dir.name,
        "input_candidate_date": candidate_date,
        "input_step2_latest_T": latest_t,
        "input_candidate_size": candidate_size,
        "input_candidate_codes": sorted(set(candidate["股票代码"].map(normalize_code))),
        "input_step2_latest_rows": int(len(latest_feature)),
    }


def validate_ranking_log(
    ranking: pd.DataFrame,
    final_top5: pd.DataFrame,
    result: pd.DataFrame,
    *,
    candidate_codes: set[str],
    candidate_size: int,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(ranking, STEP6_RANKING_LOG_COLUMNS, STEP6_OUTPUT_FILES["ranking_log"])
    if errors:
        return errors
    if len(ranking) != candidate_size:
        errors.append(f"step6_ranking_log.csv row count expected {candidate_size}, got {len(ranking)}")
    ranking_codes = set(ranking["股票代码"].map(normalize_code))
    if ranking_codes != candidate_codes:
        errors.append("step6_ranking_log.csv must cover exactly all Step-5 candidates")
    if ranking["股票代码"].map(normalize_code).duplicated().any():
        errors.append("step6_ranking_log.csv duplicate 股票代码 rows")

    bad_gate = sorted(set(ranking["gate_status"].astype(str)) - ALLOWED_GATE_STATUS)
    if bad_gate:
        errors.append(f"step6_ranking_log.csv invalid gate_status values: {bad_gate}")
    bad_sector = sorted(set(ranking["sector_constraint_status"].astype(str)) - ALLOWED_SECTOR_STATUS)
    if bad_sector:
        errors.append(f"step6_ranking_log.csv invalid sector_constraint_status values: {bad_sector}")

    removed = ranking["gate_status"].astype(str).eq("removed")
    if ranking.loc[removed, "removed_reason"].astype(str).str.strip().eq("").any():
        errors.append("removed candidates must have removed_reason")
    if pd.to_numeric(ranking["refine_score"], errors="coerce").isna().any():
        errors.append("ranking_log refine_score must be numeric")

    selected = ranking[pd.to_numeric(ranking["final_selected"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    selected_codes = set(selected["股票代码"].map(normalize_code))
    final_codes = set(final_top5["股票代码"].map(normalize_code)) if not final_top5.empty else set()
    result_codes = set(result["stock_id"].map(normalize_code)) if not result.empty else set()
    if selected_codes != final_codes or selected_codes != result_codes:
        errors.append("ranking_log final_selected stocks must match final_top5 and result")

    if not selected.empty:
        ranks = sorted(pd.to_numeric(selected["final_rank"], errors="coerce").dropna().astype(int).tolist())
        if ranks != list(range(1, len(selected) + 1)):
            errors.append("selected final_rank must be continuous from 1")
    return errors


def validate_final_top5(final_top5: pd.DataFrame, result: pd.DataFrame, *, max_stock_count: int) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(final_top5, STEP6_FINAL_TOP5_COLUMNS, STEP6_OUTPUT_FILES["final_top5"])
    if errors:
        return errors
    if len(final_top5) > max_stock_count:
        errors.append(f"step6_final_top5.csv row count must be <= {max_stock_count}")
    if final_top5["股票代码"].map(normalize_code).duplicated().any():
        errors.append("step6_final_top5.csv duplicate 股票代码 rows")
    weights = pd.to_numeric(final_top5["weight"], errors="coerce")
    if weights.isna().any() or weights.lt(0).any() or weights.sum() > 1.0000001:
        errors.append("step6_final_top5.csv weights must be numeric, non-negative, and sum <= 1")

    if not final_top5.empty:
        ranks = sorted(pd.to_numeric(final_top5["final_rank"], errors="coerce").dropna().astype(int).tolist())
        if ranks != list(range(1, len(final_top5) + 1)):
            errors.append("step6_final_top5.csv final_rank must be continuous from 1")
    if len(final_top5) != len(result):
        errors.append("step6_final_top5.csv row count must match step6_result.csv")
    return errors


def validate_result(result: pd.DataFrame, *, candidate_codes: set[str], max_stock_count: int) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(result, STEP6_RESULT_COLUMNS, STEP6_OUTPUT_FILES["result"])
    if errors:
        return errors
    if len(result) > max_stock_count:
        errors.append(f"step6_result.csv row count must be <= {max_stock_count}")
    result_codes = set(result["stock_id"].map(normalize_code)) if not result.empty else set()
    if len(result_codes) != len(result):
        errors.append("step6_result.csv duplicate stock_id rows")
    outside = sorted(result_codes - candidate_codes)
    if outside:
        errors.append(f"step6_result.csv contains stocks outside Step-5 Top30: {outside}")
    weights = pd.to_numeric(result["weight"], errors="coerce")
    if weights.isna().any():
        errors.append("step6_result.csv weight must be numeric")
    if weights.lt(0).any():
        errors.append("step6_result.csv weight must be non-negative")
    if float(weights.sum()) > 1.0000001:
        errors.append("step6_result.csv weight sum must be <= 1")
    return errors


def validate_weight_plan(weight_plan: pd.DataFrame, result: pd.DataFrame, manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(weight_plan, STEP6_WEIGHT_PLAN_COLUMNS, STEP6_OUTPUT_FILES["weight_plan"])
    if errors:
        return errors
    if len(weight_plan) != 1:
        errors.append("step6_weight_plan.csv must have exactly one row")
        return errors
    selected_count = int_value(weight_plan["selected_count"].iloc[0])
    if selected_count != len(result):
        errors.append("weight_plan selected_count mismatch result rows")
    total_weight = float_value(weight_plan["total_weight"].iloc[0])
    result_weight = float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0
    if abs(total_weight - result_weight) > 1e-8:
        errors.append("weight_plan total_weight mismatch result")
    if str(weight_plan["weighting_method"].iloc[0]) != WEIGHTING_METHOD:
        errors.append(f"weight_plan weighting_method expected {WEIGHTING_METHOD}")
    if manifest_value(manifest, "weighting_method") != str(weight_plan["weighting_method"].iloc[0]):
        errors.append("manifest weighting_method mismatch weight_plan")
    return errors


def validate_manifest(
    manifest: pd.DataFrame,
    result: pd.DataFrame,
    *,
    input_metrics: dict[str, object] | None,
    max_stock_count: int,
    max_per_sector: int,
    min_turnover: float,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(manifest, ["项目", "说明"], STEP6_OUTPUT_FILES["manifest"])
    errors += require_manifest_items(manifest, REQUIRED_STEP6_MANIFEST_ITEMS, STEP6_OUTPUT_FILES["manifest"])
    if errors:
        return errors
    if manifest_value(manifest, "schema_version") != SCHEMA_VERSION:
        errors.append(f"step6_refine_manifest.csv schema_version expected {SCHEMA_VERSION}")
    if manifest_value(manifest, "refine_set_id") != REFINE_SET_ID:
        errors.append(f"step6_refine_manifest.csv refine_set_id expected {REFINE_SET_ID}")
    if manifest_value(manifest, "weighting_method") != WEIGHTING_METHOD:
        errors.append(f"step6_refine_manifest.csv weighting_method expected {WEIGHTING_METHOD}")
    if int_value(manifest_value(manifest, "selected_count")) != len(result):
        errors.append("manifest selected_count mismatch result rows")
    if int_value(manifest_value(manifest, "max_stock_count")) != max_stock_count:
        errors.append("manifest max_stock_count mismatch")
    if int_value(manifest_value(manifest, "max_per_sector")) != max_per_sector:
        errors.append("manifest max_per_sector mismatch")
    if abs(float_value(manifest_value(manifest, "min_turnover")) - float(min_turnover)) > 1e-6:
        errors.append("manifest min_turnover mismatch")
    if not manifest_value(manifest, "generated_at").strip():
        errors.append("manifest generated_at is empty")
    if not manifest_value(manifest, "leakage_control_note").strip():
        errors.append("manifest leakage_control_note is empty")

    total_weight = float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0
    if abs(float_value(manifest_value(manifest, "total_weight")) - total_weight) > 1e-8:
        errors.append("manifest total_weight mismatch result")
    if float_value(manifest_value(manifest, "cash_weight")) < -1e-8:
        errors.append("manifest cash_weight must be non-negative")

    if input_metrics:
        for key in ["input_step5_experiment", "input_step2_experiment", "input_candidate_date", "input_candidate_size"]:
            expected = str(input_metrics.get(key, ""))
            if manifest_value(manifest, key) != expected:
                errors.append(f"manifest {key} mismatch selected input")
    return errors


def validate_leakage_check(leakage_check: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(leakage_check, STEP6_LEAKAGE_CHECK_COLUMNS, STEP6_OUTPUT_FILES["leakage_check"])
    if errors:
        return errors
    checks = set(leakage_check["检查项"].astype(str))
    missing = sorted(REQUIRED_STEP6_LEAKAGE_CHECKS - checks)
    if missing:
        errors.append(f"step6_leakage_check.csv missing checks: {missing}")
    bad = leakage_check[~leakage_check["状态"].astype(str).eq("PASS")]
    if not bad.empty:
        errors.append(f"step6_leakage_check.csv has non-PASS rows: {bad['检查项'].astype(str).tolist()}")
    return errors


def validate_outputs(
    output_dir: Path,
    *,
    input_metrics: dict[str, object] | None = None,
    max_stock_count: int = DEFAULT_MAX_STOCK_COUNT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []

    ranking = read_csv(output_dir / STEP6_OUTPUT_FILES["ranking_log"], dtype={"股票代码": str})
    final_top5 = read_csv(output_dir / STEP6_OUTPUT_FILES["final_top5"], dtype={"股票代码": str})
    result = read_csv(output_dir / STEP6_OUTPUT_FILES["result"], dtype={"stock_id": str})
    weight_plan = read_csv(output_dir / STEP6_OUTPUT_FILES["weight_plan"])
    manifest = read_csv(output_dir / STEP6_OUTPUT_FILES["manifest"])
    leakage_check = read_csv(output_dir / STEP6_OUTPUT_FILES["leakage_check"])

    candidate_codes = set(input_metrics.get("input_candidate_codes", [])) if input_metrics else set(ranking["股票代码"].map(normalize_code))
    candidate_size = int(input_metrics.get("input_candidate_size", len(ranking))) if input_metrics else len(ranking)

    errors += validate_result(result, candidate_codes=candidate_codes, max_stock_count=max_stock_count)
    errors += validate_final_top5(final_top5, result, max_stock_count=max_stock_count)
    errors += validate_ranking_log(
        ranking,
        final_top5,
        result,
        candidate_codes=candidate_codes,
        candidate_size=candidate_size,
    )
    errors += validate_weight_plan(weight_plan, result, manifest)
    errors += validate_manifest(
        manifest,
        result,
        input_metrics=input_metrics,
        max_stock_count=max_stock_count,
        max_per_sector=max_per_sector,
        min_turnover=min_turnover,
    )
    errors += validate_leakage_check(leakage_check)

    raise_if_errors(errors)

    selected = final_top5.sort_values("final_rank") if not final_top5.empty else final_top5
    total_weight = float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0
    return {
        "output_candidate_rows": int(len(ranking)),
        "output_selected_count": int(len(result)),
        "output_total_weight": total_weight,
        "output_cash_weight": max(0.0, 1.0 - total_weight),
        "output_candidate_date": manifest_value(manifest, "input_candidate_date"),
        "output_top_selected_code": "" if selected.empty else normalize_code(selected["股票代码"].iloc[0]),
        "output_top_selected_name": "" if selected.empty else str(selected["股票名称"].iloc[0]),
        "output_result_file": str(output_dir / STEP6_OUTPUT_FILES["result"]),
    }


def validate_step6(
    step5_experiment_dir: Path,
    step2_experiment_dir: Path,
    output_dir: Path,
    *,
    max_stock_count: int = DEFAULT_MAX_STOCK_COUNT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
) -> dict[str, object]:
    input_metrics = validate_inputs(step5_experiment_dir, step2_experiment_dir)
    output_metrics = validate_outputs(
        output_dir,
        input_metrics=input_metrics,
        max_stock_count=max_stock_count,
        max_per_sector=max_per_sector,
        min_turnover=min_turnover,
    )
    input_metrics.update(output_metrics)
    return input_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-6 refined result outputs.")
    parser.add_argument("--step5-experiment-dir", type=Path, required=True)
    parser.add_argument("--step2-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-stock-count", type=int, default=DEFAULT_MAX_STOCK_COUNT)
    parser.add_argument("--max-per-sector", type=int, default=DEFAULT_MAX_PER_SECTOR)
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step6(
            args.step5_experiment_dir,
            args.step2_experiment_dir,
            args.output_dir,
            max_stock_count=args.max_stock_count,
            max_per_sector=args.max_per_sector,
            min_turnover=args.min_turnover,
        )
    except Step6ValidationError as exc:
        print(f"Step-6 validation failed: {exc}")
        return 1

    print("Step-6 validation passed")
    for key, value in metrics.items():
        if key == "input_candidate_codes":
            print(f"{key}: {len(value)} codes")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
