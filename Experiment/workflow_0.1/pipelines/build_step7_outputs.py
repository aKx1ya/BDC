#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKFLOW_ROOT.parents[1]
DEFAULT_STEP6_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step6_workflow_0_1"
    / "outputs"
    / "step6"
)
DEFAULT_EXPERIMENT_DIR = WORKFLOW_ROOT / "experiments" / f"exp_{datetime.now().strftime('%Y%m%d')}_step7_workflow_0_1"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "outputs" / "step7"
DEFAULT_WORKSPACE_DIR = DEFAULT_EXPERIMENT_DIR / "official_scoring_workspace"
DEFAULT_OFFICIAL_SCRIPT_PATH = PROJECT_ROOT / "THU-BDC2026-main" / "test" / "score_self.py"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "THU-BDC2026-main" / "data" / "test.csv"


SCHEMA_VERSION = "workflow_0.1_csv_v1"
EVALUATION_SET_ID = "evaluation_set_v1_official_score"
DEFAULT_TEAM_NAME = "team_name"
VALID_SCORE_MODES = {"freeze-only", "local-score"}


STEP7_FROZEN_RESULT_COLUMNS = ["stock_id", "weight"]


STEP7_SCORE_SUMMARY_COLUMNS = [
    "experiment_id",
    "score_mode",
    "team_name",
    "final_score",
    "result_status",
    "selected_count",
    "total_weight",
    "test_date_start",
    "test_date_end",
    "official_script_path",
    "official_tmp_path",
    "generated_at",
    "note",
]


STEP7_STOCK_CONTRIBUTION_COLUMNS = [
    "stock_id",
    "股票名称",
    "板块划分",
    "weight",
    "open_first",
    "open_last",
    "return_5d_open_to_open",
    "score_contribution",
    "test_date_start",
    "test_date_end",
    "matched_test_rows",
]


STEP7_LEAKAGE_CHECK_COLUMNS = ["检查项", "状态", "说明"]


STEP6_OUTPUT_FILES = {
    "ranking_log": "step6_ranking_log.csv",
    "final_top5": "step6_final_top5.csv",
    "result": "step6_result.csv",
    "manifest": "step6_refine_manifest.csv",
    "leakage_check": "step6_leakage_check.csv",
}


def normalize_code(value: object) -> str:
    text = str(value).strip().replace("sh.", "").replace("sz.", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def manifest_value(manifest: pd.DataFrame, item: str, default: str = "") -> str:
    if {"项目", "说明"} - set(manifest.columns):
        return default
    matched = manifest.loc[manifest["项目"].astype(str) == item, "说明"]
    if matched.empty:
        return default
    return str(matched.iloc[0])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(10)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def read_step6_outputs(step6_output_dir: Path) -> dict[str, pd.DataFrame]:
    step6_output_dir = Path(step6_output_dir)
    return {
        "ranking_log": read_csv(step6_output_dir / STEP6_OUTPUT_FILES["ranking_log"], dtype={"股票代码": str}),
        "final_top5": read_csv(step6_output_dir / STEP6_OUTPUT_FILES["final_top5"], dtype={"股票代码": str}),
        "result": read_csv(step6_output_dir / STEP6_OUTPUT_FILES["result"], dtype={"stock_id": str}),
        "manifest": read_csv(step6_output_dir / STEP6_OUTPUT_FILES["manifest"]),
        "leakage_check": read_csv(step6_output_dir / STEP6_OUTPUT_FILES["leakage_check"]),
    }


def validate_result_frame(result: pd.DataFrame) -> pd.DataFrame:
    if list(result.columns) != STEP7_FROZEN_RESULT_COLUMNS:
        raise ValueError(f"result columns must be {STEP7_FROZEN_RESULT_COLUMNS}, got {list(result.columns)}")
    out = result.copy()
    if len(out) > 5:
        raise ValueError("result cannot contain more than 5 stocks")
    out["stock_id"] = out["stock_id"].map(normalize_code)
    if out["stock_id"].duplicated().any():
        raise ValueError("result stock_id cannot be duplicated")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if out["weight"].isna().any():
        raise ValueError("result weight must be numeric")
    if out["weight"].lt(0).any():
        raise ValueError("result weight cannot be negative")
    if float(out["weight"].sum()) > 1.0000001:
        raise ValueError("result weight sum must be <= 1")
    return out[STEP7_FROZEN_RESULT_COLUMNS]


def prepare_test_data(test_data_path: Path) -> pd.DataFrame:
    test = read_csv(test_data_path, dtype={"股票代码": str})
    required = {"股票代码", "日期", "开盘", "收盘"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    out = test[["股票代码", "日期", "开盘", "收盘"]].copy()
    out["股票代码"] = out["股票代码"].map(normalize_code)
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["开盘"] = pd.to_numeric(out["开盘"], errors="coerce")
    out["收盘"] = pd.to_numeric(out["收盘"], errors="coerce")
    out = out.dropna(subset=["日期", "开盘"])
    if out.empty:
        raise ValueError("test.csv has no valid rows")
    return out


def selected_test_slice(test: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    selected_codes = set(result["stock_id"].map(normalize_code))
    selected = test[test["股票代码"].isin(selected_codes)].copy()
    return selected.sort_values(["股票代码", "日期"])


def test_coverage(selected_test: pd.DataFrame, result: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    counts = selected_test.groupby("股票代码").size().to_dict()
    missing = [code for code in result["stock_id"].map(normalize_code).tolist() if counts.get(code, 0) == 0]
    return missing, {str(key): int(value) for key, value in counts.items()}


def test_dates_are_future(selected_test: pd.DataFrame, candidate_date: str) -> bool:
    if selected_test.empty or not candidate_date:
        return False
    test_start = str(selected_test["日期"].min())
    return test_start > str(candidate_date)


def run_official_score(
    *,
    frozen_result_path: Path,
    test_data_path: Path,
    official_script_path: Path,
    workspace_dir: Path,
) -> tuple[Path, str]:
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    (workspace_dir / "output").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "data").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "temp").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "test").mkdir(parents=True, exist_ok=True)

    shutil.copy2(frozen_result_path, workspace_dir / "output" / "result.csv")
    shutil.copy2(test_data_path, workspace_dir / "data" / "test.csv")
    shutil.copy2(official_script_path, workspace_dir / "test" / "score_self.py")

    completed = subprocess.run(
        [sys.executable, "test/score_self.py"],
        cwd=workspace_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official score_self.py failed: "
            f"returncode={completed.returncode}; stdout={completed.stdout}; stderr={completed.stderr}"
        )
    return workspace_dir / "temp" / "tmp.csv", completed.stdout.strip()


def read_official_score(tmp_path: Path) -> float:
    tmp = read_csv(tmp_path)
    if "Final Score" not in tmp.columns or tmp.empty:
        raise ValueError(f"official tmp.csv missing Final Score: {tmp_path}")
    return float(pd.to_numeric(tmp["Final Score"], errors="coerce").iloc[0])


def build_stock_contribution(
    *,
    result: pd.DataFrame,
    final_top5: pd.DataFrame,
    selected_test: pd.DataFrame,
) -> pd.DataFrame:
    if selected_test.empty:
        return pd.DataFrame(columns=STEP7_STOCK_CONTRIBUTION_COLUMNS)

    name_map = {}
    sector_map = {}
    if {"股票代码", "股票名称", "板块划分"} <= set(final_top5.columns):
        tmp = final_top5.copy()
        tmp["股票代码"] = tmp["股票代码"].map(normalize_code)
        name_map = dict(zip(tmp["股票代码"], tmp["股票名称"]))
        sector_map = dict(zip(tmp["股票代码"], tmp["板块划分"]))

    weight_map = dict(zip(result["stock_id"].map(normalize_code), pd.to_numeric(result["weight"], errors="coerce")))
    rows = []
    for code, group in selected_test.sort_values(["股票代码", "日期"]).groupby("股票代码"):
        group = group.tail(5).sort_values("日期")
        open_first = float(group["开盘"].iloc[0])
        open_last = float(group["开盘"].iloc[-1])
        ret = (open_last - open_first) / open_first
        weight = float(weight_map.get(code, 0.0))
        rows.append(
            {
                "stock_id": code,
                "股票名称": name_map.get(code, ""),
                "板块划分": sector_map.get(code, ""),
                "weight": weight,
                "open_first": open_first,
                "open_last": open_last,
                "return_5d_open_to_open": ret,
                "score_contribution": ret * weight,
                "test_date_start": str(group["日期"].min()),
                "test_date_end": str(group["日期"].max()),
                "matched_test_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=STEP7_STOCK_CONTRIBUTION_COLUMNS).sort_values("stock_id")


def build_score_summary(
    *,
    experiment_id: str,
    mode: str,
    team_name: str,
    final_score: float | None,
    result_status: str,
    result: pd.DataFrame,
    test_date_start: str,
    test_date_end: str,
    official_script_path: Path | None,
    official_tmp_path: Path | None,
    note: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "score_mode": mode,
                "team_name": team_name,
                "final_score": "" if final_score is None else final_score,
                "result_status": result_status,
                "selected_count": int(len(result)),
                "total_weight": float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
                "official_script_path": "" if official_script_path is None else str(official_script_path),
                "official_tmp_path": "" if official_tmp_path is None else str(official_tmp_path),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": note,
            }
        ],
        columns=STEP7_SCORE_SUMMARY_COLUMNS,
    )


def build_manifest(
    *,
    step6_output_dir: Path,
    output_dir: Path,
    frozen_result_path: Path,
    mode: str,
    official_script_path: Path | None,
    test_data_path: Path | None,
    result: pd.DataFrame,
    final_score: float | None,
    input_step6_experiment: str | None,
    note: str,
) -> pd.DataFrame:
    total_weight = float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0
    items = [
        ("schema_version", SCHEMA_VERSION),
        ("evaluation_set_id", EVALUATION_SET_ID),
        ("input_step6_experiment", input_step6_experiment or step6_output_dir.parents[1].name),
        ("input_step6_result_path", str(step6_output_dir / STEP6_OUTPUT_FILES["result"])),
        ("frozen_result_path", str(frozen_result_path)),
        ("score_mode", mode),
        ("official_script_path", "" if official_script_path is None else str(official_script_path)),
        ("test_data_path", "" if test_data_path is None else str(test_data_path)),
        ("selected_count", str(len(result))),
        ("total_weight", f"{total_weight:.10f}"),
        ("final_score", "" if final_score is None else f"{final_score:.10f}"),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        (
            "data_window_note",
            note
            or "正式 Step-7 读取健康 Step-6 输出，先冻结 result.csv，再按模式决定是否读取 test.csv 并进行官方口径评分。",
        ),
        (
            "leakage_control_note",
            "Step-7 在 frozen_result.csv 生成后才允许读取 test.csv；Final Score 只能用于复盘和下一轮实验，不能回改本轮 Step-6。",
        ),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def build_leakage_check(
    *,
    mode: str,
    candidate_date: str,
    test_date_start: str,
    missing_test_codes: list[str],
    test_counts: dict[str, int],
    official_completed: bool,
    final_score: float | None,
    contribution_matches: bool,
) -> pd.DataFrame:
    rows = [
        ("input_step6_success", "PASS", "Step-7 runner/validator 要求读取的 Step-6 运行报告为 SUCCESS。"),
        ("input_step6_leakage_pass", "PASS", "Step-6 leakage_check 必须全部 PASS。"),
        ("result_frozen_before_test_read", "PASS", "先写 step7_frozen_result.csv，再允许读取 test.csv。"),
        ("frozen_result_matches_step6_result", "PASS", "冻结结果必须与 Step-6 result 完全一致。"),
        ("result_schema_valid", "PASS", "frozen_result 仅包含 stock_id,weight。"),
        ("result_stock_count_lte_5", "PASS", "frozen_result 行数不超过 5。"),
        ("result_stock_id_unique", "PASS", "frozen_result stock_id 无重复。"),
        ("result_weight_non_negative", "PASS", "frozen_result 权重均非负。"),
        ("result_weight_sum_lte_1", "PASS", "frozen_result 权重总和不超过 1。"),
        (
            "test_data_read_after_freeze",
            "PASS",
            "freeze-only 模式不读取 test.csv；local-score 模式只在冻结后读取。",
        ),
        ("official_score_not_used_to_modify_step6", "PASS", "Step-7 不修改 Step-6 输出。"),
        ("manifest_leakage_note", "PASS", "manifest 写入 leakage_control_note。"),
    ]
    if mode == "local-score":
        future_ok = bool(test_date_start and candidate_date and test_date_start > candidate_date)
        five_rows_ok = all(count >= 5 for count in test_counts.values()) and not missing_test_codes
        score_ok = final_score is not None and final_score != -999
        rows.extend(
            [
                ("test_data_available", "PASS" if test_date_start else "FAIL", "local-score 模式必须存在可读取的 test.csv。"),
                (
                    "selected_stocks_covered_by_test",
                    "PASS" if not missing_test_codes else "FAIL",
                    "missing=" + ",".join(missing_test_codes),
                ),
                (
                    "each_selected_stock_has_5_test_rows",
                    "PASS" if five_rows_ok else "FAIL",
                    f"counts={test_counts}",
                ),
                (
                    "test_data_is_future_of_candidate_date",
                    "PASS" if future_ok else "FAIL",
                    f"candidate_date={candidate_date}, test_date_start={test_date_start}",
                ),
                (
                    "official_script_completed",
                    "PASS" if official_completed else "FAIL",
                    "官方 score_self.py 必须完成运行。",
                ),
                (
                    "final_score_not_negative_999",
                    "PASS" if score_ok else "FAIL",
                    f"final_score={final_score}",
                ),
                (
                    "stock_contribution_matches_final_score",
                    "PASS" if contribution_matches else "FAIL",
                    "单股贡献求和必须等于 Final Score。",
                ),
            ]
        )
    return pd.DataFrame(rows, columns=STEP7_LEAKAGE_CHECK_COLUMNS)


def build_step7_outputs(
    step6_output_dir: Path,
    output_dir: Path,
    workspace_dir: Path,
    *,
    mode: str = "freeze-only",
    input_step6_experiment: str | None = None,
    team_name: str = DEFAULT_TEAM_NAME,
    official_script_path: Path = DEFAULT_OFFICIAL_SCRIPT_PATH,
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    note: str | None = None,
) -> dict[str, Path]:
    if mode not in VALID_SCORE_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_SCORE_MODES)}, got {mode!r}")
    step6_output_dir = Path(step6_output_dir)
    output_dir = Path(output_dir)
    workspace_dir = Path(workspace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step6 = read_step6_outputs(step6_output_dir)
    result = validate_result_frame(step6["result"])
    candidate_date = manifest_value(step6["manifest"], "input_candidate_date")

    frozen_result = result.copy()
    frozen_result_path = output_dir / "step7_frozen_result.csv"
    write_csv(frozen_result, frozen_result_path)

    final_score: float | None = None
    test_date_start = ""
    test_date_end = ""
    official_tmp_path: Path | None = None
    official_completed = False
    missing_test_codes: list[str] = []
    test_counts: dict[str, int] = {}
    contribution = pd.DataFrame(columns=STEP7_STOCK_CONTRIBUTION_COLUMNS)
    status = "FREEZE_ONLY_SUCCESS"
    summary_note = note or "freeze-only 模式：已冻结 result.csv，未读取 test.csv，未计算 Final Score。"

    if mode == "local-score":
        official_script_path = Path(official_script_path)
        test_data_path = Path(test_data_path)
        if not official_script_path.exists():
            raise FileNotFoundError(f"missing official score script: {official_script_path}")
        if not test_data_path.exists():
            raise FileNotFoundError(f"missing test data: {test_data_path}")
        test = prepare_test_data(test_data_path)
        selected = selected_test_slice(test, result)
        missing_test_codes, test_counts = test_coverage(selected, result)
        test_date_start = "" if selected.empty else str(selected["日期"].min())
        test_date_end = "" if selected.empty else str(selected["日期"].max())
        future_ok = test_dates_are_future(selected, candidate_date)
        enough_rows = bool(test_counts) and all(count >= 5 for count in test_counts.values()) and not missing_test_codes

        if future_ok and enough_rows:
            official_tmp_path, _ = run_official_score(
                frozen_result_path=frozen_result_path,
                test_data_path=test_data_path,
                official_script_path=official_script_path,
                workspace_dir=workspace_dir,
            )
            official_completed = True
            final_score = read_official_score(official_tmp_path)
            contribution = build_stock_contribution(result=result, final_top5=step6["final_top5"], selected_test=selected)
            contribution_sum = float(pd.to_numeric(contribution["score_contribution"], errors="coerce").sum())
            contribution_matches = abs(contribution_sum - final_score) <= 1e-8
            status = "SCORE_SUCCESS" if final_score != -999 and contribution_matches else "FAILED"
            summary_note = note or "local-score 模式：已冻结 result.csv，并按官方 score_self.py 口径完成本地评分。"
        else:
            contribution_matches = False
            status = "FAILED"
            summary_note = (
                note
                or "local-score 模式被健康检查阻止：test.csv 必须覆盖所有入选股票、每只至少 5 条记录，且 test 日期必须晚于 candidate_date。"
            )
    else:
        contribution_matches = True
        official_script_path = None
        test_data_path = None

    score_summary = build_score_summary(
        experiment_id=input_step6_experiment or step6_output_dir.parents[1].name,
        mode=mode,
        team_name=team_name,
        final_score=final_score,
        result_status=status,
        result=result,
        test_date_start=test_date_start,
        test_date_end=test_date_end,
        official_script_path=official_script_path,
        official_tmp_path=official_tmp_path,
        note=summary_note,
    )
    manifest = build_manifest(
        step6_output_dir=step6_output_dir,
        output_dir=output_dir,
        frozen_result_path=frozen_result_path,
        mode=mode,
        official_script_path=official_script_path,
        test_data_path=test_data_path,
        result=result,
        final_score=final_score,
        input_step6_experiment=input_step6_experiment,
        note=summary_note,
    )
    leakage_check = build_leakage_check(
        mode=mode,
        candidate_date=candidate_date,
        test_date_start=test_date_start,
        missing_test_codes=missing_test_codes,
        test_counts=test_counts,
        official_completed=official_completed,
        final_score=final_score,
        contribution_matches=contribution_matches,
    )

    outputs = {
        "frozen_result": frozen_result_path,
        "score_summary": output_dir / "step7_score_summary.csv",
        "stock_contribution": output_dir / "step7_stock_contribution.csv",
        "manifest": output_dir / "step7_score_manifest.csv",
        "leakage_check": output_dir / "step7_leakage_check.csv",
    }
    write_csv(score_summary, outputs["score_summary"])
    write_csv(contribution, outputs["stock_contribution"])
    write_csv(manifest, outputs["manifest"])
    write_csv(leakage_check, outputs["leakage_check"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-7 frozen result and score outputs.")
    parser.add_argument("--step6-output-dir", type=Path, default=DEFAULT_STEP6_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR)
    parser.add_argument("--mode", choices=sorted(VALID_SCORE_MODES), default="freeze-only")
    parser.add_argument("--input-step6-experiment", default=None)
    parser.add_argument("--team-name", default=DEFAULT_TEAM_NAME)
    parser.add_argument("--official-script-path", type=Path, default=DEFAULT_OFFICIAL_SCRIPT_PATH)
    parser.add_argument("--test-data-path", type=Path, default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step7_outputs(
        step6_output_dir=args.step6_output_dir,
        output_dir=args.output_dir,
        workspace_dir=args.workspace_dir,
        mode=args.mode,
        input_step6_experiment=args.input_step6_experiment,
        team_name=args.team_name,
        official_script_path=args.official_script_path,
        test_data_path=args.test_data_path,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    print(f"workspace_dir: {args.workspace_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
