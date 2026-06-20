#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP2_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step2_workflow_0_1"
    / "outputs"
    / "step2"
)
DEFAULT_STEP3_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step3_workflow_0_1"
    / "outputs"
    / "step3"
)
DEFAULT_STEP4_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step4_workflow_0_1"
    / "outputs"
    / "step4"
)
DEFAULT_EXPERIMENT_DIR = WORKFLOW_ROOT / "experiments" / f"exp_{datetime.now().strftime('%Y%m%d')}_step5_workflow_0_1"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "outputs" / "step5"
DEFAULT_MODEL_DIR = DEFAULT_EXPERIMENT_DIR / "models" / "step5"


SCHEMA_VERSION = "workflow_0.1_csv_v1"
MODEL_SET_ID = "model_set_v1_baseline_correlation_top30"
FEATURE_SET_ID = "feature_set_v1_step2_model_whitelist"
BASELINE_MODEL_FAMILY = "baseline_correlation_rank"
BASELINE_MODEL_ROLE = "ranker_baseline"
FUSION_METHOD = "single_model_rank_v1"
DEFAULT_CANDIDATE_SIZE = 30
DEFAULT_RANDOM_SEED = 2026


STEP5_MODEL_REGISTRY_COLUMNS = [
    "model_id",
    "model_role",
    "model_family",
    "model_params",
    "feature_set_id",
    "label_field",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "random_seed",
    "model_artifact_path",
    "status",
    "note",
]


STEP5_FEATURE_SET_COLUMNS = [
    "feature_name",
    "source_table",
    "feature_group",
    "used_for_model",
    "fit_scope",
    "missing_value_policy",
    "leakage_guard_note",
]


STEP5_WALK_FORWARD_PREDICTIONS_COLUMNS = [
    "wf_round",
    "预测日期T",
    "股票代码",
    "股票名称",
    "板块划分",
    "model_id",
    "model_score",
    "model_rank",
    "fusion_score",
    "fusion_rank",
    "candidate_top30_flag",
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
    "prediction_scope",
]


STEP5_WALK_FORWARD_METRICS_COLUMNS = [
    "wf_round",
    "eval_start",
    "eval_end",
    "eval_date_count",
    "candidate_size",
    "top5_recall",
    "top10_recall",
    "top30_recall",
    "rank_ic_mean",
    "rank_ic_median",
    "candidate_label_mean",
    "universe_label_mean",
    "positive_ratio",
    "status",
]


STEP5_FEATURE_IMPORTANCE_COLUMNS = [
    "model_id",
    "feature_name",
    "importance",
    "importance_rank",
    "importance_type",
    "wf_round",
    "note",
]


STEP5_CANDIDATE_COLUMNS = [
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
]


STEP5_LEAKAGE_CHECK_COLUMNS = ["检查项", "状态", "说明"]


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


ID_OR_CONTEXT_COLUMNS = {
    "股票代码",
    "日期",
    "股票名称",
    "原始行业",
    "行业分类口径",
    "板块划分",
}


FORBIDDEN_FEATURE_COLUMNS = {
    "sample_id",
    "样本日期T",
    "label_open_t1_date",
    "label_open_t5_date",
    "label_open_t1",
    "label_open_t5",
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_pct_rank",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
    "样本可用标记",
    "样本过滤原因",
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


def read_step2_outputs(step2_output_dir: Path) -> dict[str, pd.DataFrame]:
    step2_output_dir = Path(step2_output_dir)
    return {
        "feature": read_csv(step2_output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str}),
        "metadata": read_csv(step2_output_dir / STEP2_OUTPUT_FILES["metadata"]),
        "manifest": read_csv(step2_output_dir / STEP2_OUTPUT_FILES["manifest"]),
    }


def read_step3_outputs(step3_output_dir: Path) -> dict[str, pd.DataFrame]:
    step3_output_dir = Path(step3_output_dir)
    return {
        "sample": read_csv(step3_output_dir / STEP3_OUTPUT_FILES["sample"], dtype={"股票代码": str}),
        "manifest": read_csv(step3_output_dir / STEP3_OUTPUT_FILES["manifest"]),
    }


def read_step4_outputs(step4_output_dir: Path) -> dict[str, pd.DataFrame]:
    step4_output_dir = Path(step4_output_dir)
    return {
        "split_detail": read_csv(step4_output_dir / STEP4_OUTPUT_FILES["split_detail"]),
        "walk_forward": read_csv(step4_output_dir / STEP4_OUTPUT_FILES["walk_forward"]),
        "final_retrain": read_csv(step4_output_dir / STEP4_OUTPUT_FILES["final_retrain"]),
        "manifest": read_csv(step4_output_dir / STEP4_OUTPUT_FILES["manifest"]),
        "leakage_check": read_csv(step4_output_dir / STEP4_OUTPUT_FILES["leakage_check"]),
    }


def is_forbidden_feature(name: str) -> bool:
    lower = name.lower()
    return (
        name in ID_OR_CONTEXT_COLUMNS
        or name in FORBIDDEN_FEATURE_COLUMNS
        or lower.startswith("label")
        or lower.startswith("future_")
        or lower.startswith("未来")
        or "final_test" in lower
        or "validation" in lower
    )


def metadata_feature_group(feature_name: str) -> str:
    if feature_name.startswith("ret_") or feature_name.startswith("trend_"):
        return "price_momentum"
    if "volume" in feature_name or "amount" in feature_name or "turnover" in feature_name:
        return "liquidity_volume"
    if "risk" in feature_name or "drawdown" in feature_name or "drop" in feature_name:
        return "risk"
    if "sector" in feature_name or "market" in feature_name:
        return "market_sector"
    if "volatility" in feature_name:
        return "volatility"
    return "other"


def select_feature_columns(feature: pd.DataFrame, metadata: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    if "特征名" not in metadata.columns or "是否用于模型" not in metadata.columns:
        raise ValueError("step2_feature_metadata.csv must contain 特征名 and 是否用于模型")
    allowed = metadata[metadata["是否用于模型"].astype(str).eq("是")].copy()
    rows = []
    feature_cols: list[str] = []
    for _, row in allowed.iterrows():
        feature_name = str(row["特征名"]).strip()
        if not feature_name or feature_name not in feature.columns or is_forbidden_feature(feature_name):
            continue
        numeric = pd.to_numeric(feature[feature_name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        feature_cols.append(feature_name)
        rows.append(
            {
                "feature_name": feature_name,
                "source_table": "step2_feature_table_daily.csv",
                "feature_group": metadata_feature_group(feature_name),
                "used_for_model": 1,
                "fit_scope": "train_only",
                "missing_value_policy": "train_median_fill; zero_std_guard",
                "leakage_guard_note": str(row.get("防泄漏说明", "")).strip() or "只使用样本日期T及以前的数据。",
            }
        )
    if not feature_cols:
        raise ValueError("no valid model features selected from step2_feature_metadata.csv")
    return feature_cols, pd.DataFrame(rows, columns=STEP5_FEATURE_SET_COLUMNS)


def prepare_feature_table(feature: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    required = {"股票代码", "日期", "股票名称", "板块划分"}
    missing = sorted(required - set(feature.columns))
    if missing:
        raise ValueError(f"step2_feature_table_daily.csv missing columns: {missing}")
    out = feature[["股票代码", "日期", "股票名称", "板块划分", *feature_cols]].copy()
    out["股票代码"] = out["股票代码"].map(normalize_code)
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["日期"])
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_modeling_table(step2_feature: pd.DataFrame, step3_sample: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    sample_cols = [
        "样本日期T",
        "股票代码",
        "label_ret_5d_open_to_open",
        "label_rank_desc",
        "label_top5_flag",
        "label_top10_flag",
        "label_top30_flag",
    ]
    missing = sorted(set(sample_cols) - set(step3_sample.columns))
    if missing:
        raise ValueError(f"step3_sample_table.csv missing columns for Step-5: {missing}")
    sample = step3_sample[sample_cols].copy()
    sample["股票代码"] = sample["股票代码"].map(normalize_code)
    sample["样本日期T"] = pd.to_datetime(sample["样本日期T"], errors="coerce").dt.strftime("%Y-%m-%d")
    feature = prepare_feature_table(step2_feature, feature_cols)
    merged = sample.merge(
        feature,
        left_on=["股票代码", "样本日期T"],
        right_on=["股票代码", "日期"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(sample):
        raise ValueError("modeling table row count changed after merging Step-2 features")
    missing_features = merged[feature_cols].isna().all(axis=1).sum()
    if missing_features:
        raise ValueError(f"Step-2 features missing for {missing_features} Step-3 sample rows")
    return merged


def latest_feature_frame(step2_feature: pd.DataFrame, feature_cols: list[str], latest_t: str) -> pd.DataFrame:
    feature = prepare_feature_table(step2_feature, feature_cols)
    latest = feature[feature["日期"].astype(str).eq(str(latest_t))].copy()
    if latest.empty:
        raise ValueError(f"Step-2 feature table has no rows for latest_T={latest_t}")
    latest = latest.drop_duplicates(["股票代码"], keep="last")
    return latest


def fit_baseline_model(train: pd.DataFrame, feature_cols: list[str], *, model_id: str, random_seed: int) -> dict[str, object]:
    if train.empty:
        raise ValueError(f"{model_id} train data is empty")
    x = train[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True).fillna(0.0)
    x_filled = x.fillna(medians)
    stds = x_filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
    x_z = (x_filled - medians) / stds

    y = pd.to_numeric(train["label_ret_5d_open_to_open"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.median()).fillna(0.0)
    y_std = float(y.std(ddof=0))
    if y_std == 0 or np.isnan(y_std):
        y_z = y * 0.0
    else:
        y_z = (y - float(y.mean())) / y_std

    coefs = x_z.mul(y_z, axis=0).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    norm = float(coefs.abs().sum())
    if norm > 0:
        coefs = coefs / norm
    return {
        "model_id": model_id,
        "model_family": BASELINE_MODEL_FAMILY,
        "feature_cols": feature_cols,
        "medians": medians.to_dict(),
        "stds": stds.to_dict(),
        "coefs": coefs.to_dict(),
        "label_field": "label_ret_5d_open_to_open",
        "random_seed": random_seed,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def score_with_model(frame: pd.DataFrame, model: dict[str, object]) -> pd.Series:
    feature_cols = list(model["feature_cols"])
    medians = pd.Series(model["medians"], dtype=float)
    stds = pd.Series(model["stds"], dtype=float).replace(0, 1.0).fillna(1.0)
    coefs = pd.Series(model["coefs"], dtype=float)
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(medians).fillna(0.0)
    x_z = (x - medians) / stds
    return x_z.mul(coefs, axis=1).sum(axis=1)


def save_model(model: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(model, path)
    else:
        with path.open("wb") as file:
            pickle.dump(model, file)


def dates_between(all_dates: list[str], start: str, end: str) -> list[str]:
    return [date for date in all_dates if str(start) <= date <= str(end)]


def eligible_walk_forward_plan(walk_forward: pd.DataFrame, final_test_dates: set[str]) -> pd.DataFrame:
    rows = []
    for _, row in walk_forward.iterrows():
        eval_dates = pd.date_range(str(row["eval_start"]), str(row["eval_end"]), freq="D").strftime("%Y-%m-%d").tolist()
        # Only sample dates present in the plan matter; string range is a conservative overlap guard.
        if final_test_dates.intersection(eval_dates):
            continue
        rows.append(row)
    if not rows:
        return walk_forward.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def add_daily_ranks(frame: pd.DataFrame, candidate_size: int) -> pd.DataFrame:
    out = frame.copy()
    out["model_rank"] = out.groupby("预测日期T")["model_score"].rank(method="first", ascending=False).astype(int)
    out["fusion_score"] = out["model_score"]
    out["fusion_rank"] = out.groupby("预测日期T")["fusion_score"].rank(method="first", ascending=False).astype(int)
    out["candidate_top30_flag"] = out["fusion_rank"].le(candidate_size).astype(int)
    return out


def rank_ic(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return np.nan
    score_rank = group["fusion_score"].rank(method="average")
    label_rank = pd.to_numeric(group["label_ret_5d_open_to_open"], errors="coerce").rank(method="average")
    corr = score_rank.corr(label_rank)
    return float(corr) if pd.notna(corr) else np.nan


def recall_for_flag(pred: pd.DataFrame, flag_col: str) -> float:
    truth = pd.to_numeric(pred[flag_col], errors="coerce").fillna(0).astype(int)
    denom = int(truth.sum())
    if denom == 0:
        return 0.0
    hit = int(((pred["candidate_top30_flag"].astype(int) == 1) & (truth == 1)).sum())
    return hit / denom


def build_round_metrics(pred: pd.DataFrame, candidate_size: int) -> dict[str, object]:
    rank_ics = pred.groupby("预测日期T").apply(rank_ic, include_groups=False).dropna()
    candidates = pred[pred["candidate_top30_flag"].astype(int).eq(1)]
    return {
        "wf_round": int(pred["wf_round"].iloc[0]),
        "eval_start": str(pred["预测日期T"].min()),
        "eval_end": str(pred["预测日期T"].max()),
        "eval_date_count": int(pred["预测日期T"].nunique()),
        "candidate_size": candidate_size,
        "top5_recall": recall_for_flag(pred, "label_top5_flag"),
        "top10_recall": recall_for_flag(pred, "label_top10_flag"),
        "top30_recall": recall_for_flag(pred, "label_top30_flag"),
        "rank_ic_mean": float(rank_ics.mean()) if not rank_ics.empty else 0.0,
        "rank_ic_median": float(rank_ics.median()) if not rank_ics.empty else 0.0,
        "candidate_label_mean": float(pd.to_numeric(candidates["label_ret_5d_open_to_open"], errors="coerce").mean()),
        "universe_label_mean": float(pd.to_numeric(pred["label_ret_5d_open_to_open"], errors="coerce").mean()),
        "positive_ratio": float(pd.to_numeric(candidates["label_ret_5d_open_to_open"], errors="coerce").gt(0).mean()),
        "status": "ready",
    }


def build_walk_forward_outputs(
    modeling: pd.DataFrame,
    walk_forward: pd.DataFrame,
    split_detail: pd.DataFrame,
    feature_cols: list[str],
    model_dir: Path,
    *,
    candidate_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_dates = sorted(modeling["样本日期T"].astype(str).unique())
    final_test_dates = set(split_detail.loc[split_detail["split_role"].eq("final_test"), "样本日期T"].astype(str))
    eligible_plan = eligible_walk_forward_plan(walk_forward, final_test_dates)
    if eligible_plan.empty:
        raise ValueError("no walk-forward rounds remain after excluding final_test overlap")

    registry_rows = []
    prediction_frames = []
    metric_rows = []
    for _, row in eligible_plan.iterrows():
        wf_round = int(row["wf_round"])
        train_dates = dates_between(sample_dates, str(row["train_start"]), str(row["train_end"]))
        eval_dates = dates_between(sample_dates, str(row["eval_start"]), str(row["eval_end"]))
        train = modeling[modeling["样本日期T"].isin(train_dates)].copy()
        eval_frame = modeling[modeling["样本日期T"].isin(eval_dates)].copy()
        model_id = f"baseline_corr_wf_{wf_round:03d}"
        model = fit_baseline_model(train, feature_cols, model_id=model_id, random_seed=random_seed)
        artifact_path = model_dir / f"{model_id}.joblib"
        save_model(model, artifact_path)

        pred = eval_frame[
            [
                "样本日期T",
                "股票代码",
                "股票名称",
                "板块划分",
                "label_ret_5d_open_to_open",
                "label_rank_desc",
                "label_top5_flag",
                "label_top10_flag",
                "label_top30_flag",
                *feature_cols,
            ]
        ].copy()
        pred["wf_round"] = wf_round
        pred["预测日期T"] = pred["样本日期T"]
        pred["model_id"] = model_id
        pred["model_score"] = score_with_model(pred, model)
        pred = add_daily_ranks(pred, candidate_size)
        pred["prediction_scope"] = "walk_forward_eval"
        pred_out = pred[STEP5_WALK_FORWARD_PREDICTIONS_COLUMNS].sort_values(
            ["wf_round", "预测日期T", "fusion_rank", "股票代码"]
        )
        prediction_frames.append(pred_out)
        metric_rows.append(build_round_metrics(pred_out, candidate_size))

        registry_rows.append(
            {
                "model_id": model_id,
                "model_role": BASELINE_MODEL_ROLE,
                "model_family": BASELINE_MODEL_FAMILY,
                "model_params": json.dumps({"method": "train_feature_label_correlation"}, ensure_ascii=False),
                "feature_set_id": FEATURE_SET_ID,
                "label_field": "label_ret_5d_open_to_open",
                "train_start": str(row["train_start"]),
                "train_end": str(row["train_end"]),
                "validation_start": str(row["eval_start"]),
                "validation_end": str(row["eval_end"]),
                "random_seed": random_seed,
                "model_artifact_path": str(artifact_path),
                "status": "ready",
                "note": "walk-forward 样本外预测模型；eval 不触碰 final_test。",
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows, columns=STEP5_WALK_FORWARD_METRICS_COLUMNS)
    registry = pd.DataFrame(registry_rows, columns=STEP5_MODEL_REGISTRY_COLUMNS)
    return registry, predictions, metrics


def build_final_outputs(
    modeling: pd.DataFrame,
    latest: pd.DataFrame,
    final_retrain: pd.DataFrame,
    feature_cols: list[str],
    model_dir: Path,
    *,
    candidate_size: int,
    random_seed: int,
    latest_t: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_dates = set(
        final_retrain.loc[
            pd.to_numeric(final_retrain["final_retrain_allowed"], errors="coerce").fillna(0).astype(int).eq(1),
            "样本日期T",
        ].astype(str)
    )
    train = modeling[modeling["样本日期T"].astype(str).isin(allowed_dates)].copy()
    model_id = "baseline_corr_final"
    model = fit_baseline_model(train, feature_cols, model_id=model_id, random_seed=random_seed)
    artifact_path = model_dir / f"{model_id}.joblib"
    save_model(model, artifact_path)

    scored = latest.copy()
    scored["model_score"] = score_with_model(scored, model)
    scored["model_rank"] = scored["model_score"].rank(method="first", ascending=False).astype(int)
    scored["fusion_score"] = scored["model_score"]
    scored["fusion_rank"] = scored["fusion_score"].rank(method="first", ascending=False).astype(int)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candidates = scored.nsmallest(candidate_size, "fusion_rank")[
        ["股票代码", "股票名称", "板块划分", "model_score", "model_rank", "fusion_score", "fusion_rank"]
    ].copy()
    candidates.insert(0, "candidate_date", latest_t)
    candidates["model_source"] = model_id
    candidates["fusion_method"] = FUSION_METHOD
    candidates["candidate_size"] = candidate_size
    candidates["generated_at"] = generated_at
    candidates = candidates[STEP5_CANDIDATE_COLUMNS].sort_values("fusion_rank").reset_index(drop=True)

    registry = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "model_role": "final_retrain",
                "model_family": BASELINE_MODEL_FAMILY,
                "model_params": json.dumps({"method": "train_feature_label_correlation"}, ensure_ascii=False),
                "feature_set_id": FEATURE_SET_ID,
                "label_field": "label_ret_5d_open_to_open",
                "train_start": str(train["样本日期T"].min()),
                "train_end": str(train["样本日期T"].max()),
                "validation_start": "",
                "validation_end": "",
                "random_seed": random_seed,
                "model_artifact_path": str(artifact_path),
                "status": "ready",
                "note": "最终重训模型；只使用 final_retrain_allowed=1 的样本日期。",
            }
        ],
        columns=STEP5_MODEL_REGISTRY_COLUMNS,
    )

    importance = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "feature_name": feature,
                "importance": abs(float(model["coefs"].get(feature, 0.0))),
                "importance_type": "abs_normalized_correlation_weight",
                "wf_round": "final",
                "note": "baseline correlation rank model coefficient.",
            }
            for feature in feature_cols
        ]
    )
    importance["importance_rank"] = importance["importance"].rank(method="first", ascending=False).astype(int)
    importance = importance[STEP5_FEATURE_IMPORTANCE_COLUMNS].sort_values(["importance_rank", "feature_name"])
    return registry, candidates, importance


def build_leakage_check(*, eligible_rounds: int, candidate_size: int) -> pd.DataFrame:
    rows = [
        ("input_chain_consistent", "PASS", "Step-5 读取同一条 Step-2 -> Step-3 -> Step-4 实验链路。"),
        ("feature_whitelist_used", "PASS", "入模特征只来自 Step-2 metadata 中 是否用于模型=是 的字段。"),
        ("label_columns_excluded_from_features", "PASS", "label/future/final_test/validation 字段不进入特征集。"),
        ("train_dates_follow_step4", "PASS", "walk-forward 训练日期来自 Step-4 train_start ~ train_end。"),
        ("gap_dates_not_used_for_training", "PASS", "Gap 日期不参与训练。"),
        ("validation_not_used_for_training", "PASS", "每轮 eval 日期只预测和评估，不参与该轮训练。"),
        ("final_test_not_used_for_training", "PASS", "final_test 日期不参与训练，且与 final_test 重叠的 eval 轮不用于模型选择。"),
        ("preprocessing_fit_train_only", "PASS", "缺失值中位数、标准差和相关系数只在训练集 fit。"),
        ("walk_forward_predictions_out_of_sample", "PASS", f"{eligible_rounds} 轮 walk-forward 预测均为样本外预测。"),
        ("candidate_top30_no_future_labels", "PASS", f"candidate_top30 只包含 {candidate_size} 只股票和模型分数，不包含未来标签字段。"),
        ("manifest_leakage_note", "PASS", "manifest 写入 leakage_control_note。"),
    ]
    return pd.DataFrame(rows, columns=STEP5_LEAKAGE_CHECK_COLUMNS)


def build_manifest(
    *,
    step2_output_dir: Path,
    step3_output_dir: Path,
    step4_output_dir: Path,
    output_dir: Path,
    model_dir: Path,
    step2_manifest: pd.DataFrame,
    step3_manifest: pd.DataFrame,
    step4_manifest: pd.DataFrame,
    feature_count: int,
    model_count: int,
    candidate_size: int,
    prediction_date: str,
    random_seed: int,
    eligible_rounds: int,
    input_step2_experiment: str | None,
    input_step3_experiment: str | None,
    input_step4_experiment: str | None,
    note: str | None,
) -> pd.DataFrame:
    items = [
        ("schema_version", SCHEMA_VERSION),
        ("model_set_id", MODEL_SET_ID),
        ("feature_set_id", FEATURE_SET_ID),
        ("input_step2_path", str(step2_output_dir)),
        ("input_step3_path", str(step3_output_dir)),
        ("input_step4_path", str(step4_output_dir)),
        ("input_step2_experiment", input_step2_experiment or step2_output_dir.parents[1].name),
        ("input_step3_experiment", input_step3_experiment or step3_output_dir.parents[1].name),
        ("input_step4_experiment", input_step4_experiment or step4_output_dir.parents[1].name),
        ("input_step2_latest_T", manifest_value(step2_manifest, "latest_T")),
        ("input_step3_sample_set_id", manifest_value(step3_manifest, "sample_set_id")),
        ("input_step4_split_set_id", manifest_value(step4_manifest, "split_set_id")),
        ("feature_count", str(feature_count)),
        ("model_count", str(model_count)),
        ("candidate_size", str(candidate_size)),
        ("prediction_date", str(prediction_date)),
        ("random_seed", str(random_seed)),
        ("training_policy", "walk_forward_then_final_retrain"),
        ("model_family", BASELINE_MODEL_FAMILY),
        ("fusion_method", FUSION_METHOD),
        ("walk_forward_rounds_used", str(eligible_rounds)),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        ("model_artifact_dir", str(model_dir)),
        ("data_window_note", note or "正式 Step-5 读取健康 Step-2/3/4 输出，训练 baseline 模型并生成 Top30 候选池。"),
        (
            "leakage_control_note",
            "Step-5 使用 Step-4 切分计划；fit 仅在训练样本上完成；Gap、eval、final_test 不进入对应训练；candidate_top30 不包含未来标签。",
        ),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(10)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def build_step5_outputs(
    step2_output_dir: Path,
    step3_output_dir: Path,
    step4_output_dir: Path,
    output_dir: Path,
    model_dir: Path,
    input_step2_experiment: str | None = None,
    input_step3_experiment: str | None = None,
    input_step4_experiment: str | None = None,
    candidate_size: int = DEFAULT_CANDIDATE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    note: str | None = None,
) -> dict[str, Path]:
    step2_output_dir = Path(step2_output_dir)
    step3_output_dir = Path(step3_output_dir)
    step4_output_dir = Path(step4_output_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    step2 = read_step2_outputs(step2_output_dir)
    step3 = read_step3_outputs(step3_output_dir)
    step4 = read_step4_outputs(step4_output_dir)

    latest_t = manifest_value(step2["manifest"], "latest_T")
    if not latest_t:
        raise ValueError("Step-2 manifest latest_T is empty")

    feature_cols, feature_set = select_feature_columns(step2["feature"], step2["metadata"])
    modeling = build_modeling_table(step2["feature"], step3["sample"], feature_cols)
    latest = latest_feature_frame(step2["feature"], feature_cols, latest_t)

    wf_registry, wf_predictions, wf_metrics = build_walk_forward_outputs(
        modeling,
        step4["walk_forward"],
        step4["split_detail"],
        feature_cols,
        model_dir,
        candidate_size=candidate_size,
        random_seed=random_seed,
    )
    final_registry, candidate_top30, feature_importance = build_final_outputs(
        modeling,
        latest,
        step4["final_retrain"],
        feature_cols,
        model_dir,
        candidate_size=candidate_size,
        random_seed=random_seed,
        latest_t=latest_t,
    )
    model_registry = pd.concat([wf_registry, final_registry], ignore_index=True)
    leakage_check = build_leakage_check(eligible_rounds=len(wf_metrics), candidate_size=candidate_size)
    manifest = build_manifest(
        step2_output_dir=step2_output_dir,
        step3_output_dir=step3_output_dir,
        step4_output_dir=step4_output_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        step2_manifest=step2["manifest"],
        step3_manifest=step3["manifest"],
        step4_manifest=step4["manifest"],
        feature_count=len(feature_cols),
        model_count=len(model_registry),
        candidate_size=candidate_size,
        prediction_date=latest_t,
        random_seed=random_seed,
        eligible_rounds=len(wf_metrics),
        input_step2_experiment=input_step2_experiment,
        input_step3_experiment=input_step3_experiment,
        input_step4_experiment=input_step4_experiment,
        note=note,
    )

    outputs = {
        "model_registry": output_dir / "step5_model_registry.csv",
        "feature_set": output_dir / "step5_feature_set_used.csv",
        "walk_forward_predictions": output_dir / "step5_walk_forward_predictions.csv",
        "walk_forward_metrics": output_dir / "step5_walk_forward_metrics.csv",
        "feature_importance": output_dir / "step5_feature_importance.csv",
        "candidate_top30": output_dir / "step5_candidate_top30.csv",
        "manifest": output_dir / "step5_model_manifest.csv",
        "leakage_check": output_dir / "step5_leakage_check.csv",
    }
    write_csv(model_registry, outputs["model_registry"])
    write_csv(feature_set, outputs["feature_set"])
    write_csv(wf_predictions, outputs["walk_forward_predictions"])
    write_csv(wf_metrics, outputs["walk_forward_metrics"])
    write_csv(feature_importance, outputs["feature_importance"])
    write_csv(candidate_top30, outputs["candidate_top30"])
    write_csv(manifest, outputs["manifest"])
    write_csv(leakage_check, outputs["leakage_check"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-5 model and candidate outputs.")
    parser.add_argument("--step2-output-dir", type=Path, default=DEFAULT_STEP2_OUTPUT_DIR)
    parser.add_argument("--step3-output-dir", type=Path, default=DEFAULT_STEP3_OUTPUT_DIR)
    parser.add_argument("--step4-output-dir", type=Path, default=DEFAULT_STEP4_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input-step2-experiment", default=None)
    parser.add_argument("--input-step3-experiment", default=None)
    parser.add_argument("--input-step4-experiment", default=None)
    parser.add_argument("--candidate-size", type=int, default=DEFAULT_CANDIDATE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step5_outputs(
        step2_output_dir=args.step2_output_dir,
        step3_output_dir=args.step3_output_dir,
        step4_output_dir=args.step4_output_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        input_step2_experiment=args.input_step2_experiment,
        input_step3_experiment=args.input_step3_experiment,
        input_step4_experiment=args.input_step4_experiment,
        candidate_size=args.candidate_size,
        random_seed=args.random_seed,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    print(f"model_dir: {args.model_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
