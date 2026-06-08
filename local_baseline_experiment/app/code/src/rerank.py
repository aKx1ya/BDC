from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils import format_stock_id_for_output
from validate_result import validate_prediction_frame


def _rank_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    if len(series) == 0:
        return series
    return series.rank(pct=True, method="average", ascending=ascending).fillna(0.5)


def build_candidate_pool(
    latest_features: pd.DataFrame,
    model_scores: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    merged = latest_features.merge(model_scores.drop(columns=["date", "sector"], errors="ignore"), on="stock_id", how="inner")
    merged = merged.sort_values(["model_rank", "stock_id"]).head(int(top_n)).copy()
    merged["trade_date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged.reset_index(drop=True)


def rerank_candidates(candidate_pool: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_pool.empty:
        raise ValueError("Candidate pool is empty; cannot build portfolio.")

    log = candidate_pool.copy()
    log["gate_status"] = "PASS"
    log["removed_reason"] = ""

    min_amount = float(config.get("min_avg_amount_3", 0.0))
    max_drawdown = float(config.get("max_drawdown_20", -0.15))
    max_drop = float(config.get("max_single_drop_20", -0.08))

    liquidity_fail = log.get("avg_amount_3", pd.Series(0, index=log.index)).fillna(0) < min_amount
    drawdown_fail = log.get("drawdown_20", pd.Series(0, index=log.index)).fillna(0) <= max_drawdown
    drop_fail = log.get("max_drop_20", pd.Series(0, index=log.index)).fillna(0) <= max_drop

    log.loc[liquidity_fail, ["gate_status", "removed_reason"]] = ["REMOVE", "low_liquidity"]
    log.loc[drawdown_fail, ["gate_status", "removed_reason"]] = ["REMOVE", "large_drawdown_20"]
    log.loc[drop_fail, ["gate_status", "removed_reason"]] = ["REMOVE", "large_single_drop_20"]

    log["ml_rank_score"] = 1.0 - _rank_score(log["model_rank"], ascending=True) + (1.0 / max(len(log), 1))
    log["sector_momentum_score"] = log.get("sector_momentum_rank_pct", pd.Series(0.5, index=log.index)).fillna(0.5)
    log["price_action_score"] = (
        0.60 * log.get("clv", pd.Series(0, index=log.index)).fillna(0).clip(-1, 1).add(1).div(2)
        + 0.40 * _rank_score(log.get("volume_close_strength", pd.Series(0, index=log.index)).fillna(0), ascending=True)
    )
    log["style_pe_rank_score"] = 0.5
    log["refine_score"] = (
        0.40 * log["ml_rank_score"]
        + 0.30 * log["sector_momentum_score"]
        + 0.20 * log["price_action_score"]
        + 0.10 * log["style_pe_rank_score"]
    )

    selected_rows = []
    sector_counts: Dict[str, int] = {}
    max_per_sector = int(config.get("max_per_sector", 2))
    portfolio_size = int(config.get("portfolio_size", 5))

    passed = log[log["gate_status"] == "PASS"].sort_values(["refine_score", "model_score"], ascending=False)
    for idx, row in passed.iterrows():
        sector = str(row.get("sector", "UNKNOWN"))
        if sector != "UNKNOWN" and sector_counts.get(sector, 0) >= max_per_sector:
            log.loc[idx, ["gate_status", "removed_reason"]] = ["SKIP", "sector_limit"]
            continue
        selected_rows.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_rows) >= portfolio_size:
            break

    if not selected_rows:
        fallback = log.sort_values(["model_rank", "stock_id"]).head(min(portfolio_size, len(log)))
        selected_rows = [row for _, row in fallback.iterrows()]
        log.loc[fallback.index, "removed_reason"] = log.loc[fallback.index, "removed_reason"].replace("", "fallback_selected")

    selected_ids = {str(row["stock_id"]) for row in selected_rows}
    log["final_selected"] = log["stock_id"].astype(str).isin(selected_ids)
    log["correlation_check"] = "not_applied_v1"

    equal_weight = float(config.get("equal_weight", 1.0 / max(portfolio_size, 1)))
    result = pd.DataFrame(
        {
            "stock_id": [format_stock_id_for_output(row["stock_id"]) for row in selected_rows[:portfolio_size]],
            "weight": [equal_weight for _ in selected_rows[:portfolio_size]],
        }
    )
    validate_prediction_frame(result)
    return result, log.sort_values(["final_selected", "refine_score"], ascending=[False, False]).reset_index(drop=True)
