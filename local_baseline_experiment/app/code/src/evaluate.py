from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import canonical_stock_id
from validate_result import validate_prediction_frame


EVAL_COLUMN_MAP = {
    "股票代码": "stock_id",
    "证券代码": "stock_id",
    "代码": "stock_id",
    "日期": "date",
    "交易日期": "date",
    "开盘": "open",
    "开盘价": "open",
}


def _normalize_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={col: EVAL_COLUMN_MAP.get(col, col) for col in df.columns}).copy()
    missing = [col for col in ["stock_id", "date", "open"] if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required evaluation columns: {missing}")
    normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["open"] = pd.to_numeric(normalized["open"], errors="coerce")
    normalized = normalized.dropna(subset=["stock_id", "date", "open"])
    return normalized.sort_values(["stock_id", "date"]).reset_index(drop=True)


def calculate_weighted_score(result_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    result = validate_prediction_frame(result_df)
    test = _normalize_eval_columns(test_df)
    result = result.copy()
    test = test.copy()
    result["match_stock_id"] = result["stock_id"].map(canonical_stock_id)
    test["match_stock_id"] = test["stock_id"].map(canonical_stock_id)
    details = []
    score = 0.0

    for _, row in result.iterrows():
        stock_id = str(row["stock_id"])
        match_stock_id = str(row["match_stock_id"])
        weight = float(row["weight"])
        stock_rows = test[test["match_stock_id"] == match_stock_id].sort_values("date").tail(5)
        if stock_rows.empty:
            raise ValueError(f"No test rows found for selected stock {stock_id}.")
        start = stock_rows.iloc[0]
        end = stock_rows.iloc[-1]
        stock_return = (float(end["open"]) - float(start["open"])) / float(start["open"])
        contribution = stock_return * weight
        score += contribution
        details.append(
            {
                "stock_id": stock_id,
                "weight": weight,
                "open_first": float(start["open"]),
                "open_last": float(end["open"]),
                "return_i": stock_return,
                "contribution": contribution,
            }
        )

    return float(score), pd.DataFrame(details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate result.csv with official open-price weighted return.")
    parser.add_argument("--result", default="app/output/result.csv", help="Path to result.csv")
    parser.add_argument("--test", default="app/data/test.csv", help="Path to local future test.csv")
    parser.add_argument("--temp", default="app/temp", help="Directory for evaluation outputs")
    args = parser.parse_args()

    result = pd.read_csv(args.result)
    test = pd.read_csv(args.test)
    score, detail = calculate_weighted_score(result, test)

    temp = Path(args.temp)
    temp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Team Name": "THU-BDC2026-local", "Final Score": score}]).to_csv(
        temp / "tmp.csv", index=False, encoding="utf-8"
    )
    detail.to_csv(temp / "evaluation_detail.csv", index=False, encoding="utf-8")
    print(f"Final Score: {score:.10f}")


if __name__ == "__main__":
    main()
