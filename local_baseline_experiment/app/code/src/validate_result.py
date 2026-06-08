from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULT_COLUMN_MAP = {
    "股票代码": "stock_id",
    "证券代码": "stock_id",
    "代码": "stock_id",
    "权重": "weight",
}


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: RESULT_COLUMN_MAP.get(col, col) for col in df.columns}).copy()


def validate_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_result_columns(df)
    required = {"stock_id", "weight"}
    if not required.issubset(result.columns):
        raise ValueError("result.csv must contain stock_id and weight columns.")
    result = result[["stock_id", "weight"]].copy()
    if len(result) > 5:
        raise ValueError("result.csv cannot contain more than 5 stocks.")
    if result["stock_id"].isna().any() or result["stock_id"].astype(str).str.strip().eq("").any():
        raise ValueError("stock_id cannot be empty.")
    result["stock_id"] = result["stock_id"].astype(str).str.strip()
    if result["stock_id"].duplicated().any():
        raise ValueError("stock_id values cannot be duplicated.")
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    if result["weight"].isna().any():
        raise ValueError("weight must be numeric.")
    if (result["weight"] < 0).any():
        raise ValueError("weight cannot be negative.")
    total_weight = float(result["weight"].sum())
    if total_weight > 1.0 + 1e-12:
        raise ValueError(f"weight sum must be <= 1.0, got {total_weight}.")
    return result


def validate_result_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_prediction_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate THU-BDC2026 result.csv.")
    parser.add_argument("result_path", help="Path to result.csv")
    args = parser.parse_args()
    validated = validate_result_file(args.result_path)
    print(f"valid result: {len(validated)} stocks, weight_sum={validated['weight'].sum():.8f}")


if __name__ == "__main__":
    main()
