from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config import ensure_directories, load_config, path_from_config
from featurework import latest_prediction_frame, read_market_data
from rerank import build_candidate_pool, rerank_candidates
from utils import set_random_seed
from validate_result import validate_prediction_frame


def _load_bundle(config: Dict[str, Any]):
    model_path = path_from_config(config, "model_path") / str(config.get("model_bundle_file", "model_bundle.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}. Run train.sh first.")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def run_prediction(config: Dict[str, Any]) -> pd.DataFrame:
    ensure_directories(config)
    set_random_seed(int(config.get("random_seed", 2026)))
    bundle = _load_bundle(config)

    train_path = path_from_config(config, "data_path") / str(config.get("train_file", "train.csv"))
    if not train_path.exists():
        raise FileNotFoundError(f"Prediction data not found: {train_path}")

    raw = read_market_data(train_path)
    latest = latest_prediction_frame(raw)
    scores = bundle.predict_scores(latest)
    candidates = build_candidate_pool(latest, scores, top_n=int(config.get("candidate_top_n", 30)))
    result, ranking_log = rerank_candidates(candidates, config)
    result = validate_prediction_frame(result)

    temp_dir = path_from_config(config, "temp_path")
    output_dir = path_from_config(config, "output_path")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(temp_dir / "candidate_top30.csv", index=False, encoding="utf-8")
    ranking_log.to_csv(temp_dir / "ranking_log.csv", index=False, encoding="utf-8")
    result.to_csv(output_dir / str(config.get("result_file", "result.csv")), index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict THU-BDC2026 result.csv.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_prediction(config)
    print(f"prediction complete: {len(result)} stocks, weight_sum={result['weight'].sum():.8f}")


if __name__ == "__main__":
    main()
