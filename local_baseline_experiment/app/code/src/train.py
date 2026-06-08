from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config import ensure_directories, load_config, path_from_config
from featurework import prepare_training_frame, read_market_data
from models import fit_model_bundle
from utils import set_random_seed


def _training_slice(frame: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    train_window = int(config.get("train_window", 252))
    unique_dates = sorted(frame["date"].drop_duplicates())
    if train_window > 0 and len(unique_dates) > train_window:
        allowed_dates = set(unique_dates[-train_window:])
        return frame[frame["date"].isin(allowed_dates)].copy()
    return frame.copy()


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    ensure_directories(config)
    set_random_seed(int(config.get("random_seed", 2026)))

    train_path = path_from_config(config, "data_path") / str(config.get("train_file", "train.csv"))
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    raw = read_market_data(train_path)
    training_frame, feature_columns = prepare_training_frame(
        raw,
        horizon=int(config.get("prediction_horizon", 5)),
        sequence_length=int(config.get("sequence_length", 60)),
    )
    training_frame = _training_slice(training_frame, config)
    if training_frame.empty:
        raise ValueError("No training rows remain after label/window filtering.")

    bundle = fit_model_bundle(training_frame, feature_columns, seed=int(config.get("random_seed", 2026)))
    model_dir = path_from_config(config, "model_path")
    model_path = model_dir / str(config.get("model_bundle_file", "model_bundle.pkl"))
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    metadata = dict(bundle.metadata)
    metadata["model_path"] = str(model_path)
    metadata["feature_columns"] = feature_columns
    pd.DataFrame([metadata]).to_csv(path_from_config(config, "temp_path") / "training_metadata.csv", index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train THU-BDC2026 ranking pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = run_training(config)
    print(
        "training complete: "
        f"backend={metadata['backend']} rows={metadata['n_rows']} "
        f"features={metadata['n_features']} model={metadata['model_path']}"
    )


if __name__ == "__main__":
    main()
