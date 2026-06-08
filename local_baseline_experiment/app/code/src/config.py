from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path": "app/data",
    "model_path": "app/model",
    "output_path": "app/output",
    "temp_path": "app/temp",
    "train_file": "train.csv",
    "test_file": "test.csv",
    "result_file": "result.csv",
    "sequence_length": 60,
    "prediction_horizon": 5,
    "train_window": 252,
    "gap": 5,
    "candidate_top_n": 30,
    "portfolio_size": 5,
    "equal_weight": 0.2,
    "random_seed": 2026,
    "min_avg_amount_3": 0.0,
    "max_drawdown_20": -0.15,
    "max_single_drop_20": -0.08,
    "max_per_sector": 2,
    "full_exposure": 1.0,
    "model_bundle_file": "model_bundle.pkl",
}


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        data[key.strip()] = _parse_scalar(raw_value)
    return data


def load_config(config_path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    path = Path(config_path) if config_path else Path("config.yaml")
    if path.exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file {path} must contain a mapping.")
            config.update(loaded)
        except ModuleNotFoundError:
            config.update(_load_simple_yaml(path))
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    for key in ["data_path", "model_path", "output_path", "temp_path"]:
        Path(str(config[key])).mkdir(parents=True, exist_ok=True)


def path_from_config(config: Dict[str, Any], base_key: str, file_key: str | None = None) -> Path:
    base = Path(str(config[base_key]))
    return base / str(config[file_key]) if file_key else base
