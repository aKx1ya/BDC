from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class CorrelationRegressor:
    """Deterministic fallback model using univariate label correlations."""

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CorrelationRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = np.nan_to_num((x - self.mean_) / self.scale_)
        y_centered = y - np.nanmean(y)
        denom = np.sqrt(np.sum(z**2, axis=0) * np.sum(y_centered**2))
        denom[denom == 0] = 1.0
        self.weights_ = np.nan_to_num(np.sum(z * y_centered[:, None], axis=0) / denom)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CorrelationRegressor is not fitted.")
        z = np.nan_to_num((np.asarray(x, dtype=float) - self.mean_) / self.scale_)
        return z @ self.weights_


def _daily_relevance(frame: pd.DataFrame) -> np.ndarray:
    relevance = (
        frame.groupby("date")["label"]
        .rank(pct=True, method="average")
        .mul(30)
        .round()
        .astype(int)
        .clip(lower=0, upper=30)
    )
    return relevance.to_numpy()


def _fit_lightgbm_models(x: np.ndarray, y: np.ndarray, frame: pd.DataFrame, seed: int) -> List[Dict[str, Any]]:
    import lightgbm as lgb  # type: ignore

    models: List[Dict[str, Any]] = []
    sorted_frame = frame.sort_values(["date", "stock_id"]).reset_index(drop=True)
    x_sorted = x[sorted_frame.index.to_numpy()]
    y_sorted = y[sorted_frame.index.to_numpy()]
    group = sorted_frame.groupby("date").size().to_list()
    rank_label = _daily_relevance(sorted_frame)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    ranker.fit(x_sorted, rank_label, group=group)
    models.append({"name": "lightgbm_ranker", "model": ranker, "weight": 0.50})

    regressor = lgb.LGBMRegressor(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed + 1,
        verbose=-1,
    )
    regressor.fit(x, y)
    models.append({"name": "lightgbm_regressor", "model": regressor, "weight": 0.30})

    positive = (y > np.nanmedian(y)).astype(int)
    if len(np.unique(positive)) == 2:
        classifier = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed + 2,
            verbose=-1,
        )
        classifier.fit(x, positive)
        models.append({"name": "lightgbm_classifier", "model": classifier, "weight": 0.20, "probability": True})

    return models


def _fit_sklearn_models(x: np.ndarray, y: np.ndarray, seed: int) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    if os.environ.get("BDC_USE_SKLEARN_FALLBACK", "0") != "1":
        fallback = CorrelationRegressor().fit(x, y)
        models.append({"name": "correlation_regressor", "model": fallback, "weight": 1.0})
        return models
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

        forest = ExtraTreesRegressor(n_estimators=80, max_depth=8, random_state=seed, n_jobs=1)
        forest.fit(x, y)
        models.append({"name": "extra_trees_regressor", "model": forest, "weight": 0.45})

        random_forest = RandomForestRegressor(n_estimators=60, max_depth=7, random_state=seed + 1, n_jobs=1)
        random_forest.fit(x, y)
        models.append({"name": "random_forest_regressor", "model": random_forest, "weight": 0.30})

        hist = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=seed + 2)
        hist.fit(x, y)
        models.append({"name": "hist_gradient_boosting_regressor", "model": hist, "weight": 0.25})
    except Exception:
        fallback = CorrelationRegressor().fit(x, y)
        models.append({"name": "correlation_regressor", "model": fallback, "weight": 1.0})
    return models


@dataclass
class ModelBundle:
    feature_columns: List[str]
    models: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def predict_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.feature_columns, fill_value=0.0).to_numpy(dtype=float)
        score_parts: Dict[str, np.ndarray] = {}
        weighted = np.zeros(len(frame), dtype=float)
        total_weight = 0.0

        for item in self.models:
            model = item["model"]
            name = str(item["name"])
            weight = float(item.get("weight", 1.0))
            if item.get("probability") and hasattr(model, "predict_proba"):
                raw_score = model.predict_proba(x)[:, 1]
            else:
                raw_score = model.predict(x)
            raw_score = np.asarray(raw_score, dtype=float)
            if np.nanstd(raw_score) > 0:
                normalized = (raw_score - np.nanmean(raw_score)) / np.nanstd(raw_score)
            else:
                normalized = np.zeros_like(raw_score)
            score_parts[name] = raw_score
            weighted += normalized * weight
            total_weight += weight

        final_score = weighted / total_weight if total_weight else weighted
        scored = frame[["date", "stock_id", "sector"]].copy()
        for name, values in score_parts.items():
            scored[f"{name}_score"] = values
        scored["model_score"] = final_score
        scored["model_rank"] = scored["model_score"].rank(ascending=False, method="first").astype(int)
        return scored.sort_values(["model_rank", "stock_id"]).reset_index(drop=True)


def fit_model_bundle(frame: pd.DataFrame, feature_columns: List[str], seed: int) -> ModelBundle:
    ordered = frame.sort_values(["date", "stock_id"]).reset_index(drop=True)
    x = ordered.reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=float)
    y = ordered["label"].to_numpy(dtype=float)
    try:
        models = _fit_lightgbm_models(x, y, ordered, seed)
        backend = "lightgbm"
    except Exception:
        models = _fit_sklearn_models(x, y, seed)
        backend = "sklearn_or_numpy_fallback"

    metadata = {
        "backend": backend,
        "n_rows": int(len(ordered)),
        "n_features": int(len(feature_columns)),
        "train_start": str(ordered["date"].min().date()),
        "train_end": str(ordered["date"].max().date()),
    }
    return ModelBundle(feature_columns=feature_columns, models=models, metadata=metadata)
