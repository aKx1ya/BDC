import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app" / "code" / "src"
sys.path.insert(0, str(SRC))


class CoreBehaviorTests(unittest.TestCase):
    def test_lightgbm_rank_relevance_stays_within_default_label_gain(self):
        from models import _daily_relevance

        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-01")] * 100,
                "label": list(range(100)),
            }
        )

        relevance = _daily_relevance(frame)

        self.assertGreaterEqual(relevance.min(), 0)
        self.assertLessEqual(relevance.max(), 30)

    def test_label_uses_t1_to_t5_open_prices(self):
        from featurework import build_labels, normalize_columns

        rows = []
        for day, open_price in enumerate([10, 11, 12, 13, 14, 15, 16], start=1):
            rows.append(
                {
                    "股票代码": "sh.600000",
                    "日期": f"2026-01-{day:02d}",
                    "开盘": open_price,
                    "收盘": open_price + 0.5,
                    "最高": open_price + 1,
                    "最低": open_price - 1,
                    "成交量": 1000 + day,
                    "成交额": 10000 + day,
                    "换手率": 1.0,
                    "涨跌幅": 0.0,
                }
            )

        labeled = build_labels(normalize_columns(pd.DataFrame(rows)), horizon=5)

        first_label = labeled.loc[labeled["date"] == pd.Timestamp("2026-01-01"), "label"].iloc[0]
        self.assertAlmostEqual(first_label, (15 - 11) / 11)

    def test_rolling_features_do_not_change_when_future_row_changes(self):
        from featurework import engineer_features, normalize_columns

        rows = []
        for day in range(1, 12):
            rows.append(
                {
                    "股票代码": "sh.600000",
                    "日期": f"2026-01-{day:02d}",
                    "开盘": 10 + day,
                    "收盘": 10 + day * 2,
                    "最高": 11 + day * 2,
                    "最低": 9 + day,
                    "成交量": 1000 + day * 10,
                    "成交额": 10000 + day * 100,
                    "换手率": 1.0,
                    "涨跌幅": 0.0,
                }
            )

        base = normalize_columns(pd.DataFrame(rows))
        changed = base.copy()
        changed.loc[changed["date"] == pd.Timestamp("2026-01-11"), "close"] = 9999

        base_features = engineer_features(base)
        changed_features = engineer_features(changed)
        target_date = pd.Timestamp("2026-01-08")

        base_value = base_features.loc[base_features["date"] == target_date, "ret_5"].iloc[0]
        changed_value = changed_features.loc[changed_features["date"] == target_date, "ret_5"].iloc[0]
        self.assertAlmostEqual(base_value, changed_value)

    def test_result_validator_rejects_duplicates_and_weight_sum_over_one(self):
        from validate_result import validate_prediction_frame

        duplicate = pd.DataFrame(
            {
                "stock_id": ["sh.600000", "sh.600000"],
                "weight": [0.2, 0.2],
            }
        )
        with self.assertRaises(ValueError):
            validate_prediction_frame(duplicate)

        overweight = pd.DataFrame(
            {
                "stock_id": ["sh.600000", "sh.600001"],
                "weight": [0.6, 0.5],
            }
        )
        with self.assertRaises(ValueError):
            validate_prediction_frame(overweight)

    def test_evaluator_matches_official_open_price_weighted_return(self):
        from evaluate import calculate_weighted_score

        result = pd.DataFrame(
            {
                "stock_id": ["600000", "000001"],
                "weight": [0.5, 0.25],
            }
        )
        test = pd.DataFrame(
            {
                "股票代码": [600000] * 5 + [1] * 5,
                "日期": [f"2026-01-{i:02d}" for i in range(1, 6)] * 2,
                "开盘": [10, 10, 10, 10, 12, 20, 20, 20, 20, 18],
                "收盘": [0] * 10,
            }
        )

        score, detail = calculate_weighted_score(result, test)

        self.assertAlmostEqual(score, 0.5 * 0.2 + 0.25 * -0.1)
        self.assertEqual(len(detail), 2)


class PipelineIntegrationTests(unittest.TestCase):
    def test_training_and_prediction_pipeline_is_reproducible_on_synthetic_data(self):
        from train import run_training
        from test import run_prediction

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "app" / "data"
            model_dir = root / "app" / "model"
            output_dir = root / "app" / "output"
            temp_dir = root / "app" / "temp"
            for path in [data_dir, model_dir, output_dir, temp_dir]:
                path.mkdir(parents=True, exist_ok=True)

            rows = []
            for stock_idx in range(8):
                stock_id = f"sh.60000{stock_idx}"
                for day in range(1, 91):
                    close = 10 + stock_idx * 0.7 + day * (0.03 + stock_idx * 0.002)
                    rows.append(
                        {
                            "股票代码": stock_id,
                            "日期": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                            "开盘": close - 0.05,
                            "收盘": close,
                            "最高": close + 0.2,
                            "最低": close - 0.2,
                            "成交量": 100000 + stock_idx * 1000 + day * 20,
                            "成交额": (100000 + stock_idx * 1000 + day * 20) * close,
                            "换手率": 1.0 + stock_idx * 0.01,
                            "涨跌幅": 0.1,
                        }
                    )

            pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)

            config = {
                "data_path": str(data_dir),
                "model_path": str(model_dir),
                "output_path": str(output_dir),
                "temp_path": str(temp_dir),
                "sequence_length": 20,
                "prediction_horizon": 5,
                "train_window": 60,
                "gap": 5,
                "candidate_top_n": 6,
                "portfolio_size": 5,
                "random_seed": 42,
            }

            run_training(config)
            first = run_prediction(config)
            second = run_prediction(config)

            self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
            self.assertTrue((output_dir / "result.csv").exists())
            self.assertLessEqual(len(first), 5)
            self.assertLessEqual(first["weight"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
