# THU-BDC2026 Code Description

## Environment

This project is an offline-first Python pipeline for the 2026 C4 Big Data Challenge task. It uses:

- Python 3.9+
- pandas
- numpy
- scikit-learn fallback models
- optional LightGBM for the primary ranker/regressor/classifier

The reproduction environment must not download data or dependencies at runtime. Put all required data in `app/data` before running.

## Data

Default training data path:

```text
app/data/train.csv
```

Optional local scoring data path:

```text
app/data/test.csv
```

The data reader accepts the official Chinese columns and maps them internally:

```text
股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率, 涨跌幅
```

Optional industry columns such as `行业` or `申万一级行业` are used by reranking when present. If no industry column exists, the pipeline uses `UNKNOWN` and does not enforce the sector limit on that pseudo-sector.

## Algorithm

The pipeline follows the planned seven-stage strategy:

1. Read offline market data from CSV.
2. Build leakage-safe daily price, volume, volatility, cross-sectional, sector, and risk features.
3. Construct labels with the competition rule:

```text
label = open(T+5) / open(T+1) - 1
```

4. Keep time order and train on the latest configured training window.
5. Fit a model bundle. LightGBM Ranker/Regressor/Classifier is used when LightGBM is installed; otherwise deterministic scikit-learn/NumPy fallback models keep the workflow reproducible.
6. Generate `candidate_top30.csv`, apply hard gates and scoring overlay, then select up to Top5.
7. Write and validate `app/output/result.csv`.

The first-version reranker uses:

- liquidity gate by recent average amount
- 20-day drawdown gate
- 20-day single-drop gate
- model rank score
- sector momentum score
- CLV and volume-close price action score
- sector max-holding constraint when real sector labels exist
- equal weight `0.2` per selected stock

## Training

Run:

```bash
sh init.sh
sh train.sh
```

`train.sh` reads `app/data/train.csv`, trains the model bundle, and writes:

```text
app/model/model_bundle.pkl
app/temp/training_metadata.csv
```

All random seeds are fixed through `config.yaml`.

## Prediction

Run:

```bash
sh test.sh
```

`test.sh` loads `app/model/model_bundle.pkl`, predicts on the latest date in `app/data/train.csv`, and writes:

```text
app/temp/candidate_top30.csv
app/temp/ranking_log.csv
app/output/result.csv
```

`result.csv` always uses:

```csv
stock_id,weight
```

It contains no more than 5 stocks. Weight sum is validated to be no greater than 1.

## Local Evaluation

After freezing `app/output/result.csv`, run:

```bash
python app/code/src/evaluate.py --result app/output/result.csv --test app/data/test.csv
```

This computes the official-style weighted open-price return using each selected stock's last 5 rows from `test.csv`:

```text
return_i = (open_last - open_first) / open_first
Final Score = sum(return_i * weight_i)
```

Outputs:

```text
app/temp/tmp.csv
app/temp/evaluation_detail.csv
```

## Result Validation

Run:

```bash
python app/code/src/validate_result.py app/output/result.csv
```

Checks:

- required columns `stock_id,weight`
- at most 5 rows
- no duplicate stocks
- numeric non-negative weights
- total weight no greater than 1

## Notes

- No external data source is downloaded by this code.
- No pretrained model is used by default.
- External data and model MD5 records should be maintained separately if added before B-stage reporting.
- `config.yaml` controls windows, TopN, risk gates, paths, and seed values.
