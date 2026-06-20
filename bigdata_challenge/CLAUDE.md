# CLAUDE.md

This file provides execution-layer guidance for Claude Code when working inside `bigdata_challenge`.

`bigdata_challenge` is the raw-data execution layer. It should not be treated as the strategy source of truth for the current experiment.

## 调度入口

Before deciding what to run, read:

```text
../Experiment/ACTIVE_WORKFLOW.md
```

Then read the active workflow files referenced there, usually:

```text
../Experiment/workflow_0.1/README.md
../Experiment/workflow_0.1/strategy/
```

The active workflow defines:

- current strategy version
- current stage, such as Step-1 or Step-2
- allowed execution commands
- expected output directory
- schema and data-output requirements

## 本目录职责

`bigdata_challenge` is responsible for:

- fetching raw market and auxiliary data
- keeping raw CSV files in `data/raw/`
- exposing stable execution scripts under `data_fetcher/`
- documenting data sources and known raw-data caveats

`bigdata_challenge` is not responsible for:

- defining the active workflow strategy
- deciding final stock screens
- producing final experiment deliverables
- writing trading conclusions
- silently running feature engineering when the active workflow only asks for Step-1

## 默认运行规则

Only run commands allowed by `../Experiment/ACTIVE_WORKFLOW.md`.

For the current `workflow_0.1` Step-1 stage, the expected raw-data commands are:

```bash
cd data_fetcher
python3 run_all.py --step 1
python3 run_all.py --step 5
```

Do not run feature engineering by default:

```bash
python3 run_all.py --with-feature
```

Only run it when the active workflow explicitly allows Step-2 or feature generation.

## 流水线能力

| 步骤 | 文件 | 数据源 | 内容 | 当前定位 |
|------|------|--------|------|----------|
| 1 | `01_price_volume.py` | baostock | 沪深300成分股和日K线 | Step-1 可用 |
| 2 | `02_northbound_flow.py` | akshare | 北向资金 | 当前 workflow_0.1 Step-1 暂不默认使用 |
| 3 | `03_margin_trading.py` | akshare | 融资融券 | 当前 workflow_0.1 Step-1 暂不默认使用 |
| 4 | `04_money_flow.py` | akshare | 资金流向 | 当前 workflow_0.1 Step-1 暂不默认使用 |
| 5 | `05_sector_momentum.py` | akshare | 行业/板块相关数据 | Step-1 可用，但口径需核对 |
| 6 | `06_fundamental.py` | baostock + akshare | 基本面和业绩预告 | 当前 workflow_0.1 Step-1 暂不默认使用 |
| 7 | `07_feature_engine.py` | 本地 CSV | 特征工程 | 不属于 Step-1 默认执行 |

## 输出边界

Raw files stay here:

```text
bigdata_challenge/data/raw/
```

Feature files, if explicitly generated, stay here:

```text
bigdata_challenge/data/features/
```

Workflow-specific deliverables should be written outside this directory, under the active experiment path, for example:

```text
../Experiment/workflow_0.1/experiments/exp_YYYYMMDD_step1_主题/outputs/step1/
```

## 关键机制

- Incremental update: scripts check local CSV dates before appending new rows when supported.
- Retry logic: akshare requests use retry helpers in `data_fetcher/utils.py`.
- Shared config: `data_fetcher/config.py` defines `RAW_DIR`, `FEATURE_DIR`, `START_DATE`, `END_DATE`, and `HS300_CODE`.

## 数据源注意事项

- baostock is used for stable market data and some fundamentals.
- akshare `_em` endpoints can be rate-limited by Eastmoney sources.
- akshare `_ths` endpoints are Tonghuashun-style board data sources.
- Step 5 output names and historical raw filenames may differ; check `docs/data_inventory.md` before assuming a file is current.

## 当前推荐工作方式

1. Read `../Experiment/ACTIVE_WORKFLOW.md`.
2. Read the active workflow strategy and schema files.
3. Run only the allowed `data_fetcher` steps.
4. Treat `data/raw/` as raw inputs.
5. Let `Experiment/workflow_x/pipelines/` convert raw inputs into workflow-specific outputs.
