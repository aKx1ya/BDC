# workflow_0.1 Pipelines

本目录用于存放 `workflow_0.1` 专用的数据整理脚本。

它和 `bigdata_challenge/data_fetcher/` 的职责不同：

- `bigdata_challenge/data_fetcher/` 负责抓原始数据。
- `Experiment/workflow_0.1/pipelines/` 负责把原始数据整理成 `workflow_0.1` 规定的实验产物。

## 当前目标

Step-1 的整理层最终应生成：

```text
outputs/step1/
├── step1_daily_raw_data.csv
├── step1_stock_summary.csv
├── step1_sector_summary.csv
└── step1_data_manifest.csv
```

## 计划中的脚本边界

```text
build_step1_outputs.py
```

读取 `bigdata_challenge/data/raw/` 中的原始数据，生成 Step-1 标准 CSV。

```text
sector_mapping.py
```

维护行业到六大风格板块的映射逻辑。

```text
manifest_writer.py
```

写出 `step1_data_manifest.csv`，记录数据窗口、来源、schema 和注意事项。

## 重要约束

- 这里不直接抓取外部数据。
- 这里不做最终筛股。
- 这里不输出交易结论。
- 这里只把 raw 数据转成 `workflow_0.1` 认可的 Step-1 数据资产。
