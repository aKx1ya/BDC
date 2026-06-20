# Data Inventory

本文件记录 `bigdata_challenge/data/` 中现有数据的当前定位。

原则：

- `data/raw/` 是原始抓取结果，不直接等同于某个 workflow 的最终产物。
- `Experiment/workflow_x/pipelines/` 负责把 raw 文件整理成 workflow 标准输出。
- 不确定口径的文件先标记，不直接删除。

## workflow_0.1 Step-1 当前核心 raw 输入

| 文件 | 当前作用 | 备注 |
|------|----------|------|
| `data/raw/hs300_stocks.csv` | 沪深300股票池 | Step-1 股票范围基础 |
| `data/raw/daily_price_volume.csv` | 个股日频行情 | Step-1 最核心 raw 数据 |
| `data/raw/stock_industry.csv` | 个股行业映射参考 | 现存历史产物，需核对与当前脚本输出口径 |
| `data/raw/sector_kline_ths.csv` | 板块历史行情参考 | 现存历史产物，当前 `05_sector_momentum.py` 输出名不同 |
| `data/raw/sector_industry_ths.csv` | 板块名称列表参考 | 现存历史产物 |

## 当前 workflow_0.1 Step-1 暂不默认使用的数据

这些文件可以保留，但不应自动进入 `workflow_0.1` Step-1 第一版核心输入：

| 文件 | 原因 |
|------|------|
| `data/raw/northbound_fund_summary.csv` | 北向资金暂缓进入第一版 |
| `data/raw/northbound_sh_hist.csv` | 北向资金暂缓进入第一版 |
| `data/raw/northbound_sz_hist.csv` | 北向资金暂缓进入第一版 |
| `data/raw/margin_sh_total.csv` | 融资融券暂缓进入第一版 |
| `data/raw/margin_summary_sh.csv` | 融资融券暂缓进入第一版 |
| `data/raw/margin_summary_sz.csv` | 融资融券暂缓进入第一版 |
| `data/raw/margin_detail_sh_latest.csv` | 融资融券暂缓进入第一版 |
| `data/raw/money_flow_realtime.csv` | 资金流/即时快照暂缓进入第一版 |
| `data/raw/industry_fund_flow.csv` | 行业资金流暂缓进入第一版 |
| `data/raw/fundamental_profit.csv` | 基本面暂缓进入第一版 |
| `data/raw/fundamental_quarterly.csv` | 基本面暂缓进入第一版 |
| `data/raw/fundamental_growth.csv` | 基本面暂缓进入第一版 |
| `data/raw/earnings_forecast.csv` | 业绩预告暂缓进入第一版 |

## Feature 数据

| 文件 | 当前定位 |
|------|----------|
| `data/features/feature_matrix.csv` | 历史或后续特征工程产物；不属于 workflow_0.1 Step-1 默认输入 |

## 后续整理建议

1. 在确认不再需要旧命名文件前，不删除 raw 数据。
2. 如果旧文件仍有参考价值，可后续移动到 `data/legacy/`。
3. 如果某文件进入 workflow 标准产物，必须在对应 workflow 的 manifest 中记录来源和口径。
