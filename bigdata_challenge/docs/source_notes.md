# Source Notes

本文件说明 `bigdata_challenge` 作为执行层时，各数据源和脚本的角色。

## 执行层定位

`bigdata_challenge/data_fetcher/` 是通用抓数工具箱。

它回答的问题是：

```text
现在仓库里有哪些脚本可以抓哪些原始数据？
```

它不回答：

```text
当前 workflow 应该采用哪些数据？
当前实验应该输出哪些标准表？
当前策略是否要筛股票？
```

这些问题由 `Experiment/ACTIVE_WORKFLOW.md` 和对应 workflow 的 `strategy/` 决定。

## 主要脚本

| 脚本 | 角色 | workflow_0.1 Step-1 当前状态 |
|------|------|------------------------------|
| `01_price_volume.py` | 抓沪深300成分股和日频行情 | 当前核心 |
| `05_sector_momentum.py` | 抓行业/板块相关数据 | 当前可用，但行业口径需核对 |
| `02_northbound_flow.py` | 抓北向资金 | 暂缓 |
| `03_margin_trading.py` | 抓融资融券 | 暂缓 |
| `04_money_flow.py` | 抓资金流 | 暂缓 |
| `06_fundamental.py` | 抓基本面和业绩预告 | 暂缓 |
| `07_feature_engine.py` | 计算特征矩阵 | Step-1 不默认执行 |

## baostock

主要用于：

- 沪深300成分股
- 个股日频行情
- 部分基本面数据

workflow_0.1 Step-1 当前主要依赖 baostock 的行情和股票池能力。

## akshare

主要用于：

- 行业/板块相关数据
- 北向资金
- 融资融券
- 资金流
- 部分业绩预告

workflow_0.1 Step-1 第一版只应谨慎使用行业/板块相关数据。资金流、北向资金、融资融券、基本面暂缓进入第一版核心输入。

## 文件命名风险

当前 raw 目录中存在一些历史产物，其文件名和当前脚本中写出的目标文件名不完全一致。

例子：

- 当前 raw 目录有 `stock_industry.csv`。
- 当前 `05_sector_momentum.py` 主路径目标输出是 `hs300_industry_mapping.csv`。
- 当前 raw 目录有 `sector_kline_ths.csv`。
- 当前 `05_sector_momentum.py` 目标输出是 `sector_daily_kline.csv`。

因此，后续 pipeline 不能只凭文件名假设口径，必须检查字段、来源和 manifest。
