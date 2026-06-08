# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

大数据挑战赛：预测沪深300成分股5日收益率（T+1开盘买入 → T+5开盘卖出）。
当前阶段：**只获取原始数据**，不做特征工程和建模。

## 运行命令

```bash
cd data_fetcher

# 日常增量更新（默认只跑步骤1~6，跳过特征工程）
python3 run_all.py

# 单独运行某一步
python3 run_all.py --step 1

# 从第3步开始
python3 run_all.py --from 3

# 包含特征工程（步骤7）
python3 run_all.py --with-feature
```

## 流水线结构

| 步骤 | 文件 | 数据源 | 内容 |
|------|------|--------|------|
| 1 | 01_price_volume.py | baostock | 300只股票日K线 |
| 2 | 02_northbound_flow.py | akshare | 北向资金（沪深股通） |
| 3 | 03_margin_trading.py | akshare | 融资融券 |
| 4 | 04_money_flow.py | akshare | 资金流向排名 |
| 5 | 05_sector_momentum.py | akshare | 行业板块K线 |
| 6 | 06_fundamental.py | baostock+akshare | 季度财务+业绩预告 |
| 7 | 07_feature_engine.py | 本地CSV | 特征工程（默认不执行） |

输出：`data/raw/`（原始数据）、`data/features/`（特征矩阵）

## 关键机制

- **增量更新**: 每个脚本运行前检查本地CSV最大日期，只下载新增数据
- **重试逻辑**: akshare请求失败自动重试3次（指数退避：5s/10s/20s），内置在 `utils.py` 的 `@retry_request` 装饰器中
- **工具函数**: `utils.py` 提供 `get_last_date()`、`append_csv()`、`next_day()`、`@retry_request`

## 配置

`config.py`:
- 路径: `RAW_DIR`、`FEATURE_DIR`
- 时间: 2023-01-01 ~ 今天
- 股票池: 沪深300 (`sh.000300`)

## 数据源注意事项

- **baostock**: 稳定无限流，用于K线和基本面
- **akshare `_em` 接口**: 东方财富源，常被限流。限流时换网络（手机热点）
- **akshare `_ths` 接口**: 同花顺源，板块数据备选
- 步骤4的资金流向接口只返回排名数据，非全量300只个股逐日数据。如需全量需逐股抓取（`stock_individual_fund_flow`），当前暂跳过
