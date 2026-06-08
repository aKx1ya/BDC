"""
STEP 2: 特征工程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略文档章节：第 2 步 特征工程 Feature Layer
核心问题：把原始 OHLCV 加工成模型能看懂的信号——每只股票每天的状态指标。
输入来源：data/train.csv（第 1 步的原始数据）
输出产物：outputs/step2/feature_table_sample.csv（前 2000 行样本，方便查看）
          outputs/step2/feature_columns.txt（所有特征名清单）
          outputs/step2/feature_stats.csv（每列特征的统计信息）
          outputs/step2/label_distribution.txt（标签分布统计）

比喻：Step 1 买菜 → Step 2 备菜（切、腌、打散、调味）
      同一个原料 close → 变成 ret_5（肉片）、ma_20（腌肉）、volatility_20（磨粉）

关键概念：
  2.4  个股价量特征：只看自己，不看别人（收益率、均线、波动率、成交量、K线形态）
  2.5  横截面特征：同一天和 300 只股票比排名
  2.7  行业/板块特征：行业整体涨跌（当前因无行业数据，全部归为 UNKNOWN）
  2.10 风险特征：回撤、大跌、流动性
  2.14 防泄漏：每条特征只用到 T 日及以前的数据

可调变量（改这里就能实验）：
  - RETURN_WINDOWS: 收益率窗口 [1,3,5,10,20,40]
  - MA_WINDOWS: 均线窗口 [3,5,10,20,40]
  - VOL_WINDOWS: 波动率窗口 [5,10,20]
  - AMOUNT_WINDOWS: 成交额窗口 [3,5,20]
  - SECTOR_MOMENTUM_WINDOW: 行业动量窗口 3
  - DRAWDOWN_WINDOW: 回撤窗口 20
  - PREDICTION_HORIZON: 标签周期 5
"""

import pandas as pd
import numpy as np

# ===== 可调变量 =====
RETURN_WINDOWS = [1, 3, 5, 10, 20, 40]
MA_WINDOWS = [3, 5, 10, 20, 40]
VOL_WINDOWS = [5, 10, 20]
AMOUNT_WINDOWS = [3, 5, 20]
SECTOR_MOMENTUM_WINDOW = 3
DRAWDOWN_WINDOW = 20
PREDICTION_HORIZON = 5  # 预测未来 5 日

# ===== 1. 加载数据 =====
print("=" * 60)
print("STEP 2: 特征工程")
print("=" * 60)

print("\n📂 读取原始数据: data/train.csv")
raw = pd.read_csv("data/train.csv")

# 列名标准化
COLUMN_MAP = {
    "股票代码": "stock_id", "日期": "date",
    "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
}
raw = raw.rename(columns=COLUMN_MAP)
raw["stock_id"] = raw["stock_id"].astype(str).str.strip()
raw["date"] = pd.to_datetime(raw["date"])
for col in ["open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# 去重排序
raw = raw.dropna(subset=["stock_id", "date", "open", "close", "high", "low"])
raw = raw.sort_values(["stock_id", "date"]).drop_duplicates(["stock_id", "date"], keep="last")
raw = raw.reset_index(drop=True)

print(f"原始数据: {len(raw):,} 行 × {len(raw.columns)} 列")
print(f"股票数: {raw['stock_id'].nunique()} 只")
print(f"日期范围: {raw['date'].min().date()} ~ {raw['date'].max().date()}")

# ===== 辅助函数 =====
def safe_div(a, b):
    """安全除法，分母为 0 时返回 NaN"""
    b = b.replace(0, np.nan)
    return (a / b).replace([np.inf, -np.inf], np.nan)


# ===== 2. 核心特征工程 =====
print("\n" + "-" * 40)
print("【特征工程】开始加工...")
print("-" * 40)

features = raw.sort_values(["stock_id", "date"]).reset_index(drop=True)
grouped = features.groupby("stock_id", group_keys=False)

# ------ 2.4.1 历史数据计数 ------
print("  1/8 历史数据计数...")
features["history_count"] = grouped.cumcount() + 1  # 这只股票到目前为止出现过多少天

# ------ 2.4.2 收益率特征 ------
print("  2/8 收益率特征...")
for w in RETURN_WINDOWS:
    if w == 1:
        features["ret_1"] = grouped["close"].pct_change(1)
    else:
        features[f"ret_{w}"] = grouped["close"].pct_change(w)
# 收益率：close_t / close_{t-w} - 1

# ------ 2.4.3 均线特征 ------
print("  3/8 均线特征...")
for w in MA_WINDOWS:
    features[f"ma_{w}"] = grouped["close"].transform(
        lambda s, ww=w: s.rolling(ww, min_periods=1).mean()
    )
    features[f"close_to_ma_{w}"] = safe_div(features["close"], features[f"ma_{w}"]) - 1
# ma = 过去 w 天收盘价平均
# close_to_ma = 今天收盘价 / 均线 - 1 → 在均线上方还是下方？

# ------ 2.4.4 波动率特征 ------
print("  4/8 波动率特征...")
for w in VOL_WINDOWS:
    features[f"volatility_{w}"] = grouped["ret_1"].transform(
        lambda s, ww=w: s.rolling(ww, min_periods=2).std()
    )
# volatility = 过去 w 天日收益率的标准差 → 波动多大

# ------ 2.4.5 成交量/成交额特征 ------
print("  5/8 成交量特征...")
for w in AMOUNT_WINDOWS:
    features[f"amount_ma_{w}"] = grouped["amount"].transform(
        lambda s, ww=w: s.rolling(ww, min_periods=1).mean()
    )
    features[f"volume_ma_{w}"] = grouped["volume"].transform(
        lambda s, ww=w: s.rolling(ww, min_periods=1).mean()
    )
features["amount_ratio_3"] = safe_div(features["amount"], features["amount_ma_3"])
features["volume_ratio_5"] = safe_div(features["volume"], features["volume_ma_5"])
features["avg_amount_3"] = features["amount_ma_3"]
# amount_ratio_3 > 1 → 今天放量；< 1 → 今天缩量

# ------ 2.4.6 K线价格行为特征 ------
print("  6/8 K线价格行为特征...")
price_range = (features["high"] - features["low"]).replace(0, np.nan)

# CLV: 收盘位置，+1 = 收在最高，-1 = 收在最低
features["clv"] = ((features["close"] - features["low"]) - (features["high"] - features["close"])) / price_range

# 下影线占比：跌下去被拉回来的程度
lower_body = features[["open", "close"]].min(axis=1)
features["lower_shadow_ratio"] = (lower_body - features["low"]) / price_range

# 上影线占比：冲高回落的程度
upper_body = features[["open", "close"]].max(axis=1)
features["upper_shadow_ratio"] = (features["high"] - upper_body) / price_range

# 实体大小：多空一方明显占优还是胶着
features["body_ratio"] = (features["close"] - features["open"]).abs() / price_range

# 振幅与收盘价之比
features["range_ratio"] = safe_div(features["high"] - features["low"], features["close"])

# 放量收高：量价配合信号 = 放量程度 × 收盘位置
features["volume_close_strength"] = features["amount_ratio_3"] * features["clv"]
# vol_close_strength 高 → 放量 + 收在高位 → 资金主动买入

# ------ 2.4.7 风险特征 ------
print("  7/8 风险特征...")
rolling_max_20 = grouped["close"].transform(lambda s: s.rolling(DRAWDOWN_WINDOW, min_periods=1).max())
features["drawdown_20"] = safe_div(features["close"], rolling_max_20) - 1
# drawdown_20 = -0.15 → 当前价比 20 日最高点跌了 15%

features["max_drop_20"] = grouped["ret_1"].transform(lambda s: s.rolling(DRAWDOWN_WINDOW, min_periods=1).min())
# max_drop_20 = -0.08 → 过去 20 天单日最大跌幅 8%

# ------ 2.4.8 横截面排名特征 ------
print("  8/8 横截面排名特征...")
rank_cols = [f"ret_{w}" for w in RETURN_WINDOWS] + ["amount", f"volatility_{VOL_WINDOWS[-1]}"]
for col in rank_cols:
    if col in features.columns:
        features[f"{col}_rank_pct"] = features.groupby("date")[col].rank(pct=True, method="average")
# rank_pct = 0.95 → 今天这个指标超过了 95% 的股票

# ------ 2.7 行业动量特征 ------
print("  行业动量特征（当前行业数据缺失，全部归为 'UNKNOWN'）...")
# 行业列缺失 → 填 UNKNOWN
if "sector" not in features.columns:
    features["sector"] = "UNKNOWN"
features["sector"] = features["sector"].fillna("UNKNOWN").astype(str)

# 每日每行业平均收益
sector_daily = (
    features.groupby(["date", "sector"], as_index=False)["ret_1"]
    .mean()
    .rename(columns={"ret_1": "sector_ret_1"})
    .sort_values(["sector", "date"])
)
# 行业近 3 日累计收益
sector_daily["sector_momentum_3"] = sector_daily.groupby("sector")["sector_ret_1"].transform(
    lambda s: s.rolling(SECTOR_MOMENTUM_WINDOW, min_periods=1).sum()
)
# 行业动量排名
sector_daily["sector_momentum_rank_pct"] = sector_daily.groupby("date")["sector_momentum_3"].rank(
    pct=True, method="average"
)
features = features.merge(sector_daily, on=["date", "sector"], how="left")

# 当前因为行业全是 UNKNOWN，所有 300 只股票属于同一个"行业"
# 所以 sector_momentum_3 = 当天所有股票的平均收益
# sector_momentum_rank_pct = 永远 1.0（只有一个行业）
print(f"  ⚠️ 行业数: {features['sector'].nunique()}（全为 UNKNOWN → 行业特征失效）")

# ------ 清理 ------
numeric_cols = features.select_dtypes(include=[np.number]).columns
features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ------ 构造标签（比赛口径：T+1 开盘买入 → T+5 开盘卖出）------
print("\n  构造标签...")
features = features.sort_values(["stock_id", "date"])
g = features.groupby("stock_id", group_keys=False)
features["future_open_t1"] = g["open"].shift(-1)
features["future_open_t5"] = g["open"].shift(-PREDICTION_HORIZON)
features["label"] = (features["future_open_t5"] - features["future_open_t1"]) / features["future_open_t1"]
# label = +0.05 → 未来 5 天预计涨 5%
# label = -0.03 → 未来 5 天预计跌 3%
features["label"] = features["label"].replace([np.inf, -np.inf], np.nan)

# ===== 3. 输出结果 =====
print("\n" + "=" * 40)
print("【输出】")
print("=" * 40)

# 识别哪些是特征列
IDENTIFIER_COLS = {"stock_id", "date", "sector"}
LEAKAGE_COLS = {"label", "future_open_t1", "future_open_t5", "history_count"}
feature_cols = [
    c for c in features.select_dtypes(include=[np.number]).columns
    if c not in IDENTIFIER_COLS
    if c not in LEAKAGE_COLS
]

print(f"\n总列数: {len(features.columns)}")
print(f"特征列: {len(feature_cols)}")
print(f"标识列: stock_id, date, sector")
print(f"泄漏列（不进模型）: label, future_open_t1, future_open_t5, history_count")

# --- 3.1 特征列清单 ---
with open("outputs/step2/feature_columns.txt", "w", encoding="utf-8") as f:
    f.write("【Step 2 特征清单】\n\n")
    f.write(f"总特征数: {len(feature_cols)}\n\n")

    categories = {
        "原始字段": ["open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"],
        "收益率": [c for c in feature_cols if c.startswith("ret_") and not c.endswith("_rank_pct")],
        "均线": [c for c in feature_cols if c.startswith("ma_") or c.startswith("close_to_ma_")],
        "波动率": [c for c in feature_cols if c.startswith("volatility_") and not c.endswith("_rank_pct")],
        "成交量/额": [c for c in feature_cols if ("amount" in c or "volume" in c) and "_ma_" in c],
        "量比": ["amount_ratio_3", "volume_ratio_5"],
        "K线形态": ["clv", "lower_shadow_ratio", "upper_shadow_ratio", "body_ratio", "range_ratio", "volume_close_strength"],
        "风险": ["drawdown_20", "max_drop_20", "avg_amount_3"],
        "横截面排名": [c for c in feature_cols if c.endswith("_rank_pct")],
        "行业": ["sector_momentum_3", "sector_momentum_rank_pct"],
    }

    for cat, cols in categories.items():
        existing = [c for c in cols if c in feature_cols]
        if existing:
            f.write(f"\n【{cat}】({len(existing)} 个)\n")
            for c in existing:
                f.write(f"  {c}\n")

print(f"  → outputs/step2/feature_columns.txt")

# --- 3.2 特征表样本 ---
sample_rows = features.head(2000)
sample_rows.to_csv("outputs/step2/feature_table_sample.csv", index=False, encoding="utf-8")
print(f"  → outputs/step2/feature_table_sample.csv（前 2000 行，完整表有 {len(features):,} 行）")

# --- 3.3 每列特征的统计信息 ---
stats_rows = []
for col in feature_cols:
    series = features[col]
    stats_rows.append({
        "特征名": col,
        "均值": series.mean(),
        "标准差": series.std(),
        "最小值": series.min(),
        "25%分位": series.quantile(0.25),
        "中位数": series.median(),
        "75%分位": series.quantile(0.75),
        "最大值": series.max(),
        "缺失数": series.isna().sum(),
    })
pd.DataFrame(stats_rows).to_csv("outputs/step2/feature_stats.csv", index=False, encoding="utf-8")
print(f"  → outputs/step2/feature_stats.csv（{len(feature_cols)} 列特征的统计）")

# --- 3.4 标签分布 ---
label_valid = features["label"].dropna()
label_positive = (label_valid > 0).sum()
label_negative = (label_valid < 0).sum()
with open("outputs/step2/label_distribution.txt", "w", encoding="utf-8") as f:
    f.write("【标签分布：未来 5 日开盘-开盘收益】\n\n")
    f.write(f"总样本数: {len(label_valid):,}\n")
    f.write(f"正收益样本: {label_positive:,} ({label_positive/len(label_valid)*100:.1f}%)\n")
    f.write(f"负收益样本: {label_negative:,} ({label_negative/len(label_valid)*100:.1f}%)\n")
    f.write(f"标签均值: {label_valid.mean():.6f}\n")
    f.write(f"标签标准差: {label_valid.std():.6f}\n")
    f.write(f"标签最小值: {label_valid.min():.6f}\n")
    f.write(f"标签最大值: {label_valid.max():.6f}\n")
    f.write(f"\n标签公式: label = (open_t5 - open_t1) / open_t1\n")
    f.write(f"标签含义: 如果在 T 日买入，持有到 T+5 开盘，预计收益率\n")

print(f"  → outputs/step2/label_distribution.txt")

# --- 3.5 打印关键信息 ---
print(f"\n📊 标签分布:")
print(f"  正收益: {label_positive:,} ({label_positive/len(label_valid)*100:.1f}%)")
print(f"  负收益: {label_negative:,} ({label_negative/len(label_valid)*100:.1f}%)")
print(f"  均值: {label_valid.mean():.6f}  |  范围: [{label_valid.min():.4f}, {label_valid.max():.4f}]")

print(f"\n📊 特征统计（举例）:")
nonzero_ret = features[features["ret_5"] != 0]["ret_5"]
print(f"  ret_5 (5日收益): 均值 {nonzero_ret.mean():.4f}  |  范围 [{nonzero_ret.min():.4f}, {nonzero_ret.max():.4f}]")
print(f"  clv (收盘位置): 均值 {features['clv'].mean():.4f}  |  范围 [{features['clv'].min():.4f}, {features['clv'].max():.4f}]")
print(f"  drawdown_20: 均值 {features['drawdown_20'].mean():.4f}  |  最小值 {features['drawdown_20'].min():.4f}")

print(f"\n⚠️  行业特征状态:")
print(f"  行业数: {features['sector'].nunique()}")
print(f"  如果 = 1 → 所有股票被视为同一行业，行业动量/约束不可用")
print(f"  原因: train.csv 中没有「行业」列")

print("\n" + "=" * 60)
print("✅ Step 2 完成 → 产出 4 个文件在 outputs/step2/")
print("=" * 60)
