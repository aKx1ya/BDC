"""
STEP 3: 构造标签、滑动窗口与排序样本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略文档章节：第 3 步 标签、滑动窗口和排序样本 Sample Layer
核心问题：把特征表包装成模型能学习的"排序题"——每天一道题，300 只股票，谁涨得多谁排前面。
输入来源：outputs/step2/ 的特征表逻辑（这里重新从 data/train.csv 跑一遍以保证独立可运行）
输出产物：outputs/step3/sample_structure.csv  （某一天的 300 只股票排序样本）
          outputs/step3/sample_summary.txt   （样本总览）
          outputs/step3/label_detail.csv     （某一天每只股票的标签明细）

比喻：Step 1 买菜 → Step 2 备菜 → Step 3 配菜出题
      把处理好的菜按每天的菜单组合，变成"今天这 300 道菜，哪 5 道最好吃？"

关键概念：
  一个样本 ≠ 一只股票
  一个样本 = 某一个交易日 T 的全部 ~300 只股票
  输入 X = 每只股票过去 60 天特征序列
  答案 y = 每只股票未来 5 天收益（label）
  训练目标 = 让未来收益更高的股票排在前面

可调变量：
  - SEQUENCE_LENGTH: 历史窗口 60 天
  - PREDICTION_HORIZON: 预测未来 5 天
"""

import pandas as pd
import numpy as np

# ===== 可调变量 =====
SEQUENCE_LENGTH = 60     # 每只股票看过去多少天
PREDICTION_HORIZON = 5   # 预测未来多少天

# ===== 1. 加载数据并做特征工程（复用 step2 逻辑） =====
print("=" * 60)
print("STEP 3: 构造标签、滑动窗口与排序样本")
print("=" * 60)

print("\n📂 读取数据: data/train.csv")
raw = pd.read_csv("data/train.csv")

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
raw = raw.dropna(subset=["stock_id", "date", "open", "close", "high", "low"])
raw = raw.sort_values(["stock_id", "date"]).drop_duplicates(["stock_id", "date"]).reset_index(drop=True)

# 快速特征工程（精简版，只保证样本构造需要）
def safe_div(a, b):
    b = b.replace(0, np.nan)
    return (a / b).replace([np.inf, -np.inf], np.nan)

df = raw.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = df.groupby("stock_id", group_keys=False)
df["history_count"] = g.cumcount() + 1

for w in [1, 3, 5, 10, 20]:
    if w == 1:
        df["ret_1"] = g["close"].pct_change(1)
    else:
        df[f"ret_{w}"] = g["close"].pct_change(w)
    df[f"ma_{w}"] = g["close"].transform(lambda s, ww=w: s.rolling(ww, min_periods=1).mean())
    df[f"close_to_ma_{w}"] = safe_div(df["close"], df[f"ma_{w}"]) - 1

for w in [5, 10, 20]:
    df[f"volatility_{w}"] = g["ret_1"].transform(lambda s, ww=w: s.rolling(ww, min_periods=2).std())

df["amount_ratio_3"] = safe_div(df["amount"], g["amount"].transform(lambda s: s.rolling(3, min_periods=1).mean()))
df["avg_amount_3"] = g["amount"].transform(lambda s: s.rolling(3, min_periods=1).mean())

price_range = (df["high"] - df["low"]).replace(0, np.nan)
df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / price_range
df["volume_close_strength"] = df["amount_ratio_3"] * df["clv"]

rolling_max = g["close"].transform(lambda s: s.rolling(20, min_periods=1).max())
df["drawdown_20"] = safe_div(df["close"], rolling_max) - 1
df["max_drop_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=1).min())

# 构造标签
df = df.sort_values(["stock_id", "date"])
g = df.groupby("stock_id", group_keys=False)
df["future_open_t1"] = g["open"].shift(-1)
df["future_open_t5"] = g["open"].shift(-PREDICTION_HORIZON)
df["label"] = (df["future_open_t5"] - df["future_open_t1"]) / df["future_open_t1"]
df["label"] = df["label"].replace([np.inf, -np.inf], np.nan)

# ===== 2. 过滤：历史不足 60 天 / 标签不完整的样本 =====
# ⚠️ 必须先过滤 NaN label，再 fillna(0.0)，否则 NaN label 会被填成 0.0 变成假样本
print(f"\n原始行数: {len(df):,}")

# 过滤历史不足的
df_filtered = df[df["history_count"] >= SEQUENCE_LENGTH].copy()
print(f"历史 ≥ {SEQUENCE_LENGTH} 天: {len(df_filtered):,} 行（过滤了 {len(df) - len(df_filtered):,} 行）")

# 过滤标签缺失的（每只股票最后 5 天没有 T+5 数据，label 为 NaN）
before_label_filter = len(df_filtered)
df_filtered = df_filtered.dropna(subset=["label"])
print(f"标签完整: {len(df_filtered):,} 行（过滤了 {before_label_filter - len(df_filtered):,} 行无 label）")

# 过滤完成后再填补 inf/nan（不影响 label 列，因为 NaN label 已经被过滤掉了）
num_cols = df_filtered.select_dtypes(include=[np.number]).columns
df_filtered[num_cols] = df_filtered[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# 补全特征列名
IDENTIFIER_COLS = {"stock_id", "date", "sector"} if "sector" in df_filtered.columns else {"stock_id", "date"}
LEAKAGE_COLS = {"label", "future_open_t1", "future_open_t5", "history_count"}
feature_cols = [c for c in df_filtered.select_dtypes(include=[np.number]).columns if c not in IDENTIFIER_COLS and c not in LEAKAGE_COLS]

# ===== 3. 按日期组织排序样本 =====
# 每天是一个"排序样本"，包含当天所有有效股票
daily_groups = df_filtered.groupby("date")

valid_dates = sorted(daily_groups.groups.keys())
n_samples = len(valid_dates)
daily_sizes = daily_groups.size()

print(f"\n有效排序样本（交易日）: {n_samples} 个")
print(f"每日股票数: {daily_sizes.min():.0f} ~ {daily_sizes.max():.0f} 只，均值 {daily_sizes.mean():.0f} 只")

# ===== 4. 解剖一个具体样本 =====
# 选一个靠中间的日期来展示
example_date = valid_dates[len(valid_dates) // 2]
example_sample = daily_groups.get_group(example_date)
print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"🔍 解剖一个排序样本: {example_date.date()}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"当天有 {len(example_sample)} 只股票")

# 这个样本的"题目"是什么
print(f"\n这道排序题的结构：")
print(f"  输入 X: {len(example_sample)} 只股票 × 过去 {SEQUENCE_LENGTH} 天 × {len(feature_cols)} 个特征")
print(f"         = 一个 ({len(example_sample)}, {SEQUENCE_LENGTH}, {len(feature_cols)}) 的张量")
print(f"  答案 y: {len(example_sample)} 个 label 值（每只股票的未来 5 日收益）")
print(f"  任务:   把 {len(example_sample)} 只股票按 label 从高到低排序")
print(f"         排在前面的 = 模型认为未来 5 天涨得更多")
print(f"         最终只选 Top5")

# 看看 label 的分布
labels = example_sample["label"]
print(f"\n这一天 label 的分布：")
print(f"  最高收益: {labels.max():.4f} ({labels.max()*100:.2f}%)")
print(f"  最低收益: {labels.min():.4f} ({labels.min()*100:.2f}%)")
print(f"  均值: {labels.mean():.4f} ({labels.mean()*100:.2f}%)")
print(f"  正收益: {(labels > 0).sum()} 只 / {len(labels)} 只 ({(labels > 0).mean()*100:.1f}%)")
print(f"  Top5 平均收益: {labels.nlargest(5).mean():.4f} ({labels.nlargest(5).mean()*100:.2f}%)")
print(f"  Top5 中最低收益: {labels.nlargest(5).min():.4f} ({labels.nlargest(5).min()*100:.2f}%)")

# 看看如果瞎选 vs 选最好的差距
print(f"\n如果这一天做预测：")
print(f"  瞎选 5 只平均收益: {labels.sample(5, random_state=42).mean():.4f}")
print(f"  选最好的 5 只收益:  {labels.nlargest(5).mean():.4f}")
print(f"  选最差的 5 只收益:  {labels.nsmallest(5).mean():.4f}")
print(f"  → 选对和选错的差距 = {labels.nlargest(5).mean() - labels.nsmallest(5).mean():.4f}")
print(f"  → 这就是模型要学的东西：把最好的排在前面，最差的排后面")

# ===== 5. 展示 Top5 vs Bottom5 的股票 =====
print(f"\n这一天实际涨幅最好的 5 只股票（正确答案）：")
top5 = example_sample.nlargest(5, "label")[["stock_id", "label", "close", "drawdown_20", "clv"]]
top5["label_pct"] = (top5["label"] * 100).round(2).astype(str) + "%"
for _, row in top5.iterrows():
    print(f"  stock={row['stock_id']:>6s}  label={row['label']:+.4f}  ({row['label']*100:+.2f}%)  close={row['close']:.2f}  drawdown={row['drawdown_20']:.3f}  clv={row['clv']:+.3f}")

print(f"\n这一天实际涨幅最差的 5 只股票：")
bottom5 = example_sample.nsmallest(5, "label")[["stock_id", "label", "close", "drawdown_20", "clv"]]
for _, row in bottom5.iterrows():
    print(f"  stock={row['stock_id']:>6s}  label={row['label']:+.4f}  ({row['label']*100:+.2f}%)  close={row['close']:.2f}  drawdown={row['drawdown_20']:.3f}  clv={row['clv']:+.3f}")

# ===== 6. 整个训练集有多少个"排序题" =====
print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"📊 整个训练集概览")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"排序样本数（交易日）: {n_samples}")
print(f"每个样本的输入形状: (~300, {SEQUENCE_LENGTH}, {len(feature_cols)})")
print(f"第一个样本日期: {valid_dates[0].date()}")
print(f"最后一个样本日期: {valid_dates[-1].date()}")

# 查看跨天的 Top5 / Bottom5 波动
top5_returns = []
for date in valid_dates[-20:]:  # 最后 20 天
    group = daily_groups.get_group(date)
    top5_returns.append(group["label"].nlargest(5).mean())
print(f"\n最近 20 天 Top5 平均收益: {np.mean(top5_returns):.4f} ({np.mean(top5_returns)*100:.2f}%)")
print(f"  范围: [{min(top5_returns):.4f}, {max(top5_returns):.4f}]")
print(f"  就是说，最好的 5 只平均能赚 {np.mean(top5_returns)*100:.1f}%，但每天波动很大")

# ===== 7. 输出文件 =====
print(f"\n📁 输出文件:")
# 保存示例样本
example_out = example_sample[["stock_id", "date", "close", "label"] + feature_cols[:8]].copy()
example_out = example_out.sort_values("label", ascending=False)
example_out["date"] = example_out["date"].dt.strftime("%Y-%m-%d")
example_out.to_csv("outputs/step3/sample_structure.csv", index=False, encoding="utf-8")
print(f"  → outputs/step3/sample_structure.csv  ({len(example_out)} 只股票 × label降序)")

# 标签明细
label_detail = example_sample[["stock_id", "label"]].copy()
label_detail = label_detail.sort_values("label", ascending=False)
label_detail["rank"] = range(1, len(label_detail) + 1)
label_detail["label_pct"] = (label_detail["label"] * 100).round(2)
label_detail.to_csv("outputs/step3/label_detail.csv", index=False, encoding="utf-8")
print(f"  → outputs/step3/label_detail.csv  ({len(label_detail)} 只股票 × label排名)")

# 汇总
with open("outputs/step3/sample_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"总排序样本数: {n_samples}\n")
    f.write(f"日期范围: {valid_dates[0].date()} ~ {valid_dates[-1].date()}\n")
    f.write(f"每日股票数: {daily_sizes.min():.0f} ~ {daily_sizes.max():.0f}，均值 {daily_sizes.mean():.0f}\n")
    f.write(f"输入形状: (~300, {SEQUENCE_LENGTH}, {len(feature_cols)})\n")
    f.write(f"特征数: {len(feature_cols)}\n")
    f.write(f"标签公式: (open_t5 - open_t1) / open_t1\n")
    f.write(f"\n示例日期 {example_date.date()}:\n")
    f.write(f"  总股票数: {len(example_sample)}\n")
    f.write(f"  Top5 平均收益: {labels.nlargest(5).mean():.4f}\n")
    f.write(f"  Bottom5 平均收益: {labels.nsmallest(5).mean():.4f}\n")
    f.write(f"  正收益比例: {(labels > 0).mean()*100:.1f}%\n")
print(f"  → outputs/step3/sample_summary.txt")

print("\n" + "=" * 60)
print("✅ Step 3 完成")
print("=" * 60)
