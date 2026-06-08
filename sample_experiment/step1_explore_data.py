"""
STEP 1: 数据获取与探查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略文档章节：第 1 步 数据获取 Data Layer
核心问题：我们手里的 train.csv / test.csv 到底长什么样？里面有什么？能支撑后面的步骤吗？
输入来源：sample/data/train.csv, sample/data/test.csv（软链接到 Baseline /app/data/）
输出产物：outputs/step1/data_summary.txt

关键概念：
  1. 数据长什么样 — 多少行、多少列、日期范围、多少只股票
  2. 时间粒度 — 每天每只股票一行（日线数据）
  3. 股票池 — 沪深300成分股（用股票代码标识）
  4. 防泄漏 — 这里只是探查，不加工任何特征

可调变量：
  - 无（这一步只读数据，不做任何修改）
"""

import pandas as pd

# ===== 1. 加载数据 =====
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"

print("=" * 60)
print("STEP 1: 数据获取与探查")
print("=" * 60)

print(f"\n📂 读取训练数据: {TRAIN_PATH}")
train_raw = pd.read_csv(TRAIN_PATH)

print(f"📂 读取测试数据: {TEST_PATH}")
test_raw = pd.read_csv(TEST_PATH)

# ===== 2. 探查：基本信息 =====
print("\n" + "-" * 40)
print("【1】基本信息")
print("-" * 40)

print(f"\n训练集: {len(train_raw):,} 行, {len(train_raw.columns)} 列")
print(f"测试集: {len(test_raw):,} 行, {len(test_raw.columns)} 列")

print(f"\n列名: {list(train_raw.columns)}")

# ===== 3. 探查：时间范围 =====
print("\n" + "-" * 40)
print("【2】时间范围")
print("-" * 40)

train_raw["日期"] = pd.to_datetime(train_raw["日期"])
test_raw["日期"]  = pd.to_datetime(test_raw["日期"])

print(f"\n训练集: {train_raw['日期'].min().date()} ~ {train_raw['日期'].max().date()}")
print(f"  - 交易日数: {train_raw['日期'].nunique()} 天")

print(f"\n测试集: {test_raw['日期'].min().date()} ~ {test_raw['日期'].max().date()}")
print(f"  - 交易日数: {test_raw['日期'].nunique()} 天")

# 训练集和测试集之间是否有重叠？
train_dates = set(train_raw["日期"].dt.date)
test_dates  = set(test_raw["日期"].dt.date)
overlap = train_dates & test_dates
print(f"\n训练/测试日期重叠: {'❌ 有重叠!' if overlap else '✅ 无重叠，时间顺序正确'}")
if overlap:
    print(f"  重叠日期: {sorted(overlap)}")

# 训练集最后一个日期 vs 测试集第一个日期
train_end = train_raw["日期"].max().date()
test_start = test_raw["日期"].min().date()
print(f"训练最后一天: {train_end}")
print(f"测试第一天:   {test_start}")
print(f"间隔天数:     {(pd.Timestamp(test_start) - pd.Timestamp(train_end)).days} 天（应接近交易间隔）")

# ===== 4. 探查：股票数量 =====
print("\n" + "-" * 40)
print("【3】股票池")
print("-" * 40)

# 注意：股票代码是数字（如 1, 600036 等）
train_raw["股票代码"] = train_raw["股票代码"].astype(str)
test_raw["股票代码"]  = test_raw["股票代码"].astype(str)

train_stocks = set(train_raw["股票代码"].unique())
test_stocks  = set(test_raw["股票代码"].unique())

print(f"\n训练集股票数: {len(train_stocks)} 只")
print(f"测试集股票数: {len(test_stocks)} 只")

# 测试集的股票是否都在训练集出现过？
new_stocks = test_stocks - train_stocks
if new_stocks:
    print(f"测试集新股票（训练集中未出现）: {len(new_stocks)} 只 → {sorted(new_stocks)}")
else:
    print(f"测试集股票 vs 训练集: ✅ 测试集所有股票都在训练集中出现过")

# 看看股票的每日存续情况
print(f"\n每只股票的平均交易日数: {train_raw.groupby('股票代码').size().mean():.0f} 天")
print(f"最少交易日数: {train_raw.groupby('股票代码').size().min()} 天")
print(f"最多交易日数: {train_raw.groupby('股票代码').size().max()} 天")

# ===== 5. 探查：每个交易日有多少只股票 =====
print("\n" + "-" * 40)
print("【4】每日股票数量")
print("-" * 40)

daily_count = train_raw.groupby("日期").size()
print(f"\n每日股票数量:")
print(f"  最少: {daily_count.min()} 只")
print(f"  最多: {daily_count.max()} 只")
print(f"  均值: {daily_count.mean():.0f} 只")
print(f"  中位数: {daily_count.median():.0f} 只")

# 哪几天股票数量异常少？
low_days = daily_count[daily_count < daily_count.quantile(0.1)]
if len(low_days) > 0:
    print(f"\n  股票数最少的5天:")
    for date, count in low_days.nsmallest(5).items():
        print(f"    {date.date()}: {count} 只")

# ===== 6. 探查：数值列的统计 =====
print("\n" + "-" * 40)
print("【5】数值列基本统计")
print("-" * 40)

numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"]
for col in numeric_cols:
    if col in train_raw.columns:
        print(f"\n  {col}:")
        print(f"    缺失: {train_raw[col].isna().sum():,} / {len(train_raw):,}")
        if train_raw[col].isna().sum() < len(train_raw):
            print(f"    范围: [{train_raw[col].min():.4f}, {train_raw[col].max():.4f}]")

# ===== 7. 输出结论到文件 =====
with open("outputs/step1/data_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"训练集: {len(train_raw):,} 行 × {len(train_raw.columns)} 列\n")
    f.write(f"测试集: {len(test_raw):,} 行 × {len(test_raw.columns)} 列\n")
    f.write(f"训练日期: {train_raw['日期'].min().date()} ~ {train_raw['日期'].max().date()} ({daily_count.nunique()} 天)\n")
    f.write(f"测试日期: {test_raw['日期'].min().date()} ~ {test_raw['日期'].max().date()} ({test_raw['日期'].nunique()} 天)\n")
    f.write(f"训练集股票数: {len(train_stocks)}\n")
    f.write(f"每日股票数: {daily_count.min()}~{daily_count.max()}, 中位数: {daily_count.median():.0f}\n")

print("\n" + "=" * 60)
print("✅ Step 1 完成 → 产出: outputs/step1/data_summary.txt")
print("=" * 60)
