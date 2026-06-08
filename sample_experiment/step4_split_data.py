"""
STEP 4: 数据切分 — 时间切分与 Walk-forward 验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略文档章节：第 4 步 数据切分 Split
核心问题：怎么把 460 天的排序样本切成「训练集」和「验证集」，才能模拟真实提交？
         怎么防止模型在训练时偷看到验证集的答案？
输入来源：复用 Step 3 的过滤逻辑（内置，保证独立可运行）
输出产物：outputs/step4/split_summary.txt       （切分方案总览）
          outputs/step4/split_detail.csv        （每天属于 train/val/gap/test）
          outputs/step4/walk_forward_plan.txt   （Walk-forward 轮次表）

比喻：Step 1 买菜 → Step 2 备菜 → Step 3 出题 → Step 4 分考卷
      把 460 道排序题分成「练习题」和「模拟考」，中间还留了隔离带防止偷看答案

关键概念：
  为什么不能随机打乱？
    → 金融数据是按时间顺序的，随机打乱 = 用未来预测过去 = 作弊
  为什么要 Gap？
    → label = 未来 5 天收益。如果训练集最后一天是 3 月 1 日，验证集第一天是 3 月 2 日
    → 模型在训练时看到了 3 月 1 日的 label（= 3 月 2 日~6 日的收益）
    → 3 月 1 日的特征 + 3 月 2 日已发生 → 模型可能猜到验证集答案
    → 留 5 天 Gap 把这段"模糊地带"隔离掉

可调变量：
  - TRAIN_WINDOW: 训练窗口 252 个交易日（≈1 年）
  - GAP_DAYS: 隔离带 5 个交易日
  - EVAL_DAYS: 每次评估 5 个交易日
  - TRAIN_RATIO: train.csv 内部切分比例 0.80
  - WALK_FORWARD_STEP: Walk-forward 推进步长 5 个交易日
  - FINAL_TEST_DAYS: 最终测试留出 5 个交易日
"""

import pandas as pd
import numpy as np

# ===== 可调变量 =====
TRAIN_WINDOW = 252         # 训练窗口：252 个交易日（≈1 年）
GAP_DAYS = 5               # 隔离带：5 个交易日（贴合 5 日标签）
EVAL_DAYS = 5              # 每次评估：5 个交易日
TRAIN_RATIO = 0.80         # train 内部切分比例
WALK_FORWARD_STEP = 5      # Walk-forward 每轮推进步长
FINAL_TEST_DAYS = 5        # 最终测试留出天数
SEQUENCE_LENGTH = 60        # 特征需要的历史窗口（来自 Step 3）
PREDICTION_HORIZON = 5      # 预测周期（来自 Step 3）

# ===== 1. 加载数据（复用 Step 3 过滤逻辑） =====
print("=" * 60)
print("STEP 4: 时间切分与 Walk-forward 验证")
print("=" * 60)

print("\n📂 读取数据: data/train.csv")
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
raw = raw.dropna(subset=["stock_id", "date", "open", "close", "high", "low"])
raw = raw.sort_values(["stock_id", "date"]).drop_duplicates(["stock_id", "date"]).reset_index(drop=True)

# 简化版特征工程（复用 Step 3 逻辑，只构造标签和 history_count）
df = raw.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = df.groupby("stock_id", group_keys=False)
df["history_count"] = g.cumcount() + 1
df["future_open_t1"] = g["open"].shift(-1)
df["future_open_t5"] = g["open"].shift(-PREDICTION_HORIZON)
df["label"] = (df["future_open_t5"] - df["future_open_t1"]) / df["future_open_t1"]
df["label"] = df["label"].replace([np.inf, -np.inf], np.nan)

# 过滤（复用 Step 3 逻辑，先过滤再 fillna）
df_filtered = df[df["history_count"] >= SEQUENCE_LENGTH].copy()
df_filtered = df_filtered.dropna(subset=["label"])
num_cols = df_filtered.select_dtypes(include=[np.number]).columns
df_filtered[num_cols] = df_filtered[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# 获取所有有效排序样本日期（每天都有足够多股票）
daily_counts = df_filtered.groupby("date").size()
valid_dates = sorted(daily_counts[daily_counts >= 200].index)  # 至少 200 只股票才算有效交易日
print(f"有效交易日: {len(valid_dates)} 天")
print(f"日期范围:   {valid_dates[0].date()} ~ {valid_dates[-1].date()}")

# ===== 2. 两层切分 =====
print("\n" + "-" * 40)
print("【两层切分】")
print("-" * 40)

# --- 第一层：最终 test 从末尾留出 ---
final_test_dates = valid_dates[-FINAL_TEST_DAYS:]
remaining_dates = valid_dates[:-FINAL_TEST_DAYS]  # 剩下的用于 train/val 内部切分

print(f"\n第一层 — 最终测试集（模拟正式提交）:")
print(f"  留出最后 {FINAL_TEST_DAYS} 个交易日")
print(f"  测试日期: {final_test_dates[0].date()} ~ {final_test_dates[-1].date()}")
print(f"  这些日期完全不参与训练和调参，只在最后评分时用一次")

# --- 第二层：剩余日期内切 train / validation ---
split_idx = int(len(remaining_dates) * TRAIN_RATIO)
inner_train_dates = remaining_dates[:split_idx]
val_dates_raw = remaining_dates[split_idx:]

# Gap：train 结束日到 validation 开始日之间留 GAP_DAYS 个交易日
# 如果 train 结束日是 T，那 validation 从 T + GAP_DAYS + 1 开始
# 但因为日期不是连续的（有周末），所以用索引偏移来实现
gap_start_idx = split_idx
gap_end_idx = min(split_idx + GAP_DAYS, len(remaining_dates))
gap_dates = remaining_dates[gap_start_idx:gap_end_idx]
val_dates = remaining_dates[gap_end_idx:]

print(f"\n第二层 — train.csv 内部切分 (80/20 + 5日Gap):")
print(f"  总内部日期:  {len(remaining_dates)} 天（已扣除最终测试集）")
print(f"  inner train: {len(inner_train_dates)} 天 ({len(inner_train_dates)/len(remaining_dates)*100:.0f}%)")
print(f"  Gap 隔离带:  {len(gap_dates)} 天")
print(f"  validation:  {len(val_dates)} 天 ({len(val_dates)/len(remaining_dates)*100:.0f}%)")

print(f"\n  时间线:")
print(f"    inner train: {inner_train_dates[0].date()} ~ {inner_train_dates[-1].date()}")
print(f"    Gap:         {gap_dates[0].date()} ~ {gap_dates[-1].date()}")
print(f"    validation:  {val_dates[0].date()} ~ {val_dates[-1].date()}")

# ===== 3. Gap 为什么重要 =====
print(f"\n" + "-" * 40)
print(f"【Gap 隔离带详解】")
print(f"-" * 40)

print(f"""
  Gap 解决的问题：
    label = (open_T+5 - open_T+1) / open_T+1

    假设没有 Gap：
      train 最后一天 T = {inner_train_dates[-1].date()}
      这一天训练样本的 label = 从 {inner_train_dates[-1].date()} 后 1 天到后 5 天的收益
      如果 validation 第一天紧接着 T+1，模型在训练时已经"见过"这段时间的行情
      → 模型可能通过 label 间接学到验证集的信息

    有 Gap 之后：
      train 最后一天 T = {inner_train_dates[-1].date()}
      Gap 把 T 到 T+{GAP_DAYS} 的 {GAP_DAYS} 天隔离掉
      validation 从 {val_dates[0].date()} 开始
      → train 的 label 只覆盖到 Gap 范围内
      → validation 的数据对训练过程完全未知
      → 模拟真实提交场景：我们训练时不知道验证期的任何行情
""")

# ===== 4. Walk-forward 验证计划 =====
print("-" * 40)
print("【Walk-forward 验证计划】")
print("-" * 40)

# Walk-forward：每次用过去 TRAIN_WINDOW 天训练，往前推 WALK_FORWARD_STEP 天评估
# 有效的 Walk-forward 起点：从 TRAIN_WINDOW 天之后开始
wf_start_idx = TRAIN_WINDOW
wf_rounds = []
for start_idx in range(wf_start_idx, len(valid_dates) - GAP_DAYS - EVAL_DAYS, WALK_FORWARD_STEP):
    train_start = valid_dates[start_idx - TRAIN_WINDOW]
    train_end = valid_dates[start_idx - 1]
    gap_start = valid_dates[start_idx]
    gap_end = valid_dates[min(start_idx + GAP_DAYS - 1, len(valid_dates) - 1)]
    eval_start = valid_dates[min(start_idx + GAP_DAYS, len(valid_dates) - EVAL_DAYS)]
    eval_end = valid_dates[min(start_idx + GAP_DAYS + EVAL_DAYS - 1, len(valid_dates) - 1)]

    # 确保 eval_end 不超出范围
    eval_end_idx = min(start_idx + GAP_DAYS + EVAL_DAYS - 1, len(valid_dates) - 1)
    if eval_end_idx < len(valid_dates):
        eval_end = valid_dates[eval_end_idx]
        wf_rounds.append({
            "round": len(wf_rounds) + 1,
            "train_start": train_start, "train_end": train_end,
            "gap_start": gap_start, "gap_end": gap_end,
            "eval_start": eval_start, "eval_end": eval_end,
        })

n_wf_rounds = min(len(wf_rounds), 10)  # 最多展示 10 轮
print(f"\n共可执行 {len(wf_rounds)} 轮 Walk-forward（展示前 {n_wf_rounds} 轮）:")
print(f"\n{'轮次':<6} {'训练区间':<28} {'Gap':<28} {'评估区间':<28}")
print("-" * 92)
for r in wf_rounds[:n_wf_rounds]:
    train_range = f"{r['train_start'].date()} ~ {r['train_end'].date()}"
    gap_range = f"{r['gap_start'].date()} ~ {r['gap_end'].date()}"
    eval_range = f"{r['eval_start'].date()} ~ {r['eval_end'].date()}"
    print(f"第{r['round']}轮    {train_range:<28} {gap_range:<28} {eval_range:<28}")

print(f"\n每轮做的事：")
print(f"  1. 用训练区间数据训练模型（第 5 步）")
print(f"  2. 在评估区间生成候选池 + 精排 Top5（第 5~6 步）")
print(f"  3. 用评估区间真实行情评分（第 7 步）")
print(f"  4. 记录本轮收益")
print(f"  5. 时间向前推进 {WALK_FORWARD_STEP} 天，重复")

# ===== 5. 每天属于哪个集合 =====
print(f"\n" + "-" * 40)
print(f"【每日归属汇总】")
print(f"-" * 40)

# 构造每日归属表
date_roles = []
for d in valid_dates:
    if d in final_test_dates:
        role = "final_test"
    elif d in inner_train_dates:
        role = "inner_train"
    elif d in gap_dates:
        role = "gap"
    elif d in val_dates:
        role = "validation"
    else:
        role = "unassigned"

    n_stocks = daily_counts.get(d, 0)
    date_roles.append({"date": d, "role": role, "n_stocks": int(n_stocks)})

role_df = pd.DataFrame(date_roles)
role_counts = role_df["role"].value_counts()
print(f"\n  各集合天数:")
for role, label in [
    ("inner_train", "训练集"),
    ("gap", "Gap 隔离带"),
    ("validation", "验证集"),
    ("final_test", "最终测试集"),
]:
    count = role_counts.get(role, 0)
    print(f"    {label}: {count} 天")

# ===== 6. 最终全量重训说明 =====
print(f"\n" + "-" * 40)
print(f"【最终全量重训】")
print(f"-" * 40)
print(f"""
  正式提交前的最后一步：
    1. 用 Walk-forward 选好特征、模型、参数、权重策略
    2. 合并 inner_train + validation（去掉 Gap）
    3. 用所有合法且有完整标签的数据重训模型
    4. 固定 best_iteration / 参数
    5. 用最新 60 天窗口预测
    6. 进入第 5 步生成 candidate_top30.csv

  注意：最终测试集（{final_test_dates[0].date()} ~ {final_test_dates[-1].date()}）
        在整个过程中完全不参与训练和调参
        只能在最终提交前评分一次
""")

# ===== 7. 输出文件 =====
print("=" * 40)
print("【输出文件】")
print("=" * 40)

# 7.1 切分汇总
with open("outputs/step4/split_summary.txt", "w", encoding="utf-8") as f:
    f.write("【Step 4 数据切分汇总】\n\n")
    f.write(f"有效交易日总数: {len(valid_dates)}\n")
    f.write(f"日期范围: {valid_dates[0].date()} ~ {valid_dates[-1].date()}\n\n")

    f.write("--- 第一层：最终测试集 ---\n")
    f.write(f"留出最后 {FINAL_TEST_DAYS} 个交易日\n")
    f.write(f"日期: {final_test_dates[0].date()} ~ {final_test_dates[-1].date()}\n\n")

    f.write("--- 第二层：train.csv 内部切分 ---\n")
    f.write(f"inner_train: {len(inner_train_dates)} 天 ({inner_train_dates[0].date()} ~ {inner_train_dates[-1].date()})\n")
    f.write(f"Gap:         {len(gap_dates)} 天 ({gap_dates[0].date()} ~ {gap_dates[-1].date()})\n")
    f.write(f"validation:  {len(val_dates)} 天 ({val_dates[0].date()} ~ {val_dates[-1].date()})\n\n")

    f.write("--- 核心参数 ---\n")
    f.write(f"训练窗口: {TRAIN_WINDOW} 交易日\n")
    f.write(f"Gap: {GAP_DAYS} 交易日\n")
    f.write(f"评估窗口: {EVAL_DAYS} 交易日\n")
    f.write(f"切分比例: {TRAIN_RATIO*100:.0f}/{100-TRAIN_RATIO*100:.0f}\n")

    f.write(f"\n--- Walk-forward 轮数 ---\n")
    f.write(f"共 {len(wf_rounds)} 轮\n")

print(f"  → outputs/step4/split_summary.txt")

# 7.2 每日归属明细
role_df["date"] = role_df["date"].dt.strftime("%Y-%m-%d")
role_df.to_csv("outputs/step4/split_detail.csv", index=False, encoding="utf-8")
print(f"  → outputs/step4/split_detail.csv  ({len(role_df)} 天 × role/stock数)")

# 7.3 Walk-forward 计划
with open("outputs/step4/walk_forward_plan.txt", "w", encoding="utf-8") as f:
    f.write("【Walk-forward 验证计划】\n\n")
    f.write(f"总轮数: {len(wf_rounds)}\n")
    f.write(f"每轮训练窗口: {TRAIN_WINDOW} 天\n")
    f.write(f"每轮 Gap: {GAP_DAYS} 天\n")
    f.write(f"每轮评估: {EVAL_DAYS} 天\n")
    f.write(f"步长: {WALK_FORWARD_STEP} 天\n\n")
    f.write(f"{'轮次':<6} {'训练区间':<28} {'Gap':<28} {'评估区间':<28}\n")
    f.write("-" * 92 + "\n")
    for r in wf_rounds[:15]:  # 输出前 15 轮
        train_range = f"{r['train_start'].date()} ~ {r['train_end'].date()}"
        gap_range = f"{r['gap_start'].date()} ~ {r['gap_end'].date()}"
        eval_range = f"{r['eval_start'].date()} ~ {r['eval_end'].date()}"
        f.write(f"第{r['round']}轮    {train_range:<28} {gap_range:<28} {eval_range:<28}\n")

print(f"  → outputs/step4/walk_forward_plan.txt")

print("\n" + "=" * 60)
print("✅ Step 4 完成 → 产出 3 个文件在 outputs/step4/")
print("=" * 60)
