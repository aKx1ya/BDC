# BDC 2026 大数据竞赛工作仓库

这个仓库用于整理、复现和迭代 2026 大数据挑战赛相关材料。当前核心任务是：基于沪深 300 成分股历史数据，预测未来 5 个交易日内更有收益潜力的股票组合，最终生成符合比赛格式的 `result.csv`。

本仓库不是单一脚本项目，而是一个团队协作型研究工作区，主要包含四类内容：

- 官方参考代码与规则说明
- 我们自己的策略文档和 workflow 迭代记录
- 本地实验流水线、历史实验输出和结果文件
- 外部公开数据抓取、特征资产和数据说明

## 比赛任务概括

比赛要求参赛队伍在预测日 `T` 基于当时已经可获得的数据，提交最多 5 只沪深 300 成分股及其权重。官方评测时使用未来真实行情计算组合收益：

```text
买入：T+1 交易日开盘价
卖出：T+5 交易日开盘价
目标：组合加权收益率尽可能高，并超过官方基准程序
```

最终提交文件固定为：

```csv
stock_id,weight
000408,0.2
000975,0.2
002028,0.2
600372,0.2
600036,0.2
```

关键约束：

- 文件名必须是 `result.csv`
- 列名必须是 `stock_id,weight`
- 股票数量不超过 5 只
- 股票代码不能重复
- 权重非负，权重总和不超过 1
- 未分配权重视为现金，现金收益率默认为 0

更完整的赛题与代码规范整理见 [`赛题描述与代码规范详细总结.md`](./赛题描述与代码规范详细总结.md)。

## 当前策略主线

我们当前不是直接照搬官方 baseline，而是以官方代码作为参考起点，逐步构建自己的七步策略流程：

```text
1. 获取原始行情和可用公开数据
2. 清洗数据并做特征工程，形成特征表
3. 构造标签、滑动窗口和排序样本
4. 按时间切分训练区间和本地评估区间
5. 训练排序模型，输出 Top30 候选池
6. 精排 Top30，生成最终 Top5 和 result.csv
7. 用未来行情评分、复盘并进入下一轮实验
```

一个重要边界是：

```text
第 5 步只负责输出 candidate_top30.csv
第 6 步才负责从 Top30 中精排出最终 result.csv
```

也就是说，训练模型不是终点。真正影响提交结果的还有候选池召回质量、行业/风险约束、权重分配、事件过滤和复盘迭代。

策略主文档见 [`Experiment/策略流程与实验方案.md`](./Experiment/策略流程与实验方案.md)。

## 仓库结构

```text
.
├── THU-BDC2026-main/
├── local_baseline_experiment/
├── bigdata_challenge/
├── Experiment/
├── sample_experiment/
├── Trial_2/
├── baseline
├── 赛题描述与代码规范详细总结.md
├── .gitattributes
└── .gitignore
```

### `THU-BDC2026-main/`

官方参考代码和官方 baseline 的本地副本。

它的主要价值是：

- 理解官方基准程序的训练、预测和评分口径
- 对照比赛要求中的输入输出格式
- 参考 `StockTransformer` 排序学习建模方式
- 复现官方基线，作为我们后续实验必须超过的比较对象

目录内包含：

- `code/src/config.py`：训练与推理配置
- `code/src/model.py`：`StockTransformer` 模型
- `code/src/train.py`：训练脚本
- `code/src/predict.py`：预测脚本
- `code/src/utils.py`：特征工程与数据集构建
- `data/`：官方/参考训练数据
- `model/60_158+39/`：已有模型权重、标准化器和配置
- `output/result.csv`：预测输出样例
- `test/`：本地评分与赛事方测试方法相关材料

运行方式可参考 [`THU-BDC2026-main/README.md`](./THU-BDC2026-main/README.md)：

```bash
cd THU-BDC2026-main
uv sync
sh train.sh
sh test.sh
```

注意：这个目录是官方参考，不代表我们的最终方案。

### `local_baseline_experiment/`

这是我们之前本地跑过的一套实验流水线。虽然名字里曾经有 baseline，但它不是官方意义上的 baseline，因此现在命名为 `local_baseline_experiment`。

它的主要价值是：

- 提供一个离线优先的训练、预测、验证闭环
- 实现七步流程的早期版本
- 生成 `candidate_top30.csv`、`ranking_log.csv` 和 `result.csv`
- 支持本地评估与结果格式校验

核心逻辑：

```text
app/data/train.csv
  -> 特征工程与标签构造
  -> LightGBM 或 sklearn fallback 模型
  -> candidate_top30.csv
  -> 风险 gate + 精排 overlay
  -> app/output/result.csv
```

常用命令：

```bash
cd local_baseline_experiment
sh init.sh
sh train.sh
sh test.sh
python app/code/src/validate_result.py app/output/result.csv
python app/code/src/evaluate.py --result app/output/result.csv --test app/data/test.csv
```

更多说明见 [`local_baseline_experiment/readme.md`](./local_baseline_experiment/readme.md)。

### `bigdata_challenge/`

这是数据抓取与数据资产方向的工作目录，主要来自同学基于我们七步策略中 Step 1 和 Step 2 的尝试。

当前定位：

- 建立可复现的数据资产层
- 抓取公开可用数据
- 逐步补齐行情、北向资金、融资融券、资金流、行业板块和基本面数据
- 生成后续建模可用的特征矩阵

流水线位于 `bigdata_challenge/data_fetcher/`：

```text
01_price_volume.py      # 个股日线行情
02_northbound_flow.py   # 北向资金
03_margin_trading.py    # 融资融券
04_money_flow.py        # 资金流向
05_sector_momentum.py   # 行业板块动量
06_fundamental.py       # 基本面与业绩预告
07_feature_engine.py    # 特征工程汇总
run_all.py              # 一键运行
config.py               # 路径、时间、股票池配置
```

常用命令：

```bash
cd bigdata_challenge/data_fetcher

# 默认运行 Step 1-6，跳过特征工程
python3 run_all.py

# 只运行某一步
python3 run_all.py --step 1

# 从第 3 步开始继续跑
python3 run_all.py --from 3

# 包含 Step 7 特征工程
python3 run_all.py --with-feature
```

数据输出：

- `bigdata_challenge/data/raw/`：原始抓取数据
- `bigdata_challenge/data/features/feature_matrix.csv`：特征矩阵

指标体系说明见 [`bigdata_challenge/data_fetcher/README_指标说明.md`](./bigdata_challenge/data_fetcher/README_指标说明.md)。

### `Experiment/`

这是策略、流程和实验组织的主目录。

当前最重要的文件：

- [`Experiment/策略流程与实验方案.md`](./Experiment/策略流程与实验方案.md)：七步策略主文档
- [`Experiment/策略流程图.md`](./Experiment/策略流程图.md)：流程结构图
- [`Experiment/workflow_0.1/`](./Experiment/workflow_0.1/)：第一个 workflow 版本

`workflow_0.1` 的含义是：当前我们基于总策略，对 Step 1 数据获取和 Step 2 特征工程/初步筛选做了一版微调。这个目录下分成两类材料：

```text
Experiment/workflow_0.1/
├── strategy/
└── experiments/
```

- `strategy/`：这一版策略的思考、修改点和边界
- `experiments/`：基于这一版策略跑出的实验记录、输出和复盘

如果后续某次实验结论改变了策略本身，就应该开新目录，例如：

```text
Experiment/workflow_0.2/
```

而不是把所有想法都继续塞进 `workflow_0.1`。

### `sample_experiment/`

这是七步流程的样例/教学式拆解目录，用来帮助理解从数据探索到样本构造、时间切分的过程。

当前包含：

- `step1_explore_data.py`
- `step2_engineer_features.py`
- `step3_build_samples.py`
- `step4_split_data.py`
- `outputs/step1/` 到 `outputs/step4/`
- `七步流程详解.md`
- `修改对照.md`

它更适合作为理解流程的 walkthrough，不建议直接当作最终实验主线。

### `Trial_2/`

历史试验材料和表格快照目录。当前主要保留早期探索过程中产生的文件，方便追溯。

后续如果基于这些文件产生新的完整训练/预测/评分结果，建议把新结果整理到 `Experiment/workflow_x.x/experiments/exp_xxx_*` 中，而不是继续把新材料散放在 `Trial_2/`。

### `baseline`

这是 GitHub 远端原本已有的 gitlink/submodule 条目，当前保留以避免覆盖远端已有内容。

请注意它和 `local_baseline_experiment/` 不是一回事：

- `baseline`：远端仓库原有引用
- `local_baseline_experiment/`：我们本地历史实验流水线

日常实验优先看 `local_baseline_experiment/`、`THU-BDC2026-main/` 和 `Experiment/`。

## 实验组织规范

我们采用“策略 workflow + 独立实验目录”的方式组织后续迭代。

推荐每次完整训练/预测/评分都创建一个独立实验目录：

```text
Experiment/workflow_0.1/experiments/exp_001_实验主题/
├── README.md
├── code_notes/
├── config/
├── outputs/
├── submission/
└── review/
```

每个实验至少记录：

- 实验目的：验证 workflow 中的哪个判断
- 代码来源：来自官方参考、本地实验流水线、数据抓取脚本，还是新代码
- 数据来源：使用哪份 `train.csv`、哪份抓取数据、哪份特征表
- 关键改动：相比上一轮改了什么
- 输出结果：候选池、排序日志、`result.csv`、评分结果
- 结论：保留、修改、放弃，或推进到下一个 workflow

实验目录只保存本次实验相关材料，不重复复制大型通用数据。大型数据和通用代码应通过路径引用，例如：

```text
data_ref = bigdata_challenge/data/features/feature_matrix.csv
source_code_ref = local_baseline_experiment/app/code/src
strategy_ref = Experiment/workflow_0.1/strategy/0.1_Step-2_特征工程与初步筛选流程与思考逻辑.md
```

## 数据与特征原则

本项目最需要警惕的是未来信息泄漏。任何特征进入模型之前，都要能回答：

```text
预测日 T 当天，这条数据是否已经真实公开并可获得？
```

当前数据层的原则：

- 原始数据放在 `data/raw/`，尽量不手工修改
- 特征矩阵由脚本生成，避免手工拼表不可复现
- 基本面数据必须按公告日对齐，不能按报告期提前使用
- 股票池应尽量使用历史截面，避免用未来成分股名单回测过去
- T+1 到 T+5 必须按交易日计算，不能简单按自然日加 5 天
- 外部数据必须免费、公开、可复现，并在后续提交材料中说明来源

## Git 与大文件管理

本仓库使用 Git LFS 管理数据、模型和压缩包等大文件。首次 clone 后建议执行：

```bash
git lfs install
git lfs pull
```

当前 `.gitattributes` 会将以下类型交给 LFS：

```text
*.csv
*.xlsx
*.pkl
*.pth
*.zip
*.tar
**/events.out.tfevents.*
```

这意味着 GitHub 页面上看到的大型数据和模型文件通常是 LFS 指针文件，真实内容需要通过 Git LFS 拉取。

本仓库 `.gitignore` 已忽略：

- `.DS_Store`
- `__pycache__/`
- `*.pyc`
- `.claude/`
- Office 临时文件 `~$*`
- `.venv/`

## 推荐上手路径

如果你是第一次看这个仓库，建议按下面顺序：

1. 先读 [`赛题描述与代码规范详细总结.md`](./赛题描述与代码规范详细总结.md)，理解任务、评分和 `result.csv` 格式。
2. 再读 [`Experiment/策略流程与实验方案.md`](./Experiment/策略流程与实验方案.md)，理解我们自己的七步策略。
3. 看 [`THU-BDC2026-main/README.md`](./THU-BDC2026-main/README.md)，了解官方参考 baseline 如何训练和预测。
4. 看 [`local_baseline_experiment/readme.md`](./local_baseline_experiment/readme.md)，了解我们本地早期实验流水线。
5. 看 [`bigdata_challenge/CLAUDE.md`](./bigdata_challenge/CLAUDE.md) 和 [`bigdata_challenge/data_fetcher/README_指标说明.md`](./bigdata_challenge/data_fetcher/README_指标说明.md)，理解数据抓取和特征资产。
6. 在 [`Experiment/workflow_0.1/`](./Experiment/workflow_0.1/) 下查看当前策略微调，并把新实验放入 `experiments/`。

## 当前状态

截至当前版本，仓库已经具备：

- 官方参考 baseline 的本地副本
- 一套我们自己的离线实验流水线
- 初步的数据抓取与特征工程资产
- 七步策略主文档
- workflow 级别的策略迭代目录
- Git LFS 大文件管理配置

下一阶段重点应该是：

- 将 `bigdata_challenge` 中的数据资产接入统一实验流程
- 补齐历史沪深 300 成分股截面与交易日历
- 严格记录每次实验的配置、候选池、评分和复盘
- 对比官方 baseline、local baseline 和新策略的收益表现
- 将有效实验结论沉淀到新的 workflow 版本

## 重要提醒

这个仓库是研究和协作仓库，不等同于最终提交包。正式提交前应单独整理：

- 最终 `result.csv`
- 可复现训练/预测代码
- 必要数据与模型文件
- 外部数据来源说明
- 大文件 md5 或版本记录
- Docker/离线环境复现说明

