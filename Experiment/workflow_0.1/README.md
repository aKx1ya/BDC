# workflow_0.1

这个目录记录 0.1 版策略微调，以及基于这版策略产生的实验。

## 目录含义

- `strategy/`：这一版 workflow 的策略和思考文档。
- `workflow_config.yaml`：这一版 workflow 的机器可读调度地图，供 `Experiment/shared/` 读取。
- `docs/`：这一版 workflow 的长期操作说明、流程图和交接文档。
- `pipelines/`：这一版 workflow 专用的数据整理脚本，把底层 raw 数据转成标准输出。
- `experiments/`：基于这一版 workflow 跑出来的实验代码记录、结果和复盘。

## 使用方式

1. 先读 `../ACTIVE_WORKFLOW.md`，确认当前激活的是 `workflow_0.1`。
2. 再在 `strategy/` 里确认本版 Step-1 / Step-2 的策略边界；Step-3 以后如果没有 workflow 专门改写，则回到 `../策略流程与实验方案.md`。
3. 正式运行 Step-1 时，使用本 workflow 的健康调度入口。
4. 调度入口会联网抓取 raw 数据、验收 raw、生成标准 CSV、验收输出并写运行报告。
5. 每次完整实验都在 `experiments/` 下新建一个实验目录。
6. 实验目录只保存本次实验相关材料，不反复复制大型通用数据。
7. 如果实验结论导致策略变化，再开下一个 `workflow_0.2/`。

## 可迁移 shared 入口

`workflow_0.1` 现在已经接入迁移版 shared 调度层：

```bash
/opt/miniconda3/bin/python3 Experiment/shared/validators/validate_workflow_config.py --workflow workflow_0.1
/opt/miniconda3/bin/python3 Experiment/shared/runners/run_step.py --step 7 --mode freeze-only --dry-run --print-context
```

它的含义是：

```text
ACTIVE_WORKFLOW.md 决定当前 workflow 和 active_stage
workflow_config.yaml 决定 Step-N 应该调用哪个 runner
shared/runners/run_step.py 负责统一分发
workflow_0.1/run_stepN.py 负责本 Step 的真实健康执行
```

当前阶段 `run_step1.py` 到 `run_step7.py` 仍保留在 `workflow_0.1/`。等未来新建 `workflow_0.2` 后，再把重复度最高、策略无关的 build / validate / report 逻辑逐步沉淀到 `Experiment/shared/`。

## 长期说明文档

这些文档用于解释 workflow 的长期流程，不是某一次实验报告：

- `docs/Step-1_正式健康版运作流程.md`
- `docs/Step-2_正式健康版体系设计.md`
- `docs/Step-3_正式健康版体系设计.md`
- `docs/Step-4_正式健康版体系设计.md`
- `docs/Step-5_正式健康版体系设计.md`
- `docs/Step-6_正式健康版体系设计.md`
- `docs/Step-7_正式健康版体系设计.md`

### 正式 Step-1 一键入口

完整图解说明见：

- `docs/Step-1_正式健康版运作流程.md`

正式 Step-1 不再手动分散执行底层脚本，而是运行：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step1.py
```

默认流程：

```text
读取 Experiment/ACTIVE_WORKFLOW.md
-> 确认 active_workflow=workflow_0.1 且 active_stage=Step-1
-> 联网执行 bigdata_challenge/data_fetcher/run_all.py --step 1
-> 校验 bigdata_challenge/data/raw/ 中的 raw 数据
-> 生成 outputs/step1/ 四张标准 CSV
-> 校验四张标准 CSV
-> 写入 notes/step1_run_report.md
```

正式 Step-1 成功条件：

```text
当前沪深300股票数 = 300
300只股票都有 daily 数据
当前300只股票最新日期一致
daily 表无 股票代码+日期 重复
step1_stock_summary.csv 行数 = 300
板块划分未匹配数量 = 0
四张标准 CSV 都存在且表头符合 workflow_0.1_csv_v1
manifest 记录 latest_T、date_start、date_end、raw_交易日数、data_source、generated_at
notes/step1_run_report.md 存在
```

Step-5 当前不属于正式健康链路。workflow_0.1 的 Step-1 使用 `stock_industry.csv` 做行业映射，并自聚合六大风格板块；外部板块行情接口后续只有在策略明确要求时才纳入。

### 正式 Step-2 一键入口

正式 Step-2 读取一个已经健康通过的 Step-1 实验输出，不联网抓数，不重新定义股票池，也不生成最终 `result.csv`。

正式运行命令：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step2.py
```

默认流程：

```text
读取 Experiment/ACTIVE_WORKFLOW.md
-> 确认 active_workflow=workflow_0.1 且 active_stage=Step-2
-> 自动寻找最近一个 SUCCESS 的 Step-1 实验
-> 校验 Step-1 输入健康
-> 生成 outputs/step2/ 七张标准 CSV
-> 校验 Step-2 输出
-> 写入 notes/step2_run_report.md
```

也可以手动指定输入 Step-1 实验：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step2.py \
  --step1-experiment exp_20260616_step1_workflow_0_1
```

正式 Step-2 成功条件：

```text
读取的 Step-1 实验是 SUCCESS
Step-2 latest_T 与 Step-1 latest_T 一致
5 个核心输出全部存在
2 个派生视图默认生成
所有 CSV 表头符合 workflow_0.1_csv_v1
feature_table_daily 无 股票代码 + 日期 重复
sector_feature_table 无 日期 + 板块划分 重复
latest_t_screen 只包含 latest_T
feature_metadata 覆盖新增特征并写明防泄漏说明
step2_data_manifest 记录 input_step1_path、feature_set_id、latest_T、generated_at
notes/step2_run_report.md 存在
```

### Step-3 健康体系设计

Step-3 当前没有在 `workflow_0.1/strategy/` 下单独改写策略，策略源头仍然是七步总策略：

```text
Experiment/策略流程与实验方案.md
```

本 workflow 已经建立 Step-3 的健康体系设计文档：

```text
docs/Step-3_正式健康版体系设计.md
```

它定义了后续正式 Step-3 应该如何：

```text
读取健康 Step-2 输出
-> 构造过去 60 日输入窗口
-> 构造未来 5 日开盘收益标签
-> 生成样本表、窗口索引、group 信息、排序标签和 manifest
-> 校验窗口完整、标签完整、无未来泄漏
-> 写入 notes/step3_run_report.md
```

当前状态：

```text
Step-3 体系设计：已建立
run_step3.py：已实现
build_step3_outputs.py：已实现
validate_step3.py：已实现
Step-3 tests：已实现
最近一次正式运行：exp_20260617_step3_workflow_0_1，Status=SUCCESS
```

### Step-4 健康体系设计

Step-4 当前没有在 `workflow_0.1/strategy/` 下单独改写策略，策略源头仍然是七步总策略：

```text
Experiment/策略流程与实验方案.md
```

本 workflow 已经建立 Step-4 的健康体系设计文档：

```text
docs/Step-4_正式健康版体系设计.md
```

它定义了后续正式 Step-4 应该如何：

```text
读取健康 Step-3 输出
-> 按样本日期做时间切分
-> 留出 final_test 日期
-> 设置 train / validation 之间的 Gap
-> 生成 walk-forward 计划
-> 生成最终全量重训计划
-> 校验无日期重叠、无随机打乱、Gap 足够、防泄漏说明完整
-> 写入 notes/step4_run_report.md
```

正式运行命令：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step4.py
```

默认流程：

```text
读取 Experiment/ACTIVE_WORKFLOW.md
-> 确认 active_workflow=workflow_0.1 且 active_stage=Step-4
-> 自动寻找最近一个 SUCCESS 的 Step-3 实验
-> 校验 Step-3 输入健康
-> 生成 outputs/step4/ 六张标准 CSV
-> 校验 Step-4 输出
-> 写入 notes/step4_run_report.md
```

正式 Step-4 输出：

```text
outputs/step4/
├── step4_split_detail.csv
├── step4_split_summary.csv
├── step4_walk_forward_plan.csv
├── step4_final_retrain_plan.csv
├── step4_split_manifest.csv
└── step4_leakage_check.csv
```

当前状态：

```text
Step-4 体系设计：已建立
run_step4.py：已实现
build_step4_outputs.py：已实现
validate_step4.py：已实现
Step-4 tests：已实现
最近一次正式运行：exp_20260617_step4_workflow_0_1，Status=SUCCESS
```

### Step-5 健康体系设计

Step-5 当前没有在 `workflow_0.1/strategy/` 下单独改写策略，策略源头仍然是七步总策略：

```text
Experiment/策略流程与实验方案.md
```

本 workflow 已经建立 Step-5 的健康体系设计文档：

```text
docs/Step-5_正式健康版体系设计.md
```

它定义了后续正式 Step-5 应该如何：

```text
读取健康 Step-2 特征
-> 读取健康 Step-3 样本和标签
-> 读取健康 Step-4 切分与 walk-forward 计划
-> 按 Step-4 的时间规则训练模型
-> 生成 walk-forward 样本外预测
-> 记录模型、特征、参数、随机种子和训练窗口
-> 用最终合法训练样本重训模型
-> 对最新预测日生成 Top30 候选池
-> 校验没有训练/验证/final_test 泄漏
-> 写入 notes/step5_run_report.md
```

正式运行命令：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step5.py
```

默认流程：

```text
读取 Experiment/ACTIVE_WORKFLOW.md
-> 确认 active_workflow=workflow_0.1 且 active_stage=Step-5
-> 自动寻找最近一个 SUCCESS 的 Step-4 实验
-> 通过 manifest 推断同链路 Step-3 和 Step-2
-> 校验 Step-2 / Step-3 / Step-4 输入健康且一致
-> 训练 baseline_correlation_rank 模型
-> 生成 outputs/step5/ 八张标准 CSV 和 models/step5/ 模型文件
-> 校验 Step-5 输出
-> 写入 notes/step5_run_report.md
```

正式 Step-5 预计输出：

```text
outputs/step5/
├── step5_model_registry.csv
├── step5_feature_set_used.csv
├── step5_walk_forward_predictions.csv
├── step5_walk_forward_metrics.csv
├── step5_feature_importance.csv
├── step5_candidate_top30.csv
├── step5_model_manifest.csv
└── step5_leakage_check.csv
```

当前状态：

```text
Step-5 体系设计：已建立
run_step5.py：已实现
build_step5_outputs.py：已实现
validate_step5.py：已实现
Step-5 tests：已实现
最近一次正式运行：exp_20260617_step5_workflow_0_1，Status=SUCCESS
当前模型版本：baseline_correlation_rank，用于打通健康链路；后续可升级 LightGBM Ranker
```

### Step-6 健康体系设计

Step-6 当前没有在 `workflow_0.1/strategy/` 下单独改写策略，策略源头仍然是七步总策略：

```text
Experiment/策略流程与实验方案.md
```

本 workflow 已经建立 Step-6 的健康体系设计文档：

```text
docs/Step-6_正式健康版体系设计.md
```

它定义了后续正式 Step-6 应该如何：

```text
读取健康 Step-5 候选池
-> 读取 Step-2 最新日可用特征、风险特征和板块特征
-> 只在 Top30 内执行 Hard Gates
-> 计算 refine_score 精排分
-> 按行业、相关性、风险和仓位约束构建组合
-> 选出最终 Top5 或少于5只
-> 分配权重
-> 生成 step6_result.csv
-> 校验格式、权重、来源和防泄漏
-> 写入 notes/step6_run_report.md
```

正式 Step-6 预计输出：

```text
outputs/step6/
├── step6_ranking_log.csv
├── step6_final_top5.csv
├── step6_result.csv
├── step6_weight_plan.csv
├── step6_refine_manifest.csv
└── step6_leakage_check.csv
```

当前状态：

```text
Step-6 体系设计：已建立
run_step6.py：已实现
build_step6_outputs.py：已实现
validate_step6.py：已实现
Step-6 tests：已实现
最近一次正式运行：exp_20260617_step6_workflow_0_1，Status=SUCCESS
当前精排版本：refine_set_v1_rule_top5_equal_weight，只在 Step-5 Top30 内精排并生成 step6_result.csv
```

### Step-7 健康体系设计

Step-7 当前没有在 `workflow_0.1/strategy/` 下单独改写策略，策略源头仍然是七步总策略：

```text
Experiment/策略流程与实验方案.md
```

本 workflow 已经建立 Step-7 的健康体系设计文档：

```text
docs/Step-7_正式健康版体系设计.md
```

它定义了后续正式 Step-7 应该如何：

```text
读取健康 Step-6 实验
-> 冻结 step6_result.csv
-> 校验 result.csv 官方格式
-> 在冻结之后读取 test.csv
-> 按官方口径计算 Final Score
-> 生成单股贡献明细
-> 生成评分 manifest 和 leakage_check
-> 写入 step7_run_report.md
-> 只把结果用于复盘和下一轮实验，不回改本轮 Step-6
```

正式 Step-7 预计输出：

```text
outputs/step7/
├── step7_frozen_result.csv
├── step7_score_summary.csv
├── step7_stock_contribution.csv
├── step7_score_manifest.csv
└── step7_leakage_check.csv
```

当前状态：

```text
Step-7 体系设计：已建立
run_step7.py：已实现
build_step7_outputs.py：已实现
validate_step7.py：已实现
Step-7 tests：已实现
最近一次正式运行：exp_20260617_step7_workflow_0_1，Status=FREEZE_ONLY_SUCCESS
local-score 尝试：exp_20260617_step7_local_score_workflow_0_1，Status=FAILED，原因是 test.csv 日期早于 Step-6 candidate_date
官方本地评分脚本：THU-BDC2026-main/test/score_self.py
本地 test.csv：THU-BDC2026-main/data/test.csv
```

## 与 bigdata_challenge 的关系

`bigdata_challenge/` 是执行层，只负责抓取和保存通用 raw 数据。

`workflow_0.1/` 是策略层和产出层，负责决定本版实验使用哪些 raw 数据，以及把它们整理成哪些标准输出。

当前 Step-1 的典型链路是：

```text
Experiment/ACTIVE_WORKFLOW.md
-> workflow_0.1/strategy/
-> workflow_0.1/run_step1.py
-> bigdata_challenge/data_fetcher/run_all.py --step 1
-> bigdata_challenge/data/raw/
-> workflow_0.1/pipelines/
-> workflow_0.1/experiments/.../outputs/step1/
-> workflow_0.1/experiments/.../notes/step1_run_report.md
```

## Step-1 / Step-2 CSV 输出格式标准

本节用于固定 `workflow_0.1` 后续实验的 Step-1 和 Step-2 输出格式。

来源样例是 `Trial_2/` 下的两个工作簿：

- `THU-BDC2026_Step1_数据资产_Trial_2.xlsx`
- `THU-BDC2026_Step2_特征工程_Trial_2.xlsx`

后续实验不再把多个表只塞进一个 Excel 工作簿里。每个 sheet 级别的逻辑表必须落成一个独立 CSV，并放入当前实验目录：

```text
Experiment/workflow_0.1/experiments/exp_xxx_实验主题/
└── outputs/
    ├── step1/
    └── step2/
```

### 通用 CSV 规则

- 编码统一使用 `UTF-8`。
- CSV 第一行必须是表头，不允许额外说明行。
- 不允许输出 pandas 默认索引列，例如 `Unnamed: 0`。
- 股票代码列统一名为 `股票代码`，必须保存为 6 位字符串，例如 `000001`，不能保存成 `1`。
- 日期统一使用 `YYYY-MM-DD`，例如 `2026-06-05`。
- 百分比类字段沿用 Trial_2 口径，使用“百分数数值”，不是小数比例。例如 `14.25` 表示 `14.25%`。
- 滚动窗口早期无法计算的字段允许为空，但不能用未来数据补齐。
- 每个 CSV 必须保持本节规定的核心列名和核心列顺序；新增字段按下面的迭代规则处理。

### CSV 规范迭代规则

当前 Trial_2 的表头定义为 `schema_version = workflow_0.1_csv_v1` 的基础契约。

后续实验采用“核心列 + 扩展列”的方式迭代：

- 核心列：本节各 CSV 表头中已经列出的字段。核心列必须存在，不能改名，不能删除，顺序固定。
- 扩展列：后续新增的特征字段。扩展列只能追加到对应 CSV 的最后，不能插入核心列中间。
- 日期窗口变化不改变 CSV 表头，只写入 manifest。
- 新特征组合变化不改变 `schema_version`，但必须更新 `feature_set_id` 和 `step2_feature_metadata.csv`。
- 字段重命名、字段删除、单位变化、百分比口径变化、唯一键变化，均视为 schema 破坏性变化，不能继续使用 `workflow_0.1_csv_v1`，必须升级为新的 schema，例如 `workflow_0.1_csv_v2`。

日期窗口变化时，必须在 manifest 中记录：

```csv
项目,说明
schema_version,workflow_0.1_csv_v1
date_start,YYYY-MM-DD
date_end,YYYY-MM-DD
latest_T,YYYY-MM-DD
raw_交易日数,N
data_window_note,本次窗口变化说明
```

Step-2 新增特征时，必须同时满足：

- 新特征追加到 `step2_feature_table_daily.csv` 的核心列之后。
- 每个新增特征或特征组必须写入 `step2_feature_metadata.csv`。
- `step2_data_manifest.csv` 必须记录 `feature_set_id`，例如 `feature_set_v1_momentum_volume_risk`。

例子：

- 如果只是把数据窗口从 `2026-01-05~2026-06-05` 扩展为更长区间，CSV 列不变，只更新 manifest。
- 如果新增 `rsi_14`、`macd_diff`，就在 `step2_feature_table_daily.csv` 末尾追加两列，并在 `step2_feature_metadata.csv` 记录来源、窗口、用途和防泄漏说明。
- 如果把 `ret_5` 从百分数数值改成小数比例，必须升级 schema，不能继续声称是 `workflow_0.1_csv_v1`。

### Step-1 输出文件

Step-1 的目标是形成数据资产，不做最终筛选，也不生成交易结论。

Step-1 尽量保持稳定。后续如果只改日期窗口、数据源或下载日期，优先更新 `step1_data_manifest.csv`，不优先改 Step-1 表头。

Step-1 标准输出目录：

```text
outputs/step1/
├── step1_stock_summary.csv
├── step1_daily_raw_data.csv
├── step1_sector_summary.csv
└── step1_data_manifest.csv
```

#### `step1_stock_summary.csv`

对应 Trial_2 工作簿中的 `Step1_一股一行`。

行粒度：每只沪深 300 股票一行。

唯一键：`股票代码`。

用途：记录每只股票在 Step-1 数据窗口内的最新状态、滚动汇总和行业/板块映射。

表头必须为：

```csv
股票代码,股票名称,成分股更新日期,raw_起始日期,raw_结束日期,raw_交易日数,最新日期,最新开盘,最新收盘,最新最高,最新最低,最新成交量,最新成交额,最新振幅,最新涨跌额,最新换手率,最新涨跌幅,近5日收益率,近5日成交量均值,近5日成交额均值,近5日换手率均值,近5日涨跌幅波动率,近20日收益率,近20日成交量均值,近20日成交额均值,近20日换手率均值,近20日涨跌幅波动率,近60日收益率,近60日成交量均值,近60日成交额均值,近60日换手率均值,近60日涨跌幅波动率,行业来源日期,原始行业,行业分类口径,板块划分
```

#### `step1_daily_raw_data.csv`

对应 Trial_2 工作簿中的 `Daily_Raw_Data`。

行粒度：每只股票、每个交易日一行。

唯一键：`股票代码 + 日期`。

用途：保留 Step-1 的原始日频行情，作为 Step-2 特征工程的直接输入。

表头必须为：

```csv
股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
```

#### `step1_sector_summary.csv`

对应 Trial_2 工作簿中的 `板块统计`。

行粒度：每个 `板块划分` 一行。

唯一键：`板块划分`。

用途：记录 Step-1 数据窗口末端的板块级汇总，方便快速检查板块样本数、成交额和近期收益。

表头必须为：

```csv
板块划分,股票数量,最新成交额合计,最新换手率均值,近5日收益率均值,近20日收益率均值,近60日收益率均值
```

#### `step1_data_manifest.csv`

对应 Trial_2 工作簿中的 `数据说明`。

行粒度：每个说明项一行。

唯一键：`项目`。

用途：记录数据来源、日期窗口、生成时间、输入文件、代码版本、注意事项等，使实验可追溯。

表头必须为：

```csv
项目,说明
```

至少必须记录以下项目：

```csv
项目,说明
schema_version,workflow_0.1_csv_v1
date_start,YYYY-MM-DD
date_end,YYYY-MM-DD
latest_T,YYYY-MM-DD
raw_交易日数,N
data_source,数据来源
data_window_note,窗口或数据源变化说明
```

### Step-2 输出文件

Step-2 的目标是把 Step-1 的 daily raw data 转成轻量特征、板块分数、风险标记和最新 T 日初筛表。

Step-2 第一版采用 `5 个核心输出 + 2 个派生视图`。

核心输出是后续流程和验收依赖的标准接口：

```text
outputs/step2/
├── step2_feature_table_daily.csv
├── step2_sector_feature_table.csv
├── step2_latest_t_screen.csv
├── step2_feature_metadata.csv
└── step2_data_manifest.csv
```

派生视图用于人工复盘和快速检查；它们可以从核心输出重新生成，不作为唯一真相来源：

```text
outputs/step2/
├── step2_sector_score_latest.csv
└── step2_risk_feature_table.csv
```

如果核心输出与派生视图出现冲突，以核心输出为准。

#### `step2_feature_table_daily.csv`

对应 Trial_2 工作簿中的 `feature_table_daily`。

行粒度：每只股票、每个交易日一行。

唯一键：`股票代码 + 日期`。

用途：Step-2 的主特征宽表，后续模型训练、候选池生成和精排都应优先从这张表取数。

以下表头是 `workflow_0.1_csv_v1` 的核心列，必须按顺序保留：

```csv
股票代码,日期,股票名称,原始行业,行业分类口径,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅,ret_1,ret_3,ret_5,ret_10,ret_20,ma5,ma10,ma20,ma5_over_ma20,trend_slope_5,volatility_5,volatility_10,volatility_20,amount_ma3,amount_ma5,amount_ma20,volume_ma5,volume_ma20,amount_ratio_5_20,volume_ratio_5_20,amount_rank_5,market_ret_5,sector_ret_5,sector_excess_ret_5,stock_vs_market_ret_5,stock_vs_sector_ret_5,rank_in_sector_ret_5,market_index,market_amount,sector_amount_ratio_5_20,sector_short_score,max_drawdown_20,extreme_drop_20_flag,low_liquidity_flag,no_trade_or_abnormal_flag,risk_any_flag,板块划分
```

新增特征必须追加在这些核心列之后，并同步登记到 `step2_feature_metadata.csv`。

#### `step2_sector_feature_table.csv`

对应 Trial_2 工作簿中的 `sector_feature_table`。

行粒度：每个交易日、每个板块一行。

唯一键：`日期 + 板块划分`。

用途：记录板块级别的收益、成交额、扩散度和短线打分明细。

表头必须为：

```csv
日期,板块划分,sector_daily_ret,sector_amount,sector_volume,sector_stock_count,new_high_20_ratio,ret_5_gt_5pct_ratio,sector_index,sector_ret_3,sector_ret_5,sector_ret_10,sector_ma5,sector_ma10,sector_amount_ma5,sector_amount_ma20,sector_amount_ratio_5_20,sector_amount_rank_5,market_ret_5,sector_excess_ret_5,score_sector_ret_5,score_sector_excess_ret_5,score_ma_bull,score_amount_ratio,score_amount_rank_5,score_new_high_20_ratio,score_ret_5_gt_5pct_ratio,score_catalyst,sector_short_score
```

#### `step2_sector_score_latest.csv`

对应 Trial_2 工作簿中的 `sector_score_latest`。

行粒度：最新预测日 T 的每个板块一行。

唯一键：`日期 + 板块划分`。

用途：固定最新 T 日的板块排序结果，供人工复盘和 Step-3 以后流程引用。

表头必须与 `step2_sector_feature_table.csv` 完全一致。

定位：派生视图。它等价于从 `step2_sector_feature_table.csv` 中筛选 `latest_T` 当天记录，主要方便快速查看最新板块排序。

#### `step2_latest_t_screen.csv`

对应 Trial_2 工作簿中的 `latest_T_初筛`。

行粒度：最新预测日 T 的每只股票一行。

唯一键：`股票代码 + 日期`。

用途：记录最新 T 日个股初筛结果。这里仍然不是最终提交名单，只是后续模型、候选池或精排的输入。

表头必须为：

```csv
股票代码,股票名称,日期,板块划分,原始行业,ret_5,trend_slope_5,stock_vs_sector_ret_5,rank_in_sector_ret_5,amount_ratio_5_20,volume_ratio_5_20,amount_rank_5,max_drawdown_20,extreme_drop_20_flag,low_liquidity_flag,no_trade_or_abnormal_flag,risk_any_flag,risk_pass_flag,score_sector_ret_5,score_sector_excess_ret_5,score_ma_bull,score_amount_ratio,score_amount_rank_5,score_new_high_20_ratio,score_ret_5_gt_5pct_ratio,score_catalyst,sector_short_score,stock_trend_score,volume_confirm_score,进入后续流程标记
```

#### `step2_risk_feature_table.csv`

对应 Trial_2 工作簿中的 `risk_feature_table`。

行粒度：每只股票、每个交易日一行。

唯一键：`股票代码 + 日期`。

用途：单独记录风险过滤字段，方便检查 Step-2 初筛剔除原因。

表头必须为：

```csv
股票代码,日期,股票名称,max_drawdown_20,extreme_drop_20_flag,low_liquidity_flag,no_trade_or_abnormal_flag,risk_any_flag,板块划分
```

定位：派生视图。它等价于从 `step2_feature_table_daily.csv` 中抽取风险相关字段，主要方便复盘风险过滤原因。

#### `step2_feature_metadata.csv`

对应 Trial_2 工作簿中的 `feature_metadata`。

行粒度：每个特征或特征组一行。

唯一键：`特征名`。

用途：记录特征来源、窗口、是否用于模型、是否用于精排、防泄漏说明。新增特征时必须同步更新这张表。

表头必须为：

```csv
特征名,特征来源,计算窗口,是否用于模型,是否用于精排,防泄漏说明
```

新增特征登记规则：

- `特征名` 必须与 `step2_feature_table_daily.csv` 中的新增列名完全一致。
- `计算窗口` 必须写明使用的历史窗口，例如 `14日`、`12/26/9日`、`5日与20日`。
- `是否用于模型` 和 `是否用于精排` 只能填写 `是` 或 `否`。
- `防泄漏说明` 必须说明该特征只使用预测日 T 及以前可得数据。

#### `step2_data_manifest.csv`

对应 Trial_2 工作簿中的 `数据说明`。

行粒度：每个说明项一行。

唯一键：`项目`。

用途：记录 Step-2 输入来源、输入行数、日期覆盖、生成时间、代码版本和注意事项。

表头必须为：

```csv
项目,说明
```

至少必须记录以下项目：

```csv
项目,说明
schema_version,workflow_0.1_csv_v1
feature_set_id,feature_set_v1_momentum_volume_risk
date_start,YYYY-MM-DD
date_end,YYYY-MM-DD
latest_T,YYYY-MM-DD
raw_交易日数,N
input_step1_path,outputs/step1/step1_daily_raw_data.csv
data_window_note,窗口或输入变化说明
feature_set_note,本次新增、删除或暂停使用的特征说明
```

### Step-1 / Step-2 边界

- Step-1 只负责数据资产：raw daily、股票汇总、板块汇总、数据说明。
- Step-2 只负责特征工程与初筛标记：主特征表、板块特征表、最新 T 日初筛、特征元数据、数据说明，以及用于复盘的最新板块分数和风险视图。
- Step-2 的核心输出是 `step2_feature_table_daily.csv`、`step2_sector_feature_table.csv`、`step2_latest_t_screen.csv`、`step2_feature_metadata.csv`、`step2_data_manifest.csv`。
- Step-2 的派生视图是 `step2_sector_score_latest.csv` 和 `step2_risk_feature_table.csv`；它们方便人工检查，但不作为唯一真相来源。
- Step-2 不输出 `result.csv`，也不决定最终权重。
- Step-3 以后如果需要使用 Step-1 / Step-2 的结果，只能读取本节规定的 CSV 文件，不直接依赖 Excel sheet 名或临时中间表。
- 不同实验之间横向比较 Step-2 特征、初筛结果或后续模型表现前，必须先检查 manifest 中的 `schema_version` 和 `feature_set_id`。只有两者兼容时，才能直接比较。
