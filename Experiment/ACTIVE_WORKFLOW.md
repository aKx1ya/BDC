# Active Workflow

本文件是当前实验调度入口。以后切换策略版本时，优先改这里，而不是改底层抓数脚本。

## 当前激活版本

```text
active_workflow: workflow_0.1
active_stage: Step-7
status: step7-freeze-only-success-local-score-blocked
```

## 当前策略文件

- `Experiment/workflow_0.1/strategy/0.1_Step-1_数据获取流程与思考逻辑.md`
- `Experiment/workflow_0.1/strategy/0.1_Step-1_真实调度图_代码视角.md`
- `Experiment/workflow_0.1/strategy/0.1_Step-2_特征工程与初步筛选流程与思考逻辑.md`
- `Experiment/策略流程与实验方案.md`
- `Experiment/workflow_0.1/README.md`

## 当前允许调用的执行层

当前已建立跨 workflow 的迁移版 shared 入口。它会读取本文件，再读取当前 workflow 的 `workflow_config.yaml`，最后分发到对应 workflow 的本地 `run_stepN.py`：

```bash
/opt/miniconda3/bin/python3 Experiment/shared/validators/validate_workflow_config.py --workflow workflow_0.1
/opt/miniconda3/bin/python3 Experiment/shared/runners/run_step.py --step 7 --mode freeze-only
```

说明：

- `Experiment/shared/` 是可迁移健康体系的第一阶段入口。
- `Experiment/workflow_0.1/run_stepN.py` 仍然是当前 workflow 的本地正式健康调度器。
- shared runner 会要求 `active_stage` 和 `--step` 一致；跨阶段调试才使用 `--allow-stage-mismatch`。

Step-1 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step1.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 联网执行 bigdata_challenge/data_fetcher/run_all.py --step 1
-> 校验 raw 数据
-> 生成 workflow_0.1 标准 Step-1 CSV
-> 校验输出
-> 写入 notes/step1_run_report.md
```

底层执行层仍然只把 `bigdata_challenge` 当作原始数据抓取层使用。正式 Step-1 调度器内部允许调用：

```bash
cd bigdata_challenge/data_fetcher
python3 run_all.py --step 1
```

说明：

- `--step 1` 用于更新沪深 300 成分股和日频行情。
- `--step 5` 当前不属于正式健康链路；workflow_0.1 使用 `stock_industry.csv` 自聚合六大风格板块。
- 不默认执行 `--with-feature`。
- 不默认执行建模、筛选、打分或交易结论。

Step-2 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step2.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-1 实验
-> 校验 Step-1 输入健康
-> 生成 workflow_0.1 标准 Step-2 CSV
-> 校验 Step-2 输出
-> 写入 notes/step2_run_report.md
```

Step-2 不联网抓取 raw 数据，不重新定义沪深300股票池，不生成最终 `result.csv`。

Step-3 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step3.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-2 实验
-> 校验 Step-2 输入健康
-> 生成 workflow_0.1 标准 Step-3 样本 CSV
-> 校验 Step-3 输出
-> 写入 notes/step3_run_report.md
```

Step-3 不联网抓取 raw 数据，不重新计算 Step-2 特征，不训练模型，不生成 `candidate_top30.csv` 或 `result.csv`。

Step-4 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step4.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-3 实验
-> 校验 Step-3 输入健康
-> 生成 workflow_0.1 标准 Step-4 切分与 walk-forward CSV
-> 校验 Step-4 输出
-> 写入 notes/step4_run_report.md
```

Step-4 不联网抓取 raw 数据，不重新计算 Step-2 特征，不重新构造 Step-3 标签，不训练模型，不生成 `candidate_top30.csv` 或 `result.csv`。

Step-5 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step5.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-4 实验
-> 通过 Step-4 / Step-3 manifest 推断同链路 Step-3 / Step-2 实验
-> 校验 Step-2 / Step-3 / Step-4 输入健康且一致
-> 训练 baseline 模型并生成 workflow_0.1 标准 Step-5 CSV
-> 校验 Step-5 输出
-> 写入 notes/step5_run_report.md
```

Step-5 会训练模型并生成 `step5_candidate_top30.csv`，但不生成最终 `result.csv`。

Step-6 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step6.py
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-5 实验
-> 通过 Step-5 manifest 推断同链路 Step-2 实验
-> 校验 Step-5 / Step-2 输入健康且日期一致
-> 只在 Step-5 Top30 内执行精排、组合约束和权重分配
-> 生成 outputs/step6/ 的 result.csv 与审计 CSV
-> 校验 Step-6 输出
-> 写入 notes/step6_run_report.md
```

Step-6 会生成最终 `step6_result.csv`，但不训练模型、不重新召回股票、不执行 Step-7 评分。

Step-7 当前正式入口是 workflow 自己的健康调度器：

```bash
/opt/miniconda3/bin/python3 Experiment/workflow_0.1/run_step7.py --mode freeze-only
```

它会自动完成：

```text
读取 ACTIVE_WORKFLOW
-> 自动寻找最近一个 SUCCESS 的 Step-6 实验
-> 校验 Step-6 输入健康
-> 冻结 step6_result.csv 为 step7_frozen_result.csv
-> 按 mode 决定是否读取 test.csv 并执行官方口径评分
-> 生成 outputs/step7/ 的冻结、评分、manifest 和 leakage_check CSV
-> 校验 Step-7 输出
-> 写入 notes/step7_run_report.md
```

当前健康完成的是 `freeze-only` 模式。`local-score` 已尝试，但被健康检查阻止：当前 `THU-BDC2026-main/data/test.csv` 的测试窗口早于 Step-6 的 `candidate_date=2026-06-15`，不能作为本轮健康评分使用。

## 当前标准输出位置

每次正式实验应在 `Experiment/workflow_0.1/experiments/` 下新建实验目录：

```text
Experiment/workflow_0.1/experiments/exp_YYYYMMDD_step7_主题/
├── inputs/
├── outputs/
│   └── step7/
│       ├── step7_frozen_result.csv
│       ├── step7_score_summary.csv
│       ├── step7_stock_contribution.csv
│       ├── step7_score_manifest.csv
│       └── step7_leakage_check.csv
└── notes/
```

## 调度原则

1. 先读本文件，确认当前激活 workflow。
2. 再读对应 workflow 的 `strategy/` 和 `README.md`。
3. 当前 `workflow_0.1` 的正式本地入口仍是 `Experiment/workflow_0.1/run_stepN.py`；跨 workflow 迁移入口是 `Experiment/shared/runners/run_step.py --step N`。
4. 只把 `bigdata_challenge/data_fetcher/` 当作底层执行器。
5. 原始抓取结果留在 `bigdata_challenge/data/raw/`。
6. workflow 标准产物写入当前实验目录的 `outputs/`。
7. 每次正式 Step-1 必须生成 `notes/step1_run_report.md`；每次正式 Step-2 必须生成 `notes/step2_run_report.md`；每次正式 Step-3 必须生成 `notes/step3_run_report.md`；每次正式 Step-4 必须生成 `notes/step4_run_report.md`；每次正式 Step-5 必须生成 `notes/step5_run_report.md`；每次正式 Step-6 必须生成 `notes/step6_run_report.md`；每次正式 Step-7 必须生成 `notes/step7_run_report.md`。
