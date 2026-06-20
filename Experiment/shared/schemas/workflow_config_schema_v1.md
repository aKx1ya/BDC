# workflow_config.yaml Schema v1

`workflow_config.yaml` 是每个 workflow 的机器可读调度地图。策略文档给人看，config 给 shared runner 和 validator 看。

## 最小结构

```yaml
workflow_id: workflow_0.1
schema_version: workflow_0.1_csv_v1
health_system_version: workflow_health_v1

paths:
  strategy_dir: strategy
  docs_dir: docs
  pipelines_dir: pipelines
  experiments_dir: experiments

strategy_sources:
  - strategy/0.1_Step-1_数据获取流程与思考逻辑.md
  - ../策略流程与实验方案.md

steps:
  step1:
    stage: Step-1
    runner: run_step1.py
    default_args: []
    output_dir_name: step1
    report_name: step1_run_report.md
    policy_source:
      - strategy/0.1_Step-1_数据获取流程与思考逻辑.md
    health_doc: docs/Step-1_正式健康版运作流程.md
```

## 字段说明

| 字段 | 必填 | 说明 |
|---|---:|---|
| `workflow_id` | 是 | 必须等于目录名，例如 `workflow_0.1`。 |
| `schema_version` | 是 | 当前 workflow 输出 CSV 的 schema 版本。 |
| `health_system_version` | 建议 | 健康体系版本，例如 `workflow_health_v1`。 |
| `inherit_from` | 可选 | 新 workflow 继承哪个旧 workflow。 |
| `paths` | 建议 | 声明策略、文档、pipeline、实验产物目录。 |
| `strategy_sources` | 建议 | 本 workflow 的主要策略来源。 |
| `steps.stepN.stage` | 建议 | 应写成 `Step-N`，用于人机对齐。 |
| `steps.stepN.runner` | 是 | shared runner 最终调用的本地 runner。 |
| `steps.stepN.default_args` | 建议 | 默认透传给本地 runner 的参数列表。 |
| `steps.stepN.output_dir_name` | 建议 | 标准输出目录名，应保持 `stepN`。 |
| `steps.stepN.report_name` | 建议 | 标准运行报告名，应保持 `stepN_run_report.md`。 |
| `steps.stepN.policy_source` | 建议 | 这个 Step 使用的策略文档。 |
| `steps.stepN.health_doc` | 建议 | 这个 Step 的健康体系说明文档。 |

## 路径规则

相对路径默认相对于当前 workflow 目录：

```text
Experiment/workflow_0.1/
```

也可以使用模板：

```text
{project_root}
{experiment_root}
{workflow_dir}
```

## 健康约束

shared validator 会检查：

```text
workflow_id 与目录/active context 一致
schema_version 存在
steps 是 mapping
runner 文件存在
default_args 是 list
output_dir_name 与 stepN 一致
report_name 与 stepN_run_report.md 一致
policy_source 和 health_doc 指向真实文件
paths 中声明的目录真实存在
```

这份 schema 的核心作用是让新 workflow 可以迁移：

```text
复制 workflow_config.yaml
改 workflow_id
改策略来源
改必要参数
再用 shared validator 检查
```
