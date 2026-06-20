# Experiment/shared

`shared/` 是跨 workflow 的健康体系底座。它的目标不是替代某一版策略，而是让 `workflow_0.1`、未来 `workflow_0.2`、`workflow_0.3` 都能沿用同一套调度、配置、校验和报告习惯。

当前这一版是迁移第一阶段：

```text
shared 负责读 ACTIVE_WORKFLOW + workflow_config.yaml
shared 负责校验配置和分发命令
workflow_0.1 仍然保留自己的 run_step1.py ~ run_step7.py
```

## 核心文件

- `workflow_context.py`：读取 `Experiment/ACTIVE_WORKFLOW.md`，定位当前 workflow，并加载 `workflow_config.yaml`。
- `runners/run_step.py`：通用入口，按配置分发到当前 workflow 的 `run_stepN.py`。
- `validators/validate_workflow_config.py`：检查 config 是否能安全接入 shared runner。
- `schemas/workflow_config_schema_v1.md`：说明 `workflow_config.yaml` 应该怎么写。
- `tests/`：证明 shared 层对临时 workflow 也能工作，不只绑定 `workflow_0.1`。

## 推荐用法

先验证当前 workflow 配置：

```bash
/opt/miniconda3/bin/python3 Experiment/shared/validators/validate_workflow_config.py --workflow workflow_0.1
```

再通过 shared runner 调度当前 active stage：

```bash
/opt/miniconda3/bin/python3 Experiment/shared/runners/run_step.py --step 7 --mode freeze-only
```

如果只是查看会调用什么，不实际运行：

```bash
/opt/miniconda3/bin/python3 Experiment/shared/runners/run_step.py --step 7 --mode freeze-only --dry-run --print-context
```

## 和 workflow_x.x 的分工

```text
Experiment/shared/
  放通用能力：上下文读取、配置校验、统一调度、通用 schema、报告规范。

Experiment/workflow_x.x/
  放策略差异：strategy/、workflow_config.yaml、docs/、experiments/，以及尚未抽象到 shared 的本地 runner。
```

第一阶段先把“入口和规则”抽出来。等未来 `workflow_0.2` 跑起来后，再把重复度最高的 build / validate / report 逻辑逐步迁入 shared。
