# shared/validators

这里放跨 workflow 复用的健康校验逻辑。

当前已落地：

```text
validate_workflow_config.py
```

它负责检查一个 workflow 是否能接入 shared runner。它不替代每一步自己的业务校验，例如：

```text
workflow_0.1/pipelines/validate_step1.py
workflow_0.1/pipelines/validate_step2.py
...
workflow_0.1/pipelines/validate_step7.py
```

分工可以这样理解：

```text
validate_workflow_config.py
  校验“这版 workflow 能不能被调度”

validate_stepN.py
  校验“这一步产物健不健康”
```
