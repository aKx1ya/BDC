# shared/reports

这里预留给跨 workflow 复用的运行报告生成逻辑。

现在 `workflow_0.1/run_stepN.py` 已经会各自写：

```text
notes/stepN_run_report.md
```

未来可以把重复的报告结构抽成 shared helper，例如：

```text
Status
Active Workflow
Paths
Params
Input Metrics
Output Metrics
Error
```

这样新建 `workflow_0.2` 时，只需要提供每一步自己的 metrics，不需要重新手写报告模板。
