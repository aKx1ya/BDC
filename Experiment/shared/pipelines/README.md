# shared/pipelines

这里预留给未来跨 workflow 复用的 build 逻辑。

当前阶段先不急着把 `workflow_0.1/pipelines/build_step*.py` 全部搬进来，因为那些脚本里还有不少 `workflow_0.1` 的策略细节。更稳妥的迁移顺序是：

```text
1. 先用 shared/runners/run_step.py 统一入口
2. 再用 workflow_config.yaml 把参数变成可配置
3. 最后把重复度最高、策略无关的 build helper 抽到 shared/pipelines/
```

判断一个函数是否适合迁移到这里，可以看它是否满足：

```text
不写死 workflow_0.1
不写死某一次 experiment 名
不依赖某个策略版本独有字段
可以通过 config 参数控制行为
```
