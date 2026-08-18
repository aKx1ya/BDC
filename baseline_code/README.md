# THU-BigDataCompetition-2026-baseline

本项目是面向 **THU-BDC2026 C 赛题（量化投资中的机器学习）** 的排序学习选股方案：

- **任务**：从沪深 300 成分股中选出不超过 5 只股票并分配权重（权重和 ≤ 1），预测未来 5 个交易日 open-to-open 加权收益率；
- **输入**：每只股票过去 60 个交易日的量价与技术特征序列；
- **模型**：`StockTransformer` + `XGBRanker` + `LGBMRanker` 多模型融合；
- **流程**：两阶段推理 —— Stage 1 多模型融合召回 Top30 候选池，Stage 2 精排（硬门控 / 行业约束 / 相关性约束 / 均值-方差权重优化）产出 Top5；
- **输出**：`output/result.csv`（`stock_id,weight`）。

---

## 1. 整体流程

```text
Stage 1 召回（Top30）                         Stage 2 精排（Top5）
─────────────────────                        ─────────────────────
StockTransformer 分数                        Hard Gates 硬门控
XGBRanker 分数         日截面排名百分位融合    行业集中度约束
LGBMRanker 分数   ────► ensemble_score ────► 相关性约束
                              │              均值-方差权重优化
                              v                    │
                    candidate_top30.csv             v
                                             output/result.csv
```

训练主流程：

1. 读取历史行情数据（`data/train.csv`）；
2. 特征工程（`39` 或 `158+39` 特征，可开启额外因子）；
3. 构造标签（`label.py` 支持绝对收益 / 超额收益 / 排序 / 方向 / 波动率标签）；
4. 按日期组织排序样本（同一天约 300 只股票为一个排序题）；
5. 训练排序模型，监控验证集 `final_score` 并保存最优权重；
6. `predict.py` 两阶段推理，生成 `output/result.csv`。

---

## 2. 代码结构说明

### [config.py](code/src/config.py)
统一管理训练与推理参数，全部支持环境变量覆盖（`BDC_` 前缀），关键项：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `sequence_length` | 60 | 历史输入窗口（交易日） |
| `feature_num` | `158+39` | 特征集：`39` 或 `158+39` |
| `label_type` | `excess_return` | `absolute_return` / `excess_return` / `rank` / `direction` |
| `label_buy_offset` / `label_sell_offset` | 1 / 6 | T+1 开盘买入、T+6 开盘卖出（防泄漏口径） |
| `learning_rate` | `2e-5` | warm-start 微调步长 |
| `num_epochs` / `early_stopping_patience` | 15 / 3 | 训练轮数与早停 |
| `pairwise_weight` | 0.0 | 配对损失权重 |
| `enable_xgb_ranker` / `enable_lgb_ranker` | 1 / 1 | 是否启用树模型排序器 |
| `xgb/lgb/transformer_ensemble_weight` | 0.40 / 0.35 / 0.25 | 融合权重 |
| `enable_extra_factors` | 0 | 额外时序因子开关 |
| `portfolio_weighting` | `rank_softmax` | `equal` / `softmax` / `rank_decay` / `rank_softmax` |
| `candidate_pool_size` | 30 | 召回候选池大小 |
| `max_per_sector` / `max_correlation` | 2 / 0.3 | 精排约束 |

### [model.py](code/src/model.py)
核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- `CrossStockAttention`：在同一交易日内建模股票间关系；
- `ranking_layers` + `score_head`：输出每只股票的排序分数。

输入形状：`[batch, num_stocks, seq_len, feature_dim]`  
输出形状：`[batch, num_stocks]`。

### [loss.py](code/src/loss.py)
`WeightedRankingLoss`：鲁棒 listwise 排序损失（median/MAD 归一化 + Huber 截尾），对真实 Top-k 样本施加更高权重，可叠加小型 pairwise 稳定项；另含多任务联合损失（方向 BCE、波动率 Huber）。

### [label.py](code/src/label.py)
标签构造模块：绝对收益、**超额收益**（默认，个股收益 − 当日等权平均）、排序百分位、方向、波动率五种标签，以及方向 / 波动率辅助任务标签。

### [features.py](code/src/features.py)
增强特征工程：申万一级行业映射（内嵌 300 只成分股）、行业动能 / 超额收益 / 成交额变化、市场状态（涨跌比、市场波动率、20 日均线上方判定）、统一入口 `engineer_all_features()`。

### [models.py](code/src/models.py)
树模型排序框架：
- `XGBRankerWrapper`（`rank:pairwise`，支持连续标签）；
- `LGBRankerWrapper`（`lambdarank`，自动将连续标签转组内整数排名）；
- `ModelEnsemble`：日截面排名百分位加权融合；
- `train_tree_models()`：一键训练两棵树模型；
- `sequences_to_tabular()`：3D 序列 → 2D 表格（`last_day` / `stats` / `flatten`）。

### [postprocess.py](code/src/postprocess.py)
精排模块：硬门控（涨跌停、回撤、暴跌）、行业集中度约束、相关性约束、均值-方差权重优化（SLSQP），以及 `fine_ranking()` 主流程（Top30 → Top5）。

### [validation.py](code/src/validation.py)
`WalkForwardValidator`：walk-forward 滚动验证（训练窗口 / 净化带 / 滚动步长可配），并提供与比赛口径一致的 `compute_final_score()`。

### [ensemble.py](code/src/ensemble.py)
多 checkpoint 排名融合脚本：按各模型 `final_score` 加权平均排名，生成融合后的 `result.csv`。

### [nightly_automl.py](code/src/nightly_automl.py)
夜间 AutoML 调度器：循环执行 A/B/C/D 策略（额外因子 / 因果掩码残差 / 鲁棒 listwise+softmax / 余弦退火正则化），带本地 hurdle 检查、成功标记提交，7 小时或 3 次成功后触发集成。

### [utils.py](code/src/utils.py)
特征工程与数据集构建：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `add_extra_factor_features()` / `add_cross_sectional_features()`：额外时序因子与横截面特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [train.py](code/src/train.py)
训练主脚本，关键内容：
- 数据预处理：`_preprocess_common()` 多进程特征工程 + 标签构建（`label.py`）；
- 数据集组织：`RankingDataset` + `collate_fn`（padding + mask 处理每日股票数不一致）；
- 训练支持 warm-start 预训练权重、AdamW + CosineAnnealing、AMP、OOM 重试、早停；
- 以验证集 `final_score` 选择最优模型。

训练产物：
- `best_model.pth`：最佳模型参数；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：最佳分数记录；
- `log/`：TensorBoard日志。

### [predict.py](code/src/predict.py)
两阶段推理主脚本：
1. 加载 `best_model.pth` + `scaler.pkl`，对最新日期全部股票打分；
2. 若存在 `xgboost_ranker.json` / `lightgbm_ranker.txt` 则一起融合，输出 `output/candidate_top30.csv`；
3. `postprocess.fine_ranking()` 精排，输出 `output/result.csv`。

### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（Baostock）：
- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

输出文件：
- `output/candidate_top30.csv`：Stage 1 召回候选池；
- `output/result.csv`：最终提交（最多 5 只，权重和 ≤ 1）。

防未来信息泄漏底线：
- 标签为 T+1 开盘买、T+6 开盘卖，特征只用当前及过去行；
- 验证 / 测试集标签只用于评分，不参与训练调参；
- 横截面特征只使用同一天已知股票。

---

## 4. 运行方法（推荐使用 uv）

1) 使用 `uv` 安装依赖

`uv sync`

2) 激活虚拟环境

Linux/macOS：`source .venv/bin/activate`  
Windows：`.\.venv\Scripts\activate`

3) 训练模型

```
sh train.sh
```

Windows 可直接运行 `python code/src/train.py`。

4) 生成预测结果

```
sh test.sh
```

Windows 可直接运行 `python code/src/predict.py`。

5) 本地自评（对照 `data/test.csv`）

```
python test/score_self.py
```

---

## 5. 常见问题

1) `TA-Lib` 安装失败  
本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装Python包。
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

2) 多进程相关问题  
`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS下请保持通过脚本入口运行（不要在交互式环境里直接多进程调用主逻辑）。Windows 下 `num_workers` 自动降为 0（除非设置 `BDC_ALLOW_WINDOWS_WORKERS=1`）。

3) GPU/CPU自动选择  
代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无GPU时可直接CPU运行。4GB GPU 默认小 batch + 梯度累积。

4) 实验复现  
所有可调参数均可通过 `BDC_*` 环境变量覆盖（见 `config.py`），`OPTIMIZE_LOG.md` 记录了历次 AutoML 实验与最佳分数。
