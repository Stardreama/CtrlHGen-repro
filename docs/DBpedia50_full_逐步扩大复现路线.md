# DBpedia50 full 自洽路线逐步扩大复现方案

本文档用于后续按阶段扩大 CtrlHGen 复现实验规模。路线选择官方仓库中最自洽的一条线：

- 采样入口：`scripts/sample/sample_full.sh`
- 采样配置：`akgr/configs/config-sampling.yml` 中 `full: datasets: ["DBpedia50"]`
- 训练目标：`DBpedia50 --scale full`

注意：官方 `scripts/train/db-g2.sh` 默认带 `-r=380`，表示作者本地已有 checkpoint 后续训；从零复现时不要直接使用该参数，需要先训练自己的 unconditional checkpoint，再做 pattern 条件训练。

## 1. 规模估算

当前代码采样逻辑：

```python
train_per_pattern = num_train_edges // scale
valid_per_pattern = train_per_pattern // 8
test_per_pattern = train_per_pattern // 8
```

DBpedia50 在当前代码重新划分并添加 inverse edges 后，`num_train_edges` 约为 `55074`。当前使用 13 个 pattern。

| 阶段 | scale | 每 pattern train | train 总量 | valid 总量 | test 总量 | 目标 |
|---|---:|---:|---:|---:|---:|---|
| S0 | 5000 | 11 | 143 | 13 | 13 | DBpedia50 smoke test |
| S1 | 1000 | 55 | 715 | 78 | 78 | 小规模流程验证 |
| S2 | 500 | 110 | 1430 | 169 | 169 | 小规模可报告实验 |
| S3 | 100 | 550 | 7150 | 884 | 884 | 中等规模 supervised |
| S4 | 50 | 1101 | 14313 | 1781 | 1781 | 较稳定 supervised |
| S5 | 10 | 5507 | 71591 | 8944 | 8944 | 大规模本地压力测试 |
| S6 | 1 | 55074 | 715962 | 89492 | 89492 | 官方 full 规模 |

这些是估算值，实际行数以 `wc -l sampled_data/DBpedia50/*jsonl` 为准。

当前进度：

- S0 `db50_s5000` 已完成：实际 train/valid/test 为 143 / 13 / 13。
- S0 已产出 checkpoint：`DBpedia50-db50_s5000-32-20-unconditional.pth`、`DBpedia50-db50_s5000-32-40-pattern.pth`。
- S0 baseline 指标：Smatch 0.0110，Jaccard/Dice/Overlap 0。
- S0 rerank 指标：Smatch 0.0944，Validity/Entity-number/Relation-number 0.0769，Jaccard/Dice/Overlap 0。
- S1 `db50_s1000` 已完成：实际 train/valid/test 为 715 / 78 / 78。
- S1 已产出 checkpoint：`DBpedia50-db50_s1000-32-20-unconditional.pth`、`DBpedia50-db50_s1000-32-40-pattern.pth`。
- S1 baseline 指标：Smatch 0.0154，Entity-number 0.0385，Jaccard/Dice/Overlap 0。
- S1 rerank 指标：Smatch 0.0392，Jaccard 0.0001，Dice 0.0001，Overlap 0.0009，Validity 0.0385，Entity-number/Relation-number 0.0513。
- S1 full rerank 耗时约 9 小时 24 分钟，是当前路线的主要时间瓶颈；S2 之后不建议直接 full rerank。
- 详细命令和结果记录见 `CtrlHGen_debug_run_log.md`。

## 2. 先添加阶段化采样配置

在 `CtrlHGen/akgr/configs/config-sampling.yml` 中加入：

```yaml
db50_s5000:
  datasets: ["DBpedia50"]
  scale: 5000
db50_s1000:
  datasets: ["DBpedia50"]
  scale: 1000
db50_s500:
  datasets: ["DBpedia50"]
  scale: 500
db50_s100:
  datasets: ["DBpedia50"]
  scale: 100
db50_s50:
  datasets: ["DBpedia50"]
  scale: 50
db50_s10:
  datasets: ["DBpedia50"]
  scale: 10
```

`full` 已经存在：

```yaml
full:
  datasets: ["DBpedia50"]
  scale: 1
```

## 3. 每个阶段的统一流程

以下命令用变量写法，执行时只需要改 `SCALE`、训练 epoch 和 checkpoint epoch。

进入目录：

```bash
conda activate ctrlhgen
cd ~/project/南开考核/CtrlHGen
```

### 3.1 采样

```bash
SCALE=db50_s1000
python -m akgr.sampling.sample_parallel -s ${SCALE} -a 32 -p 8
```

检查：

```bash
wc -l sampled_data/DBpedia50/DBpedia50-${SCALE}-32-*-a2q.jsonl
cat sampled_data/DBpedia50/stats.txt
```

### 3.2 无条件 supervised training

小规模建议先用 batch size 16 或 32，避免 RTX 4060 8GB 爆显存。

```bash
SCALE=db50_s1000
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d DBpedia50 --scale ${SCALE} -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training --condition unconditional \
  --overwrite_batchsize 16 --save_frequency 5 --override_nepoch 50
```

检查 checkpoint：

```bash
ls -lh checkpoints/GPT2_6_act_nt/DBpedia50-${SCALE}-32-*-unconditional.pth
tail -n 20 results/GPT2_6_act_nt/DBpedia50-${SCALE}-32_results.txt
```

### 3.3 pattern 条件 supervised training

从无条件 checkpoint 继续训练。假设无条件训练到 epoch 50：

```bash
SCALE=db50_s1000
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern \
  --data_root ./sampled_data/ -d DBpedia50 --scale ${SCALE} -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training -r 50 \
  --overwrite_batchsize 16 --save_frequency 5 --override_nepoch 100
```

这里 `-r 50` 表示加载：

```text
checkpoints/GPT2_6_act_nt/DBpedia50-${SCALE}-32-50-unconditional.pth
```

注意：第一次从 unconditional checkpoint 开始做 pattern 条件训练时不要加 `--tuning`。如果后续要从已有 pattern checkpoint 继续训练，例如从 `DBpedia50-${SCALE}-32-100-pattern.pth` 继续，则需要加 `--tuning`。

并训练到 epoch 100，保存：

```text
checkpoints/GPT2_6_act_nt/DBpedia50-${SCALE}-32-100-pattern.pth
```

### 3.4 测试 baseline

```bash
SCALE=db50_s1000
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale ${SCALE} -a 32 \
  --checkpoint_root checkpoints/ -r 100 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True --rerank_k 1
```

### 3.5 测试 reranking 改进

```bash
SCALE=db50_s1000
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale ${SCALE} -a 32 \
  --checkpoint_root checkpoints/ -r 100 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True \
  --rerank_k 4 --rerank_alpha 0.5 --rerank_log_candidates
```

### 3.6 汇总指标

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

scale = "db50_s1000"
epoch = 100
paths = [
    (f"baseline_k1", Path(f"results/GPT2_6_act_nt/DBpedia50-{scale}-32-{epoch}-scores(test|1.0xtest_topk0_True_False).csv")),
    (f"rerank_k4", Path(f"results/GPT2_6_act_nt/DBpedia50-{scale}-32-{epoch}-scores(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).csv")),
]
metrics = ["smatch", "jaccard", "dice", "overlap", "validity", "enumber", "pnumber"]
for name, path in paths:
    if not path.exists():
        print("missing", name, path)
        continue
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    print(name, {m: float(df.loc["all", (m, "mean")]) for m in metrics})
PY
```

## 4. 推荐推进顺序

### S0：`db50_s5000`

目标：确认 DBpedia50 线路无采样/训练/测试路径问题。

建议：

- sampling：`db50_s5000`
- unconditional：10 到 20 epoch
- pattern：再到 30 或 50 epoch
- test：全量 test
- 若全部指标都是 0，也可以继续；这个阶段主要看流程。

进入下一阶段标准：

- 能生成 train/valid/test jsonl；
- 能保存 unconditional 和 pattern checkpoint；
- 能生成 score CSV；
- 不出现路径、OOM、pickle、tokenizer 返回值等错误。

### S1：`db50_s1000`

目标：小规模但不再是纯 smoke test。

建议：

- unconditional：50 epoch
- pattern：训练到 100 epoch
- baseline vs rerank：`rerank_k=1` vs `rerank_k=4`

实际已完成的 S1 先采用轻量设置：unconditional 到 20 epoch，pattern 到 40 epoch。该设置已经足够验证阶段扩展可行性。

进入下一阶段标准：

- loss 明显下降；
- Smatch 或 Validity 至少有非零；
- 若 Jaccard/Dice/Overlap 仍为 0，在报告中解释为小数据和无 RL 限制。

### S2：`db50_s500`

目标：形成可报告的小规模复现实验。

建议：

- unconditional：100 epoch
- pattern：再训练 50 epoch
- baseline 测试全量 test；
- reranking 先用 `--test_proportion 0.1` 或 `0.25` 抽样验证，不建议直接 full rerank。

进入下一阶段标准：

- 指标稳定优于 S1，或者至少结构指标更稳定；
- 候选日志里出现非零 reward 样本。

### S3：`db50_s100`

目标：中等规模 supervised 复现。

建议：

- sampling 可能开始耗时，先用 `-p 8`，机器稳定后再提高；
- unconditional：100 到 200 epoch；
- pattern：50 epoch；
- 先 `--test_proportion 0.1` 快速测试，再全量 test。

进入下一阶段标准：

- test 过程可接受；
- 不爆显存；
- 指标不再完全依赖随机候选。

### S4：`db50_s50`

目标：较稳定 supervised 结果。

建议：

- batch size 16 起步；
- checkpoint 保存间隔设大一些，例如 `--save_frequency 10`；
- 先只做 pattern 条件；
- 暂不跑 relation/entity 其他条件。

进入下一阶段标准：

- 训练时间和硬盘占用可接受；
- 至少能完整跑 baseline 和 rerank 对比。

### S5：`db50_s10`

目标：大规模本地压力测试，接近论文实验量级前的最后验证。

建议：

- 只在 S4 稳定后尝试；
- 先短 epoch 试跑，比如 unconditional 10 epoch；
- 测试先用 `--test_proportion 0.05` 或 `0.1`；
- 保存完整日志，记录耗时。

进入 full 标准：

- 采样、训练、测试都有明确耗时估计；
- 显存不爆；
- 硬盘空间足够；
- 中途 resume 路径确认无误。

### S6：`full`

目标：官方 DBpedia50 full。

采样命令：

```bash
python -m akgr.sampling.sample_parallel -s full -a 32 -p 16
```

从零训练不直接使用官方 `db-g2.sh` 的 `-r=380`。建议改成：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d DBpedia50 --scale full -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training --condition unconditional \
  --overwrite_batchsize 16 --save_frequency 10
```

条件训练：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern \
  --data_root ./sampled_data/ -d DBpedia50 --scale full -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training -r <unconditional_epoch> \
  --overwrite_batchsize 16 --save_frequency 10
```

论文完整设置是 unconditional 400 epoch、pattern 条件 50 epoch。但本地 8GB GPU 很可能无法按论文硬件配置完成，应在报告中说明：

- 数据集使用 DBpedia50 full；
- batch size 小于论文；
- epoch 可能少于论文；
- 未做或只部分做 RL；
- 只复现 pattern 条件，其他条件可作为扩展。

## 5. 每阶段必须记录

每次推进一个阶段，都在运行日志中记录：

- scale 名称；
- 采样命令；
- train/valid/test 行数；
- 训练命令；
- checkpoint 路径；
- 训练 loss；
- 测试命令；
- score CSV 路径；
- Smatch / Jaccard / Dice / Overlap / Validity；
- rerank candidate log 摘要；
- 报错与修复。

建议记录文件继续使用：

```text
CtrlHGen_debug_run_log.md
```

## 6. 报告中的表述口径

推荐最终报告这样表述：

```text
由于官方仓库中 full sampling 配置仅包含 DBpedia50，而 DBpedia50 的采样入口与训练脚本能够形成自洽流程，本项目选择 DBpedia50 作为主要复现数据集。实验按 scale 从 debug/small 逐步扩大到 full，优先复现 pattern 条件下的 supervised training 与 testing，并使用 Jaccard、Dice、Overlap、Smatch、Validity 等指标评估。该结果可视为论文在 DBpedia50 单数据集上的部分复现，不等同于完整复现论文跨 DBpedia50、WN18RR、FB15k-237 的全部结论。
```

## 7. 暂不优先做的内容

以下内容放到 DBpedia50 supervised 稳定后再做：

- relation-number / entity-number / specific-relation / specific-entity 全条件实验；
- 子逻辑分解增强 `sample_add.py`；
- GRPO/RL；
- FB15k-237 与 WN18RR full；
- 多 GPU accelerate 完整复现。
