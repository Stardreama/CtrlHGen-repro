# CtrlHGen debug 运行记录

## 2026-05-27 环境与最小流程

### 环境修复

- 已安装 PyTorch `2.6.0+cu118`，验证 `torch.cuda.is_available()` 为 `True`。
- `akgr` 首次导入失败：
  - 报错：`TypeError: Too few parameters for <class 'class_resolver.func.FunctionResolver'>`
  - 原因：`pykeen==1.11.0` 与自动安装的 `class-resolver==0.6.1` 不兼容。
  - 修复命令：`python -m pip install class-resolver==0.5.4`
- 验证通过：
  - `python -c "import pykeen.datasets as d; print('pykeen datasets import ok')"`
  - `python -c "import akgr; print('akgr import ok')"`
  - `python -c "import akgr.abduction_model.main as m; print('main import ok')"`

### 代码修改

- `akgr/configs/config-sampling.yml`
  - 新增 `debug_wn18rr`，只采 WN18RR，避免 debug 首轮同时下载和采样三个数据集。
- `akgr/kgdata/load_kg_util.py`
  - WN18RR 实体名保持字符串，不再转换为 NLTK `Synset`，避免 KG 缓存 pickle 失败。
  - 写 KG 缓存前自动创建目录。
- `akgr/abduction_model/transformer.py`
  - GPT2 不再依赖缺失的 `./hug_model`。
  - 将 `num_layers` 映射为 GPT2 的 `n_layer`。
  - debug 默认 GPT2 配置：`n_layer=6, n_embd=256, n_head=4, n_positions=256`。
- `akgr/abduction_model/main.py`
  - 修复 resume checkpoint 路径 f-string。
  - 修复 `entitynumber` 分支误用 pattern extractor。
  - 新增 `--override_nepoch`，用于 debug 小规模训练覆盖 epoch 数。
- `akgr/tokenizer.py`
  - 修复 unconditional extractor 多返回一个 `target` 导致 unpack 失败的问题。

### Sampling

命令：

```bash
mkdir -p sampled_data/WN18RR
python -m akgr.sampling.sample_parallel -s debug_wn18rr -a 32 -p 1
```

结果：

- 数据集：WN18RR
- scale：`debug_wn18rr`
- 每个 pattern 实际采样：train 29、valid 3、test 3
- 输出：
  - `sampled_data/WN18RR/WN18RR-debug_wn18rr-32-train-a2q.jsonl`：377 行
  - `sampled_data/WN18RR/WN18RR-debug_wn18rr-32-valid-a2q.jsonl`：39 行
  - `sampled_data/WN18RR/WN18RR-debug_wn18rr-32-test-a2q.jsonl`：39 行
  - `sampled_data/WN18RR/WN18RR.pkl`
  - `sampled_data/WN18RR/stats.txt`

### Unconditional supervised training

命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --save_frequency 1 --mode training --condition unconditional \
  --overwrite_batchsize 16 --override_nepoch 1
```

结果：

- train loss：`9.569142659505209`
- checkpoint：`checkpoints/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-1-unconditional.pth`

### Pattern conditional supervised training

命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --save_frequency 1 --mode training --condition pattern -r 1 \
  --overwrite_batchsize 16 --override_nepoch 2
```

结果：

- train loss：`8.688724001248678`
- checkpoint：`checkpoints/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-2-pattern.pth`

### Testing

命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ -r 2 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16
```

结果文件：

- `results/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-2-scores(test|1.0xtest_topk0_False_False).csv`

整体指标（39 条 test 样本）：

- Smatch：`0.0`
- Jaccard：`0.0`
- Dice：`0.0`
- Overlap：`0.0`
- Validity / Accuracy：`0.0`

说明：当前只训练了无条件 1 epoch + pattern 条件 1 epoch，且使用随机初始化的小 GPT2 debug 配置。该结果只能说明最小流程、数据、checkpoint、测试和指标生成链路已经跑通，不能代表论文完整复现效果。

## 2026-05-29 Task2 / Task3：推理阶段 reranking

### 继续训练到 50 epoch

为避免 2 epoch/10 epoch 模型几乎不会生成可执行 action，继续在同一 WN18RR debug 数据上训练到 50 epoch。

命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --save_frequency 10 --mode training --condition pattern --tuning -r 10 \
  --overwrite_batchsize 16 --override_nepoch 50
```

结果：

- epoch 50 train loss：`1.7401197055975597`
- checkpoint：`checkpoints/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-50-pattern.pth`

### Task2 改进：reward reranking / MBR-style reranking

修改文件：

- `akgr/abduction_model/main.py`

新增参数：

- `--rerank_k`：每个样本生成候选数，默认 1。
- `--rerank_alpha`：语义分与条件分混合权重，默认 0.5。
- `--rerank_log_candidates`：保存候选、候选分数和是否被选中。

重排分数：

```text
semantic = Jaccard + 0.5 * Dice + 0.5 * Overlap
condition = pattern validity / relation-number / entity-number / specific condition score
reward = alpha * semantic + (1 - alpha) * condition
```

本实验条件为 `pattern`，所以 condition 使用 `validity`。

### Task3 对比实验

共同设置：

- 数据集：WN18RR
- scale：`debug_wn18rr`
- checkpoint：`WN18RR-debug_wn18rr-32-50-pattern.pth`
- condition：pattern
- test set：39 条
- constrained decoding：开启
- baseline：`rerank_k=1`
- improved：`rerank_k=4`

baseline 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ -r 50 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True --rerank_k 1
```

improved 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d WN18RR --scale debug_wn18rr -a 32 \
  --checkpoint_root checkpoints/ -r 50 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True \
  --rerank_k 4 --rerank_alpha 0.5 --rerank_log_candidates
```

结果文件：

- baseline：`results/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-50-scores(test|1.0xtest_topk0_True_False).csv`
- improved：`results/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-50-scores(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).csv`
- candidate log：`results/GPT2_6_act_nt/WN18RR-debug_wn18rr-32-50-candidates(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).jsonl`

整体指标：

| method | rerank_k | Smatch | Jaccard | Dice | Overlap | Validity | Entity-number | Relation-number |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0.0267 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1026 | 0.0000 |
| improved | 4 | 0.0928 | 0.0000 | 0.0000 | 0.0000 | 0.1026 | 0.1282 | 0.1026 |

候选日志摘要：

- 候选总数：156
- 被选中候选：39
- 非零 reward 候选：6
- 非零 reward 且被选中：4
- max reward：0.5

解释：

- `rerank_k=4` 相比 baseline 提高了 Smatch 和 pattern validity，说明多个候选中确实出现了结构更接近目标 pattern 的结果，重排能挑出来。
- Jaccard/Dice/Overlap 仍为 0，说明 debug 训练虽然学到部分结构，但生成的具体实体/关系还没有让 KG 执行答案集合命中 observation。
- 该结果符合小规模复现实验预期：推理重排能改善结构控制，但语义命中仍受训练规模、训练轮数、模型大小和未做 RL 的限制。

## DBpedia50 full 路线 S0：scale=5000 探路

目标：按 `DBpedia50 full` 自洽路线先跑最小阶段 S0，验证完整流程是否可用，包括采样、unconditional 训练、pattern 训练、baseline 测试和 rerank 对照测试。

### S0 采样

配置项：`config-sampling.yml` 中新增 `db50_s5000`

```yaml
db50_s5000:
  datasets: ["DBpedia50"]
  scale: 5000
```

执行命令：

```bash
mkdir -p sampled_data/DBpedia50
python -m akgr.sampling.sample_parallel -s db50_s5000 -a 32 -p 4
```

采样输出：

- `sampled_data/DBpedia50/DBpedia50-db50_s5000-32-train-a2q.jsonl`：143 条
- `sampled_data/DBpedia50/DBpedia50-db50_s5000-32-valid-a2q.jsonl`：13 条
- `sampled_data/DBpedia50/DBpedia50-db50_s5000-32-test-a2q.jsonl`：13 条
- `sampled_data/DBpedia50/DBpedia50.pkl`
- `sampled_data/DBpedia50/stats.txt`

KG 统计：

- nentity：24624
- nrelation：702
- reverse 后 train/valid/test edge 数：55074 / 6884 / 6884

说明：PyKEEN 在加载 DBpedia50 时提示部分 valid/test 实体或关系不在 training set 中，并过滤对应 triples。这是数据集划分本身导致的 warning，本阶段未阻断流程。

### S0 unconditional 训练

执行命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s5000 -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training --condition unconditional \
  --overwrite_batchsize 16 --save_frequency 10 --override_nepoch 20
```

产物：

- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s5000-32-10-unconditional.pth`
- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s5000-32-20-unconditional.pth`

### S0 pattern 训练

执行命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s5000 -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training -r 20 \
  --overwrite_batchsize 16 --save_frequency 10 --override_nepoch 40
```

产物：

- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s5000-32-30-pattern.pth`
- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s5000-32-40-pattern.pth`

注意：这里是从 unconditional epoch 20 初始化 pattern 训练，不加 `--tuning`。`--tuning` 只用于继续已有 pattern checkpoint。

### S0 测试与 rerank 对照

baseline 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s5000 -a 32 \
  --checkpoint_root checkpoints/ -r 40 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True --rerank_k 1
```

rerank 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s5000 -a 32 \
  --checkpoint_root checkpoints/ -r 40 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True \
  --rerank_k 4 --rerank_alpha 0.5 --rerank_log_candidates
```

结果文件：

- baseline：`results/GPT2_6_act_nt/DBpedia50-db50_s5000-32-40-scores(test|1.0xtest_topk0_True_False).csv`
- improved：`results/GPT2_6_act_nt/DBpedia50-db50_s5000-32-40-scores(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).csv`
- candidate log：`results/GPT2_6_act_nt/DBpedia50-db50_s5000-32-40-candidates(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).jsonl`

整体指标：

| method | rerank_k | Smatch | Jaccard | Dice | Overlap | Validity | Entity-number | Relation-number |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0.0110 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| improved | 4 | 0.0944 | 0.0000 | 0.0000 | 0.0000 | 0.0769 | 0.0769 | 0.0769 |

候选日志摘要：

- 候选总数：52
- 被选中候选：13
- 非零 reward 候选：1
- 非零 reward 且被选中：1
- max reward：0.5

阶段结论：

- S0 已验证 DBpedia50 自洽路线可跑通：采样、两阶段训练、测试、rerank 对照都能产出结果。
- 在 13 条 test 样本上，`rerank_k=4` 相比 `rerank_k=1` 提高了 Smatch，并带来非零 pattern validity / entity-number / relation-number。
- 答案集合类指标仍为 0，说明 S0 只适合作为流程探路，不适合报告为正式复现结果。
- 下一阶段建议进入 S1：`scale=1000`，约 train 715、valid/test 78。S1 仍属于小规模，但比 S0 更适合观察训练和指标是否稳定。

## DBpedia50 full 路线 S1：scale=1000 小规模验证

目标：在 S0 路线跑通后，将 DBpedia50 采样规模扩大 5 倍，验证训练和测试指标是否仍能稳定产出。

### S1 采样

配置项：`config-sampling.yml` 中新增 `db50_s1000`

```yaml
db50_s1000:
  datasets: ["DBpedia50"]
  scale: 1000
```

执行命令：

```bash
python -m akgr.sampling.sample_parallel -s db50_s1000 -a 32 -p 4
```

采样输出：

- `sampled_data/DBpedia50/DBpedia50-db50_s1000-32-train-a2q.jsonl`：715 条
- `sampled_data/DBpedia50/DBpedia50-db50_s1000-32-valid-a2q.jsonl`：78 条
- `sampled_data/DBpedia50/DBpedia50-db50_s1000-32-test-a2q.jsonl`：78 条

KG 统计沿用 DBpedia50：

- nentity：24624
- nrelation：702
- reverse 后 train/valid/test edge 数：55074 / 6884 / 6884

### S1 unconditional 训练

执行命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s1000 -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training --condition unconditional \
  --overwrite_batchsize 16 --save_frequency 10 --override_nepoch 20
```

产物：

- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s1000-32-10-unconditional.pth`
- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s1000-32-20-unconditional.pth`

### S1 pattern 训练

执行命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s1000 -a 32 \
  --checkpoint_root checkpoints/ --result_root ./results/ \
  --mode training -r 20 \
  --overwrite_batchsize 16 --save_frequency 10 --override_nepoch 40
```

产物：

- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s1000-32-30-pattern.pth`
- `checkpoints/GPT2_6_act_nt/DBpedia50-db50_s1000-32-40-pattern.pth`

### S1 测试与 rerank 对照

baseline 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s1000 -a 32 \
  --checkpoint_root checkpoints/ -r 40 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True --rerank_k 1
```

rerank 命令：

```bash
python -m akgr.abduction_model.main \
  --modelname GPT2_6_act_nt --condition pattern --tuning \
  --data_root ./sampled_data/ -d DBpedia50 --scale db50_s1000 -a 32 \
  --checkpoint_root checkpoints/ -r 40 --result_root ./results/ \
  --mode testing --test_split test --test_proportion 1 \
  --test_top_k 0 --overwrite_batchsize 16 --constrained True \
  --rerank_k 4 --rerank_alpha 0.5 --rerank_log_candidates
```

结果文件：

- baseline：`results/GPT2_6_act_nt/DBpedia50-db50_s1000-32-40-scores(test|1.0xtest_topk0_True_False).csv`
- improved：`results/GPT2_6_act_nt/DBpedia50-db50_s1000-32-40-scores(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).csv`
- candidate log：`results/GPT2_6_act_nt/DBpedia50-db50_s1000-32-40-candidates(test|1.0xtest_topk0_True_False_rerank4_alpha0.5).jsonl`

整体指标：

| method | rerank_k | Smatch | Jaccard | Dice | Overlap | Validity | Entity-number | Relation-number |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0385 | 0.0000 |
| improved | 4 | 0.0392 | 0.0001 | 0.0001 | 0.0009 | 0.0385 | 0.0513 | 0.0513 |

补充指标：

| method | F1 | Recall | Precision | Tanimoto |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| improved | 0.0001 | 0.0009 | 0.0001 | 0.0001 |

候选日志摘要：

- 候选总数：312
- 被选中候选：78
- 非零 reward 候选：4
- 非零 reward 且被选中：4
- max reward：0.5

耗时观察：

- baseline full test 约 11 分钟。
- rerank full test 约 9 小时 24 分钟，明显慢于训练和 baseline。主要瓶颈在 constrained testing、候选重排和 KG 执行评估。
- 最后一批 14 条样本耗时异常长，是本阶段最大的时间风险点。

阶段结论：

- S1 已验证 DBpedia50 自洽路线可扩展到 715/78/78 的小规模设置。
- `rerank_k=4` 相比 baseline 继续提高 Smatch，并首次出现非零 Jaccard / Dice / Overlap / F1，说明扩大数据后语义答案集合开始有极少量命中。
- 绝对指标仍然很低，S1 仍属于小规模验证，不适合作为最终复现结论。
- S2 可以继续推进到 `scale=500`，但不建议直接 full rerank。建议先全量 baseline，再用 `--test_proportion 0.1` 或 `0.25` 做 rerank 抽样；只有在需要提交最终对照表时再考虑 full rerank。
