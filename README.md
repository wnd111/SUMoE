# SUMoE: Finley Paper-Aligned Public Source Release

This repository provides a paper-aligned implementation and public source release accompanying the paper
《Leveraging Syntactic Uncertainty Driven Mixture-of-Experts Framework to
Enhance Long Text Understanding and Reasoning》. Source code and configurations
are public. Model weights and training checkpoints are not distributed.

Implementation details that map the paper notation to the executable modules
are documented in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)
and [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md). The exact contents and
limits of this source-only archive are stated in
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md).

## 1. 已实现范围

工程包含以下可执行部分：

- Stanza、spaCy `en_core_web_trf`、Stack-Transformer 三解析器；
- 解析器温度校准、LAS 先验、候选树去重、句内全局归一化和 Top-5 依存森林；
- 字符偏移到 LLaMA 子词的严格对齐，首子词继承入边，其余子词使用自环；
- 在 LLaMA 第 4、8、12、16、20、24、28、32 个解码层之后分别使用独立参数严格执行 Eq. (4)–(6) 的森林聚合与门控融合；
- 两层四头 GAT、结构—语义路由、八个独立轻量 Transformer 专家和确定性 Top-2 激活；
- LLaMA、LLaMA+Forest、LLaMA+MoE、DCC、MEO、SUMoE 六个受控条件；
- SCROLLS 的 GovReport、SummScreenFD、QMSum、Qasper、NarrativeQA、QuALITY、ContractNLI 七项任务；
- 全参数训练、DeepSpeed ZeRO-3、BF16、梯度检查点、FlashAttention 2、断点恢复和运行清单；
- greedy 推理、ROUGE-1/2/L、QA F1、QuALITY EM、ContractNLI accuracy；
- 消融、效率、路由统计、三种子配对 t 检验、95% 置信区间和 Holm 校正；
- CPU 微型模型测试、训练烟雾测试和八卡正式训练预检。

## 2. 固定实现规格

| 项目 | 固定值 |
|---|---|
| 基座 | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| 上下文总长度 | 4096 token |
| 最大生成长度 | 512 token |
| 森林候选数 | 每句 5 棵 |
| 森林加权聚合 | `softmax_j(A_ij / τ)`，对全部源 token 归一化，`τ=1.0` |
| 森林注入位置 | 第 4、8、12、16、20、24、28、32 个解码层之后，共 8 个独立模块 |
| GAT | 2 层、4 头、节点维度 256 |
| 路由维度 | 1024 |
| 专家 | 8 个轻量 Transformer block，Top-2，8 头自注意力，SwiGLU 中间维度 4096，dropout 0.1 |
| 任务头 | 七项 SCROLLS 任务使用 LM head 或 4/3 类分类头；另预留模型级 span-head API |
| 损失 | 任务特定损失 + 已含 0.05 系数的平衡损失 + 森林余弦损失 |
| 优化器 | AdamW，LR `3e-5`，betas `(0.9, 0.98)`，epsilon `1e-8`，weight decay `0.01` |
| 批量 | 8 卡 × 每卡 1 × 累积 8 = 全局 64 |
| 数值格式 | BF16 |
| 随机种子 | 13、21、42 |
| 解码 | beam=1，argmax，`do_sample=false` |
| 效率计时 | batch=1，20 次预热，100 次计时，长度 512/1024/2048/4096 |

正文未逐项规定的执行细节在本工程中固定如下：训练 3 个 epoch，3% 线性预热后余弦下降到 `3e-6`；Eq. (4) 中不存在的边按 `A_ij=0` 纳入全 `T` 分母，不加入 Q/K 相似度；Eq. (5) 严格先执行 `FFN(W_V h_j)` 再聚合；依存根转为子词自环；GAT 树向量由根节点和节点均值拼接后线性映射；DCC 训练使用预计算的固定分区、推理使用平方欧氏距离的最近 Top-2；MEO 从已训练 SUMoE checkpoint 导入并合并完整 Transformer expert block 参数；QuALITY 在官方答案文本与 A–D 类别之间双向映射；ContractNLI 使用官方三个规范标签。

Eq. (10) 只定义离散 Top-2 使用频率及其损失值，没有规定离散选择的梯度估计器。本实现以全局有效批量的硬计数计算该值，并用 straight-through estimator 训练 router；跨微批次保存的是 stop-gradient 特征快照，因此该辅助损失不会单独向 backbone 或 tree-readout 回传梯度。任务损失和森林损失仍按通常方式训练上游模块。这个范围是显式固定的工程细节，而不是论文中另一个公式。

模型参数由配置和模型结构即时统计；该命令使用代码内固定的公开 LLaMA-3.1-8B 架构元数据并在 meta device 上构造模型，不访问 gated Hub，也不下载或分发模型权重：

```bash
python scripts/count_parameters.py --config configs/model/sumoe.yaml
```

## 3. 目录

```text
configs/                 论文主配置、ZeRO-3 配置、六条件与敏感性覆盖配置
scripts/                 数据、解析、训练、推理、评价、统计的命令入口
src/sumoe/               完整 Python 实现
tests/                   单元测试、集成测试和 CPU 训练烟雾测试
data/                    运行时生成的数据与依存森林，不进入 Git
checkpoints/             解析器、DCC 和模型检查点，不进入 Git
outputs/                 训练输出，不进入 Git
predictions/             预测结果，不进入 Git
reports/                 指标、效率和统计报告，不进入 Git
```

## 4. 环境

正式复现实验固定使用 Ubuntu 22.04、Python 3.10.14、CUDA 12.4、8 张 NVIDIA H20 80GB。

```bash
conda create -n sumoe-repro python=3.10.14 -y
conda activate sumoe-repro
python -m pip install --upgrade pip==24.2
python -m pip install -r requirements.txt
python -m pip install -e .
python -m spacy download en_core_web_trf
python -c "import stanza; stanza.download('en')"
huggingface-cli login
```

Hugging Face 账户必须已获准访问 `meta-llama/Meta-Llama-3.1-8B-Instruct`。所有命令均从项目根目录执行。

## 5. 下载并规范化七个 SCROLLS 任务

```bash
python scripts/prepare_scrolls.py --dataset tau/scrolls --output-dir data/scrolls
```

输出固定为 `data/scrolls/<task>/<split>.jsonl`。训练集的多参考答案展开为独立训练样本；验证集和测试集保留完整参考答案集合；每个源文档的稳定标识为 `task::split::source_id`。

## 6. PTB 依存数据与 Stack-Transformer

PTB 数据受 LDC 许可证约束，工程不分发语料。将持有许可证的 PTB3 `parsed/mrg/wsj` 放在 `data/ptb/raw/wsj`，将 Stanford CoreNLP 4.5.7 解压到 `third_party/stanford-corenlp-4.5.7`，然后执行：

```bash
mkdir -p data/ptb/sections
find data/ptb/raw/wsj -name '*.mrg' -print0 | while IFS= read -r -d '' file; do
  name=$(basename "$file" .mrg)
  java -cp 'third_party/stanford-corenlp-4.5.7/*' edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile "$file" -conllx > "data/ptb/sections/${name}.conllu"
done
python scripts/prepare_ptb_dependencies.py --section-dir data/ptb/sections --output-dir data/ptb
python scripts/train_transition_parser.py --config configs/parsers/transition_training.yaml
```

划分固定为 WSJ 02–21 训练、22 开发、23 测试。最佳检查点写入 `checkpoints/transition/best.pt`。

## 7. 解析器校准

先在 PTB 开发集上收集三解析器候选的原始分数和 labeled attachment，再拟合温度并按开发集 LAS 归一化解析器先验：

```bash
python scripts/collect_parser_calibration.py --gold data/ptb/ptb-dev.conllu --transition-checkpoint checkpoints/transition/best.pt --output data/parser-calibration-observations.jsonl
python scripts/calibrate_parsers.py --input data/parser-calibration-observations.jsonl --output data/parser-calibration.json
```

## 8. 构建默认 Top-5 依存森林

```bash
for task in gov_report summ_screen_fd qmsum qasper narrative_qa quality contract_nli; do
  for split in train validation test; do
    if [ -f "data/scrolls/${task}/${split}.jsonl" ]; then
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct --top-k 5 --sources stanza spacy transition
    fi
  done
done
```

每个森林文件包含构建清单；同目录生成 `.sha256` 文件。训练预检会拒绝缺失或哈希不一致的森林。

Top-1、Top-3 与单解析器敏感性森林使用独立目录：

```bash
for task in gov_report summ_screen_fd qmsum qasper narrative_qa quality contract_nli; do
  for split in train validation test; do
    if [ -f "data/scrolls/${task}/${split}.jsonl" ]; then
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests-k1/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --top-k 1 --sources stanza spacy transition
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests-k3/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --top-k 3 --sources stanza spacy transition
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests-stanza/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --top-k 5 --sources stanza
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests-spacy/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --top-k 5 --sources spacy
      python scripts/build_forests.py --input "data/scrolls/${task}/${split}.jsonl" --output "data/forests-transition/${task}/${split}.jsonl" --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --top-k 5 --sources transition
    fi
  done
done
```

## 9. 训练前验证与 CPU 烟雾测试

CPU 烟雾测试使用随机初始化的两层微型 LLaMA，执行真实前向、反向、优化和 checkpoint 写入：

```bash
python scripts/train.py --config configs/model/sumoe.yaml --output-dir outputs/smoke --smoke-test
python scripts/train_transition_parser.py --config tests/fixtures/transition_tiny.yaml --smoke-test
pytest -q
```

八卡机器先查看全部检查，再执行严格预检：

```bash
python scripts/preflight.py --config configs/model/sumoe.yaml --check-only
python scripts/preflight.py --config configs/model/sumoe.yaml
```

严格预检要求 Python 和依赖版本完全一致、GPU 数为 8、设备为 H20 80GB、BF16 可用、七任务训练文件存在、森林 SHA-256 有效。

## 10. DCC 固定聚类

DCC 使用无森林 LLaMA 的训练源文本 max-pool 表示，MiniBatchKMeans 参数固定为 8 簇、batch size 4096、`n_init=10`、`max_iter=300`、seed 13。聚类文件同时保存每个训练 `source_id` 的固定分区；训练时只激活对应专家，推理时才组合两个最近中心专家：

```bash
python scripts/extract_dcc_embeddings.py --config configs/model/sumoe.yaml --overlay configs/ablation/llama.yaml --output data/dcc/train-embeddings.pt
python scripts/fit_dcc_clusters.py --embeddings data/dcc/train-embeddings.pt --output checkpoints/dcc/centroids.pt --num-experts 8 --seed 13
```

## 11. 三种子正式训练

完整 SUMoE：

```bash
for seed in 13 21 42; do
  accelerate launch --num_processes 8 scripts/train.py --config configs/model/sumoe.yaml --seed "$seed" --output-dir "outputs/sumoe/seed-${seed}" --distributed
done
```

其余可独立训练的受控条件：

```bash
for condition in llama llama_forest llama_moe dcc; do
  for seed in 13 21 42; do
    accelerate launch --num_processes 8 scripts/train.py --config configs/model/sumoe.yaml --overlay "configs/ablation/${condition}.yaml" --seed "$seed" --output-dir "outputs/${condition}/seed-${seed}" --distributed
  done
done
```

`sumoe` 已由前一命令训练。MEO 不是从随机专家独立训练的条件；它在预测和性能分析时从已训练 SUMoE checkpoint 导入 backbone、semantic router 与完整 expert blocks，再按文档语义权重执行参数合并。运行目录内的 `run-manifest.json` 固定记录合并配置、依赖版本、数据 SHA-256 和森林 SHA-256。

中断后从该运行最后一次完整保存恢复：

```bash
accelerate launch --num_processes 8 scripts/train.py --config configs/model/sumoe.yaml --seed 13 --output-dir outputs/sumoe/seed-13 --resume-from-checkpoint "outputs/sumoe/seed-13/$(cat outputs/sumoe/seed-13/last-checkpoint.txt)" --distributed
```

恢复前会逐项比较当前清单与 checkpoint 清单；配置、依赖、数据或森林发生变化时立即终止。

## 12. 敏感性实验

每次只加载一个覆盖配置：

```bash
for ablation in k1 k3 experts4 experts16 top1 tau05 tau20 no_balance parser_stanza parser_spacy parser_transition; do
  for seed in 13 21 42; do
    accelerate launch --num_processes 8 scripts/train.py --config configs/model/sumoe.yaml --overlay "configs/ablation/${ablation}.yaml" --seed "$seed" --output-dir "outputs/${ablation}/seed-${seed}" --distributed
  done
done
```

## 13. 合并 ZeRO-3 权重

每个正式运行完成后执行：

```bash
python scripts/consolidate_zero3.py --run-dir outputs/sumoe/seed-13 --output-dir outputs/sumoe/seed-13/consolidated
```

输出固定为 `outputs/sumoe/seed-13/consolidated/pytorch_model.bin`，并同时写入强制校验的 `architecture.json`，供预测和性能分析加载。

## 14. 确定性预测与任务指标

以 SUMoE seed 13 验证集为例：

```bash
mkdir -p predictions/sumoe/seed-13 reports/sumoe
for task in gov_report summ_screen_fd qmsum qasper narrative_qa quality contract_nli; do
  python scripts/predict.py --config configs/model/sumoe.yaml --checkpoint outputs/sumoe/seed-13/consolidated --task "$task" --split validation --output "predictions/sumoe/seed-13/${task}.jsonl"
done
cat predictions/sumoe/seed-13/*.jsonl > predictions/sumoe/seed-13/all.jsonl
python scripts/evaluate.py --predictions predictions/sumoe/seed-13/all.jsonl --output reports/sumoe/seed-13.json
```

其他条件在预测时加载与训练相同的覆盖配置。例如 LLaMA seed 13：

```bash
python scripts/predict.py --config configs/model/sumoe.yaml --overlay configs/ablation/llama.yaml --checkpoint outputs/llama/seed-13/consolidated --task gov_report --split validation --output predictions/llama/seed-13/gov_report.jsonl
```

MEO 覆盖配置必须指向同一种子的已训练 SUMoE checkpoint；加载器会剥离 forest wrapper，并逐项验证和导入 backbone、semantic router、任务头及完整 expert blocks：

```bash
python scripts/predict.py --config configs/model/sumoe.yaml --overlay configs/ablation/meo.yaml --checkpoint outputs/sumoe/seed-13/consolidated --task gov_report --split validation --output predictions/meo/seed-13/gov_report.jsonl
```

预测脚本拒绝覆盖已有文件。输入只包含 system、任务指令和源文本；生成任务执行 greedy decoding，QuALITY 与 ContractNLI 直接读取分类头，参考答案仅在预测结束后写入 JSONL 元数据。

## 15. 路由、效率和显著性

SUMoE 路由报告：

```bash
python scripts/routing_statistics.py --assignments predictions/sumoe/seed-13/all.jsonl --output reports/sumoe/routing-seed-13.json --num-experts 8
```

表格式效率指标使用 prefill 模式，分别报告在线解析、森林构建、单次模型前向和完整 prefill 管线：

```bash
python scripts/profile.py --mode prefill --config configs/model/sumoe.yaml --checkpoint outputs/sumoe/seed-13/consolidated --example data/scrolls/gov_report/validation.jsonl --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --output reports/sumoe/profile-prefill-seed-13.json
```

完整的确定性贪心自回归生成需要单独使用 generation 模式；该报告包含首 token 延迟、缓存生成延迟、解析至生成完成的端到端延迟，以及实际生成 token 吞吐率：

```bash
python scripts/profile.py --mode generation --max-new-tokens 512 --generation-warmups 1 --generation-iterations 5 --config configs/model/sumoe.yaml --checkpoint outputs/sumoe/seed-13/consolidated --example data/scrolls/gov_report/validation.jsonl --calibration data/parser-calibration.json --transition-checkpoint checkpoints/transition/best.pt --output reports/sumoe/profile-generation-seed-13.json
```

三种子显著性要求先完成 LLaMA+MoE 与 SUMoE 的六份评价报告，然后执行：

```bash
python scripts/collect_seed_scores.py --baseline reports/llama_moe/seed-13.json reports/llama_moe/seed-21.json reports/llama_moe/seed-42.json --sumoe reports/sumoe/seed-13.json reports/sumoe/seed-21.json reports/sumoe/seed-42.json --output reports/paired-scores.json
python scripts/significance.py --scores reports/paired-scores.json --output reports/significance.json
```

报告给出 SUMoE−baseline 的均值、标准误、df=2、双侧配对 t 检验、95% t 置信区间和 Holm 调整后的 p 值。

## 16. 质量验证

```bash
pytest -q
ruff check .
mypy src/sumoe
python -m compileall -q src scripts
git diff --check
```

测试覆盖森林结构不变量、温度校准、偏移对齐、Eq. (4)–(6) 森林加权聚合、确定性路由、完整 Transformer-expert MEO 参数合并、DCC 最近中心、生成/分类/span 任务头、七任务归一化、缓存解码、全部指标、统计检验、checkpoint 往返和一优化步训练。

## 17. 构建公开发布包

公开发布包只包含源代码、配置、测试和文档；构建器排除内部规划、Git 元数据、缓存、数据、训练输出、报告和模型或检查点文件。该命令拒绝覆盖已有 ZIP：

```bash
python scripts/build_release.py --output SUMoE-Finley-paper-aligned-code-20260803.zip --archive-root SUMoE-Finley-paper-aligned-code
```

## 18. 确定性故障处理

- `forest SHA-256 sidecar mismatch`：删除对应森林及其 `.sha256`，使用第 8 节同一条构建命令重建。
- `source was fully truncated`：输入源文本为空或格式损坏；重新运行 `prepare_scrolls.py`，不得跳过该样本。
- `missing cached forest`：源数据与森林不是同一份规范化输出；重新生成该任务、该 split 的森林。
- `non-finite training loss`：查看运行目录的 `non-finite-batch.json`，保留其中的样本 ID 和运行清单，终止该次训练。
- `configuration SHA-256 changed`：使用原运行目录中的 `run-manifest.json` 对应配置恢复；不同配置启动新的输出目录。
- CUDA OOM：正式规格不降低 4096 长度、全局批量或专家数；确认 ZeRO-3、BF16、梯度检查点和 FlashAttention 2 均已启用，并确认每卡仅一个训练进程。

## 19. 许可

本工程原创代码采用 MIT License。LLaMA、SCROLLS、PTB、Stanza、spaCy、Stanford CoreNLP 及其模型和数据分别服从各自许可证；本工程不重新分发这些权重或数据。
