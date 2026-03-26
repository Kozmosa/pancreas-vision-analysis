# PROJECT_BASIS

## Purpose

本文件作为当前仓库从数据工作区过渡到最小可运行工程时的实施依据，记录每一步做了什么、为什么这样做、有哪些约束还没有解决。由于用户要求在构建 `src/`、训练模型和推进需求时持续更新文档，本文件承担阶段性依据与工作日志双重角色。

## Step 1 - Reality Check And Constraint Freeze

### What was verified

- 仓库当前没有 `pyproject.toml`、`requirements.txt`、`environment.yml`、测试目录或现成训练脚本。
- `docs/ProjectDemand.md` 是当前唯一权威需求文档，要求做 ADM vs PanIN CNN 分类，并报告准确率、灵敏度、特异度和 ROC。
- `docs/Answer_Batch_2.md` 是现有数据语义的主要依据。
- `docs/agentic-work/DataMigratingImpl0323.md` 已把中文目录复制到英文语义目录，后续工程应优先使用英文目录。
- 用户在 2026-03-26 的新指令中明确给出两条新前提：本地可直接使用 `4x A800` 启动实验；当前目录语义可视为病理标注结果。

### Constraint statement

- 不能假装仓库已存在成熟训练平台；只能从最小工程开始。
- 不能把文件夹名直接当成权威病理标签；当前只能称为 working labels。
- 不能声称 7:3 切分已经解决数据泄漏，因为缺失鼠级/切片级 provenance。

### Constraint update after user clarification

- 从本轮开始，`data/caerulein_adm/`、`data/multichannel_adm/`、`data/multichannel_kc_adm/`、`data/KC/`、`data/KPC/` 的目录语义被视为当前实验阶段可直接使用的病理标签。
- 这解决了“目录是否只能作为弱标签”的前一轮保守假设，但并不自动解决鼠级/病灶级 split 的泄漏风险。
- 本地 `4x A800` 已可用，因此后续实验默认按多卡可并行 benchmark 的资源上限设计。

## Step 2 - Data Basis For The First Baseline

### Audited buckets

- `data/caerulein_adm/`: 12 张，按现有说明作为 ADM working label。
- `data/KC/`: 15 张，按现有说明作为 PanIN working label。
- `data/KPC/`: 8 张，按现有说明作为 early PanIN working label。
- `data/multichannel_adm/`: 9 张，暂不作为主线训练输入。
- `data/multichannel_kc_adm/`: 9 张，暂不作为主线训练输入。
- `data/multichannel_unresolved/`: 35 张，因标签不明确排除出主线 baseline。

### Chosen first-task dataset

- 正类/类别 1: `PanIN = KC + KPC`
- 负类/类别 0: `ADM = caerulein_adm`

### Why this choice

- 它最接近 `ProjectDemand` 的 ADM/PanIN 二分类目标。
- 它避免把 `multichannel_unresolved` 这类不确定目录掺入主线。
- 它能立刻形成可运行基线，同时把高风险假设局限在文档中明示。

## Step 3 - Minimal Engineering Decisions

### Code organization

- `src/pancreas_vision/data.py`: 数据发现、记录结构、切分与 manifest 输出。
- `src/pancreas_vision/training.py`: transform、模型、训练、评估和指标保存。
- `src/train_baseline.py`: 命令行入口。

### Modeling choice

- 使用 `torchvision` 预训练 `ResNet18`。
- 理由：小样本下比从零训练 CNN 更稳，且能快速完成第一轮验证。

### Evaluation choice

- 输出 `accuracy`, `sensitivity`, `specificity`, `roc_auc`。
- 这些指标直接对齐 `docs/ProjectDemand.md` 中的需求表述。

## Step 4 - Explicit Limitations Of This Round

- 当前切分仍是图像级而非鼠级/切片级，存在泄漏风险。
- 当前标签来自文件夹语义与说明文档，不是病理金标准。
- 当前 baseline 只证明工程可运行，不证明生物学结论已成立。
- 当前未引入传统免疫组化方法对照，因为仓库尚无规范化对照表。

## Step 5 - Completion Marker For This Round

- 已完成：工程骨架建立。
- 已完成：baseline 数据流与指标实现。

## Step 6 - First Training Run And Immediate Interpretation

### Executed run

运行命令：`PYTHONPATH=src python src/train_baseline.py --epochs 12 --batch-size 8 --image-size 224 --output-dir artifacts/baseline`

### Produced artifacts

- `artifacts/baseline/metrics.json`
- `artifacts/baseline/train_manifest.csv`
- `artifacts/baseline/test_manifest.csv`

### Observed metrics

- Accuracy: 1.000
- Sensitivity: 1.000
- Specificity: 1.000
- ROC AUC: 1.000
- Test confusion counts: TN=4, FP=0, FN=0, TP=7

### Interpretation guardrails

- 这个结果只能说明当前弱标签、小样本、图像级切分的 baseline 在一次运行中可以把测试集完全分开。
- 这个结果不能直接解释为模型已经真实学会 ADM vs PanIN 的稳定病理差异，因为测试集只有 11 张图，而且可能存在同源样本或相似视野泄漏。
- `caerulein_adm/40x-merge.tif` 等特殊命名文件也被当作单独图像样本纳入，说明当前实验混合了普通截图与 merge 图，后续需要更严格的数据协议。
- 因此，本轮最重要的产出是“工程已经跑通并暴露出数据协议风险”，而不是“需求已经被科学上完全实现”。

### Completion marker update

- 已完成：工程骨架建立。
- 已完成：baseline 数据流与指标实现。
- 已完成：首轮训练与结果归档。
- 下一步重点：收紧数据协议，做更保守的 split 与误差分析。

## Step 7 - Multi-Experiment Expansion On Local 4x A800

### Experiment objective

- 在新的用户前提下，把目录语义当作病理标注使用。
- 利用本地 `4x A800` 并行启动多组基线实验，而不是只做单次验证。
- 同时把“下一步提高模型精度需要的数据”写入文档闭环。

### Engineering updates

- `src/pancreas_vision/data.py` 现在为每张图额外记录 `lesion_id`、`magnification`、`channel_name`，并支持是否纳入 `multichannel_*` 目录。
- `src/train_baseline.py` 新增 `--include-multichannel` 参数，并把训练历史、参数和运行摘要保存到 `experiment_summary.json` 与 `history.json`。
- `src/pancreas_vision/training.py` 新增逐 epoch 历史保存逻辑，便于后续多实验比较。

### Executed experiment matrix

- `artifacts/exp_single_image_all/`: 单图像基线，使用 `caerulein_adm + KC + KPC`
- `artifacts/exp_single_image_kc_only/`: 单图像基线，排除 `KPC`
- `artifacts/exp_with_multichannel/`: 单图像基线，纳入 `multichannel_adm` 与 `multichannel_kc_adm`
- `artifacts/exp_frozen_backbone/`: 单图像基线，冻结预训练 backbone

### Resource note

- 四组实验分别绑定到 `CUDA_VISIBLE_DEVICES=0/1/2/3` 并行运行。
- 单次实验时长约 `52s` 到 `88s`，总 GPU 时间约 `0.08 GPU-hours`，远低于当前本地资源上限。

### Observed results

- `exp_single_image_all`: accuracy `1.000`, sensitivity `1.000`, specificity `1.000`, roc_auc `1.000`
- `exp_single_image_kc_only`: accuracy `1.000`, sensitivity `1.000`, specificity `1.000`, roc_auc `1.000`
- `exp_with_multichannel`: accuracy `0.875`, sensitivity `0.857`, specificity `0.889`, roc_auc `0.984`
- `exp_frozen_backbone`: accuracy `0.727`, sensitivity `1.000`, specificity `0.250`, roc_auc `1.000`

### Interpretation

- 预训练 backbone 的可训练性对当前任务非常关键；冻结 backbone 会明显伤害特异度。
- 把 `multichannel_*` 目录直接当作普通单图像样本并入训练后，准确率反而下降，说明这些图更适合按“同病灶多视角/多通道组样本”建模，而不是平铺成独立图像。
- 目前最强信号不是“架构已经到顶”，而是“样本组织单位定义错误会立刻吃掉性能”。

## Step 8 - Data Needed For The Next Accuracy Gain

### Highest priority

- 病灶级 linkage 表：明确哪些 `10x/20x/40x` 与 `merge/amylase/ck-19` 属于同一病灶。
- 鼠级或 specimen 级 ID：让 split 从图像级提升到鼠级/病灶级，避免同源样本泄漏。
- 更多平衡的 ADM 与早期 PanIN 样本，尤其是当前数量偏少的早期、多样化 PanIN 形态。
- 容易误判为导管的 hard negatives，包括血管、炎症组织、良性导管样结构。

### Second-phase value

- ROI 框或粗分割掩码，帮助模型聚焦病灶区域而不是背景和染色边缘。
- 混合病灶或边界病例的置信度标签，便于训练时显式处理不确定样本。
- 更完整的通道级标志物确认，例如 Sox9、Claudin18、CK19、amylase 在病灶级的配对结果。
- 批次、采集日期、设备与操作者信息，用于分层评估 domain shift。

### Practical conclusion

- 下一步精度提升最应优先投入的是“病灶级数据组织和 hard negatives”，而不是继续堆更复杂的单图像分类器。

## Step 9 - Error Analysis For The Multichannel Run

### Why this analysis was added

- 用户选择先做误差分析与可视化方向，因此本轮先把评估输出扩展到样本级预测记录。
- 现在每次训练除 `metrics.json` 外，还会额外输出 `predictions.json`，便于直接定位错例。

### Files produced

- `artifacts/exp_with_multichannel_error_analysis/predictions.json`
- `artifacts/exp_with_multichannel_error_analysis/test_manifest.csv`
- `artifacts/exp_with_multichannel_error_analysis/experiment_summary.json`

### Error summary

- 该实验测试集共 `16` 张图，错 `2` 张。
- false positive: `data/multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif`
- false negative: `data/KPC/1-amylase.tif`

### What the two mistakes suggest

- `multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif` 被预测成 PanIN，说明模型在 `KC-adm` 这类 ADM 标签样本上，可能更容易被 `KC` 背景名义或 `CK19` 导管样形态吸引到 PanIN 决策边界。
- `KPC/1-amylase.tif` 被预测成 ADM，说明单独 `amylase` 通道可能强化了腺泡样信号，使早期 PanIN 在缺乏配对上下文时向 ADM 偏移。
- 两个错例都不是普通明场单图，而是“通道特异视图”，这进一步支持下一步不要把多通道图像平铺成独立样本。

### Immediate interpretation

- 当前模型对普通单图像样本已经非常容易拟合。
- 真正的困难样本集中在带有明确 marker 语义的单通道图，尤其是 `CK19` 与 `amylase` 这类会把病灶解释推向不同方向的视图。
- 这意味着下一步的可视化重点不应只是看普通错图，而应优先看“同一病灶的不同通道是否给出相反证据”。

### Recommended visualization checklist

- 对错例优先做四联图：原图、预测标签与分数、真实标签、来源 bucket。
- 对同病灶可配对样本，按 `merge / amylase / ck-19` 与 `10x / 20x / 40x` 并排查看，检查不同视图是否推动了相反判断。
- 后续如果加入 Grad-CAM，先重点看 `data/multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif` 与 `data/KPC/1-amylase.tif` 这两个错例。

### Data implication from error analysis

- 最应补充的是“同病灶多通道配对表”，否则模型无法知道 `amylase` 与 `CK19` 是互补证据而不是彼此独立样本。
- 需要更多 `KC-adm` 风格 ADM 样本和更多 `KPC amylase` 风格 PanIN 样本，专门覆盖当前这两类边界情况。
