# Idea Discovery Report

**Direction**: 根据项目中已有的数据、`docs/ProjectDemand.md` 与参考文献，搭建一个最小可运行的 ADM vs PanIN 显微图像分类工程，并把过程和约束持续记录到文档中。
**Date**: 2026-03-26
**Pipeline**: repo reality check -> data audit -> baseline idea selection -> implementation bootstrap -> first training run

## Executive Summary

当前仓库与 `docs/ProjectDemand.md` 一致，目标是做 ADM 与 PanIN 的 AI 形态学区分，但实际可用资产仍然是一个小样本、弱标签、无权威元数据的数据工作区。基于现有条件，最稳妥的起步方向不是直接承诺完整病理级结论，而是先建立一个可以复现的数据清单、7:3 图像级切分、预训练 CNN baseline 和文档化限制说明。

在此基础上，本轮已选择并实现一个最小工程：使用 `data/caerulein_adm/` 作为 ADM 代理标签，使用 `data/KC/` 与 `data/KPC/` 作为 PanIN 代理标签，训练一个 ImageNet 预训练 `ResNet18` 二分类 baseline，并将结果写入 `artifacts/baseline/`。这能为后续人工校核、补元数据、扩大样本与更稳健实验提供可执行起点。

## Literature And Requirement Landscape

- `docs/ProjectDemand.md` 明确要求以 CNN 为起点，做 ADM/PanIN 区分，并报告准确率、灵敏度、特异度与 ROC。
- `docs/Answer_Batch_2.md` 提供了当前唯一可依赖的工作语义：`雨蛙素` 偏 ADM，`kc` 偏 PanIN，`KPC` 为早期 PanIN，`merge/amylase/ck-19` 对应同一病灶区域不同通道，`10x/20x/40x` 为逐级放大。
- `docs/refs.md` 与本地 PDF 文献支持两个关键判断：一是胰腺癌早期病变具有重要临床意义；二是显微图像上的 AI/CNN 路线具有合理性，但必须注意病理标签质量与外部验证。
- 仓库现状同时暴露出关键缺口：没有鼠级/切片级 provenance，没有权威标注，没有 train/val/test 规范，也没有现成训练工程。

## Data Reality Check

- 当前英语化后的可用目录包含：`data/caerulein_adm/` 12 张，`data/KC/` 15 张，`data/KPC/` 8 张，`data/multichannel_adm/` 9 张，`data/multichannel_kc_adm/` 9 张，`data/multichannel_unresolved/` 35 张。
- 可直接用于最小二分类 baseline 的是单图像弱标签集合：`caerulein_adm` vs `KC + KPC`。
- 多通道组图虽然生物学信息更强，但样本极少且字段命名不统一，不适合作为第一步唯一建模对象。
- 所有抽样 TIFF 当前均可被 `PIL` 读取，常见尺寸为 `2048x981` 或 `2560x1277`，颜色模式为 `RGBA`。
- 运行环境具备 `torch`, `torchvision`, `sklearn`, `numpy` 与 CUDA GPU，可支持小规模快速实验。

## Ranked Ideas

### 1. Weakly Supervised Image-Level ADM vs PanIN Baseline — RECOMMENDED

- Hypothesis: 即使标签只来自文件夹与研究说明，预训练 CNN 也可能学到可分辨的早期组织形态模式，形成第一版量化基线。
- Why now: 与 `ProjectDemand` 最一致，工程实现最短，且最能暴露数据/标签问题。
- Required assumptions: `caerulein_adm -> ADM`，`KC/KPC -> PanIN` 为工作标签；7:3 先做图像级切分。
- Main risks: 数据泄漏风险、样本量极小、弱标签噪声高、无法代表真实临床泛化。
- Decision: 立即实现。

### 2. Multi-Channel Paired-View Classifier — BACKUP

- Hypothesis: 合并 `merge`, `amylase`, `ck-19` 三类图像后，可得到比单 RGB 截图更强的形态与标志物提示。
- Why backup: 当前仅 9 组 ADM 和 9 组 KC-ADM 风格样本，且缺少 PanIN 对应配对，不足以支撑主线实验。
- Blocker: 标签体系未闭合，数据太少。

### 3. Patch Mining Over Whole Images — DEFERRED

- Hypothesis: 从大图中裁 patch 可扩充样本量并减少背景干扰。
- Why deferred: 没有病灶框与像素标注，盲目切 patch 容易把背景、空白和血管伪差引入训练。
- Blocker: 需要至少轻量病理校核或 ROI 标记。

## Selected Implementation Plan

- 建立 `src/` 下最小 Python 工程，不引入额外框架。
- 实现数据发现、弱标签记录、7:3 分层切分、manifest 输出。
- 使用预训练 `ResNet18` 做二分类 baseline。
- 计算 `accuracy`, `sensitivity`, `specificity`, `roc_auc`，与 `ProjectDemand` 对齐。
- 把所有强假设与风险同步写入工程文档，避免把 baseline 误读为最终生物学结论。

## Deliverables Produced In This Round

- Code: `src/pancreas_vision/data.py`
- Code: `src/pancreas_vision/training.py`
- Entrypoint: `src/train_baseline.py`
- Work log: `docs/agentic-work/PROJECT_BASIS.md`
- This report: `docs/agentic-work/IDEA_REPORT.md`
- Run artifacts: `artifacts/baseline/metrics.json`, `artifacts/baseline/train_manifest.csv`, `artifacts/baseline/test_manifest.csv`

## First Run Result

- Command: `PYTHONPATH=src python src/train_baseline.py --epochs 12 --batch-size 8 --image-size 224 --output-dir artifacts/baseline`
- Outcome: 成功完成首轮训练并输出指标。
- Metrics: accuracy=1.0, sensitivity=1.0, specificity=1.0, roc_auc=1.0
- Immediate reading: 结果过于理想，更可能说明当前样本极少且切分过于宽松，需要把它视为工程验证信号，而不是最终研究结论。
- Main caution: 现有实验仍混合弱标签、可能同源视野、不同命名风格图像，尚不满足高可信病理研究标准。

## Next Steps

- 做第二轮更保守实验，例如排除 `merge` 风格文件、排除 `KPC`、或按命名组近似去重切分。
- 补充误差分析与 Grad-CAM/特征可视化，检查模型是否在利用病灶而非边框、标尺或染色风格差异。
- 争取获取鼠级/切片级 metadata 或人工病理复核，替换当前图像级弱标签切分。

## Update After New User Clarification

- 用户已明确：本地有 `4x A800` 可直接启动实验；当前目录语义按病理标注结果使用。
- 因此本报告中的“弱监督/working label”表述在后续实验阶段收紧为“当前可执行标注来源”，但 split 风险分析继续保留。
- 基于这两个新前提，本轮自动扩展实验后得到的结论是：
  - 单图像预训练 `ResNet18` 在当前目录标签下可轻松得到接近满分结果。
  - 冻结 backbone 会显著损害表现，说明需要保留端到端微调。
  - 把 `multichannel_*` 数据平铺成普通单图像样本会拉低准确率，提示下一阶段主线应转向“病灶级多视角/多通道建模”。
- 因而当前 pipeline 的推荐主线已从“单图像 baseline”升级为：以单图像 CNN 作为对照，以病灶级多视角、多通道融合模型作为主模型方向。

## Data Request For Accuracy Improvement

- 最值得补充的数据已同步沉淀到 `docs/agentic-work/PROJECT_BASIS.md`：
  - 病灶级 `10x/20x/40x + merge/amylase/ck-19` 配对关系表
  - 鼠级或 specimen 级 ID
  - 更多平衡的 ADM 与早期 PanIN 样本
  - 血管、炎症、良性导管样结构等 hard negatives
  - ROI 标注与标志物级确认信息
