# 胰腺显微图像 ADM vs PanIN 训练说明与结果分析

## 1. 文档目的

本文档用于汇总本次 `pancreas-vision-analysis` 项目的训练方案、数据来源、CSV 与 JSON 标注文件的作用、训练脚本的实现逻辑，以及本轮结果文件的逐项分析。本文档基于以下实际文件内容整理：

- 项目根目录：`D:\srpgpt\pancreas-vision-analysis`
- 本次训练脚本：`D:\srpgpt\pancreas-vision-analysis\src\train_improved.py`
- 本次结果目录：`D:\srpgpt\pancreas-vision-analysis\artifacts\improved_hybrid_pycharm`
- 辅助元数据：`D:\srpgpt\pancreas-vision-analysis\data\1.csv`
- 辅助元数据：`D:\srpgpt\pancreas-vision-analysis\data\2.csv`
- 辅助元数据：`D:\srpgpt\pancreas-vision-analysis\data\3.csv`
- ROI 标注：`D:\srpgpt\pancreas-vision-analysis\data\KC\7.json`
- ROI 标注：`D:\srpgpt\pancreas-vision-analysis\data\KC\8.json`
- ROI 标注：`D:\srpgpt\pancreas-vision-analysis\data\KC\10.json`
- ROI 标注：`D:\srpgpt\pancreas-vision-analysis\data\KC\11.json`
- ROI 标注：`D:\srpgpt\pancreas-vision-analysis\data\KC\12.json`

## 2. 本次训练的总体结论

本次训练已经成功完成，最终指标如下：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 0.925 |
| Sensitivity | 1.000 |
| Specificity | 0.857 |
| ROC AUC | 0.977 |
| True Positive | 19 |
| False Negative | 0 |
| True Negative | 18 |
| False Positive | 3 |

这组结果的含义是：

- 模型对 `PanIN` 的召回率很高，本轮测试中没有漏检 `PanIN`。
- 模型对 `ADM` 的区分仍有改进空间，共有 3 个 `ADM` 被误判成 `PanIN`。
- 相比早期那种接近满分但更容易受图像级泄漏影响的基线，这一轮结果更可信，因为它引入了 `ROI crop`、`metadata`、`group-aware split` 和更严格的数据组织方式。

## 3. 本次训练所使用的方法

### 3.1 核心脚本

本次训练主要由以下代码文件组成：

- `src/train_improved.py`
- `src/pancreas_vision/data.py`
- `src/pancreas_vision/training.py`

### 3.2 训练流程

本次训练不是简单地扫文件夹后整图分类，而是做了一个混合训练管线：

1. 从现有原始目录中读取整图样本。
2. 从 `data/2.csv` 中读取额外元数据。
3. 把 `KC/*.json` 里的 ROI 多边形转换为 `PanIN ROI crop` 样本。
4. 使用 `group_id / lesion_id` 做分组切分，尽量避免同组样本同时进入训练集和测试集。
5. 用 `WeightedRandomSampler` 缓解类别不平衡。
6. 采用 `ResNet34 + dropout + AdamW + label smoothing + cosine annealing` 进行训练。

### 3.3 模型与训练参数

来自 `experiment_summary.json` 的实际运行参数如下：

| 参数 | 数值 |
| --- | --- |
| 模型骨干 | `resnet34` |
| 输入尺寸 | `224 x 224` |
| Epochs | `24` |
| Batch size | `8` |
| Learning rate | `0.0002` |
| Weight decay | `0.0001` |
| Label smoothing | `0.05` |
| Dropout | `0.2` |
| Test size | `0.3` |
| 随机种子 | `42` |
| 分组切分 | `true` |
| ROI crop | `true` |
| KPC | `true` |
| multichannel | `true` |
| 采样器 | `Weighted sampler = true` |
| ROI padding | `0.12` |

### 3.4 与旧方案相比的改进点

相对于最早的 `train_baseline.py` 整图分类方案，本次改进主要体现在：

- 不再只依赖文件夹标签，而是吸收 `2.csv` 的 metadata。
- 不再只用整图，而是新增了 `PanIN ROI crop` 样本。
- 不再只做普通图像级随机切分，而是按 `group_id` 约束切分。
- 不再只用最基础的训练器，而是加入更稳的优化器和正则策略。

## 4. 原始数据与辅助文件说明

### 4.1 原始图像目录

当前可见的主要图像桶如下：

| 目录 | 文件数 | 当前用途 | 说明 |
| --- | ---: | --- | --- |
| `data/caerulein_adm` | 12 | ADM 整图 | 当前作为 ADM 主来源之一 |
| `data/KC` | 15 | PanIN 整图 + ROI 来源 | 当前作为 PanIN 主来源之一 |
| `data/KPC` | 8 | PanIN 整图 | 当前按早期 PanIN 处理 |
| `data/multichannel_adm` | 9 | ADM 整图 | 多通道、多倍率 ADM 组 |
| `data/multichannel_kc_adm` | 9 | ADM 整图 | `KC-adm` 风格多通道组 |
| `data/multichannel_unresolved` | 35 | 部分由 metadata 解析为补充 ADM | 原本标签不清，现部分通过 `2.csv` 纳入 |

### 4.2 `data/1.csv` 的作用与评价

`1.csv` 更像是一个总览型工作簿首页，主要作用是汇总：

- 总记录数
- 可训练记录数
- 待人工核查记录数
- 不同来源目录的统计

评价如下：

- 优点：适合作为项目管理和人工审核的总表首页。
- 缺点：不能直接驱动训练，也不适合做程序解析。
- 结论：保留价值高，但不是训练输入文件。

### 4.3 `data/2.csv` 的作用与评价

`2.csv` 是目前最关键的辅助元数据文件。它一共 88 行，对应 88 张 tif，字段包括：

- `record_id`
- `source_folder`
- `file_name`
- `full_path`
- `biological_group`
- `background_model`
- `treatment`
- `coarse_label`
- `magnification`
- `image_type`
- `lesion_id`
- `variant_id`
- `naming_confidence`
- `exclude_from_train`
- `needs_manual_check`
- `notes`
- `folder_level_prior`
- `label_basis`

本文件的统计特征如下：

| 项目 | 数值 |
| --- | ---: |
| 总行数 | 88 |
| `ADM-like` | 45 |
| `PanIN-like` | 15 |
| `uncertain` | 28 |
| `exclude_from_train=0` | 45 |
| `needs_manual_check=1` | 44 |

重要现象：

- `exclude_from_train=0` 的 45 行全部是 `ADM-like`。
- 这说明 `2.csv` 原本是偏向“保守清洗和人工审核”的表，而不是一张已经平衡好的二分类训练清单。
- 其中 `source_folder` 仍使用旧语义，比如 `caerulein solved`、`many colour`，而不是当前仓库里的英文目录结构。
- `full_path` 也是旧路径，不是当前仓库绝对路径。

本次训练对 `2.csv` 的实际使用方式是：

- 用它来补充 `lesion_id / magnification / image_type` 等信息。
- 用它把 `multichannel_unresolved` 中一部分可以明确使用的样本转成训练记录。
- 不盲目完全按照它的 `exclude_from_train` 逻辑替代当前整图训练逻辑。

评价如下：

- 优点：非常适合做 metadata 补充和后续清洗升级。
- 缺点：仍然带有旧路径、旧目录命名和“ADM 偏置”的问题。
- 结论：它是“很有用的元数据底稿”，但不是最终版训练 manifest。

### 4.4 `data/3.csv` 的作用与评价

`3.csv` 是字段词典，说明了 `2.csv` 每一列是什么意思。

评价如下：

- 优点：对后续整理 metadata 很重要，能避免字段含义混乱。
- 缺点：本身不包含训练样本，只是说明书。
- 结论：建议长期保留，并在 `2.csv` 更新时同步维护。

### 4.5 `KC/*.json` ROI 标注文件的作用与评价

当前 `KC` 目录中共有 5 个 ROI 标注文件：

- `7.json`
- `8.json`
- `10.json`
- `11.json`
- `12.json`

它们的共同特征：

- 格式是 LabelMe 风格 JSON。
- `imagePath` 与对应 tif 一一对应。
- `shape_type` 全部为 `polygon`。
- 标签全部为 `panIN`。
- 总 ROI 数为 28。

分文件统计如下：

| 文件 | ROI 数 | 标签 | 评价 |
| --- | ---: | --- | --- |
| `7.json` | 11 | `panIN` | ROI 数最多，信息量大，但单图内病灶较多，后续最需要人工再复核一次边界一致性 |
| `8.json` | 3 | `panIN` | ROI 数不多，但每个 ROI 面积都较大，质量较好 |
| `10.json` | 3 | `panIN` | 结构清晰，适合作为 ROI crop 来源 |
| `11.json` | 4 | `panIN` | 中等复杂度，标注质量较稳 |
| `12.json` | 7 | `panIN` | ROI 较多，但其中存在一对高度重叠的重复区域，代码中已做近重复抑制 |

这些 JSON 的作用不是直接喂给当前整图分类器，而是：

- 把 `PanIN` 病灶多边形转成可训练的 `ROI crop`
- 让模型学习到更聚焦的病灶形态
- 为下一步从整图分类过渡到 ROI/检测/分割任务打基础

评价如下：

- 优点：对 `PanIN` 类非常有帮助，本轮测试中所有 `ROI crop` 都预测正确。
- 缺点：目前只有 `PanIN ROI`，还没有 `ADM ROI` 对照。
- 结论：这是本轮性能提升最有价值的新资产之一。

## 5. 本次训练实际使用了哪些数据

本次运行最终一共构造了 106 条训练记录。

### 5.1 按标签统计

| 标签 | 数量 |
| --- | ---: |
| ADM | 56 |
| PanIN | 50 |

### 5.2 按样本类型统计

| 样本类型 | 数量 |
| --- | ---: |
| whole_image | 79 |
| roi_crop | 27 |

### 5.3 训练集组成

来自 `train_manifest.csv` 的统计如下：

| 维度 | 结果 |
| --- | --- |
| 总样本 | 66 |
| 标签分布 | ADM 35, PanIN 31 |
| 样本类型 | whole_image 50, roi_crop 16 |
| 来源目录 | `caerulein_adm=9`, `KC=26`, `KPC=5`, `multichannel_unresolved=26` |

对训练集的评价：

- 类别已接近平衡，优于早期简单 folder scan。
- `ROI crop` 已进入训练集，说明模型不是只看整图。
- `multichannel_unresolved` 中有 26 条样本被成功利用，这是对旧方案的重要扩充。

### 5.4 测试集组成

来自 `test_manifest.csv` 的统计如下：

| 维度 | 结果 |
| --- | --- |
| 总样本 | 40 |
| 标签分布 | ADM 21, PanIN 19 |
| 样本类型 | whole_image 29, roi_crop 11 |
| 来源目录 | `caerulein_adm=3`, `multichannel_adm=9`, `multichannel_kc_adm=9`, `KC=16`, `KPC=3` |

对测试集的评价：

- 测试集包含了 `multichannel_adm` 和 `multichannel_kc_adm` 的整组样本，挑战性更高。
- 测试集同时包含 `PanIN ROI crop` 和多通道 `ADM`，因此更像一个“混合泛化测试”而不是简单随机抽样。
- 这让本轮 `0.925 accuracy` 更有参考价值。

## 6. 结果文件逐项分析

### 6.1 `metrics.json`

内容：

- 只包含最终指标和混淆矩阵统计。

评价：

- 优点：最直接、最适合快速读结论。
- 结果：整体很好，尤其是 `sensitivity=1.0`，说明本轮没有漏判 `PanIN`。
- 风险：`specificity=0.857` 表示模型仍会把一部分 `ADM` 判成 `PanIN`。

结论：

- 这是一个明显好于“纯拍脑袋 baseline”的结果。
- 但还没有达到可以放心用于病理辅助判断的稳定程度。

### 6.2 `experiment_summary.json`

内容：

- 训练参数
- 设备信息
- 数据总量
- 训练集/测试集分布
- 历史曲线
- 最终指标

本次记录的关键信息：

- 训练时长约 103.19 秒
- 设备：`cuda`
- GPU 数：1
- 记录总数：106
- 训练集：66
- 测试集：40

评价：

- 这是本轮最完整的实验归档文件。
- 它证明本次不是随意跑了一次，而是完整保留了实验上下文。
- 从 `source_bucket_counts` 能清楚看出训练集和测试集的数据域差异。

结论：

- 这是最值得长期保留的实验说明文件之一。

### 6.3 `history.json`

内容：

- 24 个 epoch 的 `train_loss`
- 24 个 epoch 的 `train_accuracy`
- 每轮学习率

训练曲线特点：

- 第 1 轮：`train_accuracy=0.833`
- 第 4 轮：`train_accuracy=0.985`
- 第 9 轮：`train_accuracy=1.000`
- 后期大多维持在 `0.95~1.00`
- 学习率随 cosine 策略逐步下降到接近 `8.6e-07`

评价：

- 优点：收敛速度快，训练过程稳定，没有明显崩掉。
- 风险：后期训练准确率长期接近 1，说明模型已经具备明显的过拟合能力。
- 但由于测试集仍保持 0.925，说明当前还不是完全无效的过拟合。

结论：

- 训练策略有效，但下一步如果继续堆 epoch，收益可能不大。

### 6.4 `train_manifest.csv`

内容：

- 每条训练记录的来源图像
- 标签
- 来源目录
- 组 ID
- 样本类型
- 是否来自 ROI

评价：

- 优点：可以追踪“模型究竟看过哪些样本”。
- 训练集里已经混入 `roi_crop` 与 `metadata_resolved`，说明改进方案真正把新信息利用起来了。
- `group_count=27`，比单纯图像级切分更合理。

结论：

- 这是检查训练样本是否脏、是否偏、是否重复的核心文件。

### 6.5 `test_manifest.csv`

内容：

- 每条测试记录的来源
- `group_id`
- `sample_type`
- `crop_box`
- `label_source`

评价：

- 优点：可以准确定位错例属于哪一组、哪一通道、哪一类样本。
- 从本文件可见，测试集包含两个很重的 ADM 多通道组：
  - `LESION_MC_001` 共 12 条
  - `LESION_MC_002` 共 6 条
- 同时还包含 `KC/7.tif` 的 11 个 PanIN ROI crop。

结论：

- 这个测试集并不轻松，尤其对 ADM 多通道样本很有压力。

### 6.6 `predictions.json`

内容：

- 每条测试样本的预测标签
- 预测分数
- 是否预测正确

本次总预测数：

- 40 条
- 正确 37 条
- 错误 3 条

三条错例如下：

| 错例 | 真值 | 预测 | 分数 | 说明 |
| --- | --- | --- | ---: | --- |
| `data/multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif` | ADM | PanIN | 0.5088 | 边界最模糊的一条，接近阈值 |
| `data/multichannel_kc_adm/KC-20x-merge.tif` | ADM | PanIN | 0.8186 | merge 图上 PanIN 倾向明显 |
| `data/multichannel_kc_adm/KC-40x-merge.tif` | ADM | PanIN | 0.9242 | 高置信误判，说明该类样本最具挑战性 |

进一步统计显示：

- 所有错例都来自 `multichannel_kc_adm`
- 所有错例都是 `whole_image`
- 所有 `roi_crop` 全部预测正确
- `single` 通道样本 0 错
- `merge` 通道样本错 2 条
- `ck19` 通道样本错 1 条
- `amylase` 通道样本 0 错

评价：

- 这是本轮最有诊断价值的结果文件。
- 它明确指出当前瓶颈不是 `PanIN ROI`，而是 `KC-adm` 风格 ADM 的多通道整图。
- 也说明 ROI 标注方向是正确的，因为 ROI 样本在测试中表现非常稳定。

结论：

- 下一步最值得优化的是 `ADM ROI` 和 `multichannel_kc_adm` 的组织方式，而不是盲目扩大 backbone。

## 7. 本轮结果的综合评价

### 7.1 优点

- 真正把原始目录、CSV 元数据和 ROI 标注整合到了同一条训练管线。
- ROI crop 对 `PanIN` 的帮助很明确。
- 整体指标已经达到较强水平，AUC 接近 0.98。
- 没有出现 `PanIN` 漏检，说明模型对阳性识别比较敏感。

### 7.2 主要问题

- `ADM` 的特异度还不够高。
- 所有错误都集中在 `multichannel_kc_adm`，说明该类样本语义最混杂。
- `2.csv` 仍然不是一张最终版训练 metadata。
- 目前 ROI 只有 `PanIN`，没有 `ADM` 对照。
- 当前分组仍然主要依赖文件名和工作假设，没有鼠级或切片级 metadata。

### 7.3 对本轮结果可信度的判断

本轮结果比旧版 baseline 明显更可信，原因有三：

- 引入了 group-aware split
- 引入了 ROI crop
- 测试集包含更难的多通道 ADM 组

但它依然不是最终科研结论，因为：

- `2.csv` 还在半清洗状态
- ROI 标注只覆盖 `PanIN`
- 数据量仍然偏小

## 8. 对每一类文件的最终评价结论

### 8.1 原始图像目录

- 可用于当前训练。
- 但多通道组还需要更细粒度组织。

### 8.2 `1.csv`

- 适合做首页和总览。
- 不适合直接进训练代码。

### 8.3 `2.csv`

- 很有价值。
- 但必须继续修正路径、目录命名和训练掩码逻辑。

### 8.4 `3.csv`

- 是字段词典。
- 适合作为 metadata 文档长期维护。

### 8.5 `KC/*.json`

- 是本轮改进最关键的新信息来源之一。
- 标注质量总体可用。
- 后续必须补 `ADM ROI`，否则 ROI 训练路线天然偏向 `PanIN`。

### 8.6 `metrics.json`

- 最适合快速看成绩。
- 结论是“结果很好，但不是完美”。

### 8.7 `experiment_summary.json`

- 最完整的实验归档。
- 强烈建议每次训练都保留。

### 8.8 `history.json`

- 适合看收敛与过拟合趋势。
- 本轮显示训练已充分收敛。

### 8.9 `train_manifest.csv`

- 适合查训练样本组成。
- 是检查数据脏点的重要依据。

### 8.10 `test_manifest.csv`

- 适合定位测试集压力来源。
- 也是解释错例的关键清单。

### 8.11 `predictions.json`

- 本轮最值得后续继续深挖的结果文件。
- 所有错例都能从这里直接定位。

## 9. 下一步建议

1. 优先为 `ADM` 补 ROI 标注，至少补 `caerulein_adm` 和 `multichannel_kc_adm`。
2. 优先复核 `multichannel_kc_adm` 的 `lesion_id` 组织关系，尤其是 `KC-merge` 与 `ck-19-KC-adm` 的对应关系。
3. 把 `2.csv` 升级成真正可执行的 metadata v2，而不是只作为历史整理底稿。
4. 下一轮重点分析 `predictions.json` 里的 3 个 ADM 假阳性，而不是单纯换更大的网络。
5. 如果后续 ROI 数据丰富，可以进一步尝试：
   - ROI-only 分类
   - Whole-image + ROI 双分支融合
   - 基于 ROI 的检测或实例分割路线

## 10. 复现实验的关键文件路径

- 项目根目录：`D:\srpgpt\pancreas-vision-analysis`
- 训练脚本：`D:\srpgpt\pancreas-vision-analysis\src\train_improved.py`
- 结果目录：`D:\srpgpt\pancreas-vision-analysis\artifacts\improved_hybrid_pycharm`
- 本文档：`D:\srpgpt\result.md`

