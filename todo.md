# 双人协作详细 ToDo：

## Summary

- 主线目标固定为：在当前 `ADM vs PanIN` 任务上，把单图分类升级为 `lesion-level bag + UNI/DINOv2 特征 + CLAM`，并把 `StyleGAN3` 作为后续增强支线。
- 阶段顺序固定为：`bag 数据协议 -> UNI 特征缓存 -> CLAM 主实验 -> 错误分析 -> StyleGAN3 消融`。后一步不得阻塞前一步。

## Progress

| 阶段 | 状态 | 产出 |
|------|------|------|
| B-1: Bag 数据协议 | ✅ 完成 | `artifacts/bag_protocol_v1/` |
| B-2: Split 协议 | ✅ 完成 | `artifacts/split_protocol_v1/` |
| B-3: 特征缓存 | ✅ 完成 | `artifacts/feature_cache_v1/` (292 features, UNI2-h) |
| B-4: CLAM 训练 | ✅ 完成 | `src/pancreas_vision/models/clam.py`, `src/train_clam.py` |
| B-5: 错误分析 | ✅ 完成 | `src/pancreas_vision/analysis/`, `artifacts/error_analysis_v1/` |
| B-6: StyleGAN3 增强支线 | ✅ 完成 | `src/pancreas_vision/gan/`, `artifacts/gan_augmented_v1/` |

## A 同学任务：

### A-1. 冻结病灶分组规则
1. 目标：确认当前是否可以直接把 `lesion_id` 当成 `bag_id`。做法：只看 B 先生成的 bag 草表，不自己整理全量数据，重点抽查 `KC`、`KPC`、`multichannel_adm`、`multichannel_kc_adm` 这四类。产出：一句明确结论，“可直接使用 / 哪些前缀例外”。
2. 目标：确认 `10x/20x/40x` 是否仍按同一病灶处理。做法：只审查明显高风险组合，不逐文件全看，优先看当前错误最集中的 `multichannel_kc_adm`。产出：一个简短规则备注，说明哪些倍率组允许并入同一 bag。
3. 目标：确认 `merge/amylase/ck-19` 是否统一进入同一 bag。做法：按老师已确认口径审核，不重新定义任务。产出：一个通道并入规则说明。

### A-2. 审核少数歧义样本
4. 目标：处理机器无法可靠判断的样本，不做全量清洗。做法：让 B 先自动列出“单视图 bag、字段缺失 bag、名称不一致 bag、source_bucket 异常 bag”，A 只在这个列表里圈定真正需要备注的条目。产出：`歧义样本备注表`，行数尽量控制在少量样本。
5. 目标：确认 ROI 是否全部保留。做法：默认全部保留，只有在发现明显重复、错标或无意义 crop 时才标记排除。产出：一个 `ROI 保留/排除` 小表。
6. 目标：确认 hard-case 范围。做法：优先审查 `multichannel_kc_adm` 的 ADM 样本和当前 3 个误判来源样本，决定哪些可以作为 GAN 增强源。产出：`hard-case 白名单`。

### A-3. 审核中期与最终结果
7. 目标：检查第一版 `UNI+CLAM` 是否真的在解决核心问题。做法：看 B 汇总的错误分析，不自己跑实验，重点看 `multichannel_kc_adm` 假阳性有没有下降。产出：一段中期审核意见。
8. 目标：判断 GAN 是否值得保留。做法：只比较 `UNI+CLAM` 和 `UNI+CLAM+StyleGAN3` 的假阳性、敏感度和错误分布。产出：一句结论，“保留 / 放弃 GAN 增强”。
9. 目标：给最终结论背书。

### A-4. 实现 StyleGAN3 增强支线

 10.目标：让 GAN 只服务主任务，不引入新任务。做法：只用当前 `ADM vs PanIN` hard-case patch 训练 GAN，不做 `TERT` 相关实现。产出：GAN 训练数据集。

 11.目标：控制 synthetic 数据风险。做法：每个 hard-case bag 最多注入 `2` 个 synthetic instances，且只允许进入训练集，不进入验证或测试。产出：增强版训     练 manifest。

 12.目标：避免 GAN 记忆训练集。做法：做最近邻相似度检查或简单重合度检查，未通过的 synthetic patch 直接丢弃。产出：`accepted_synthetic_instances.csv`。

 13.目标：验证 GAN 是否真有帮助。做法：只跑一个清晰对照实验 `uni_clam_stylegan3`，不同时改别的超参。产出：GAN 消融结果。

### A-5. 汇总结果与交付审核材料

14.目标：把所有结果整理成 可审核、老师可汇报的形式。做法：输出一张总对比表，至少包含 `accuracy`、`sensitivity`、`specificity`、`roc_auc`、`multichannel_kc_adm 假阳性数`。产出：最终对比表。

15.目标：突出真正有决策价值的信息。做法：单独做一页“错误分布变化”，说明从 `improved_hybrid` 到 `UNI+CLAM` 再到 `GAN` 的变化。产出：错误分析页。

## B 同学任务：

### B-1. 搭建 bag 数据协议
1. 目标：把当前单图样本改造成病灶级样本单位。做法：以 `lesion_id` 为主键生成 `bag_manifest`，把同一病灶的多倍率、多通道、ROI crop 都归到一个 bag。产出：`bag_manifest.csv`。
2. 目标：让后续特征提取和 MIL 训练都能直接消费数据。做法：再生成 `instance_manifest.csv`，每行代表一个可训练实例，字段固定包括 `bag_id`、`image_path`、`label_name`、`source_bucket`、`magnification`、`channel_name`、`sample_type`、`is_roi`、`split_key`。产出：`instance_manifest.csv`。
3. 目标：先发现数据协议问题，不把脏问题带到模型里。做法：自动统计每个 bag 的视图数、倍率缺失、通道缺失、ROI 数量和来源分布。产出：一页数据质检摘要。
4. 目标：减少 A 的工作量。做法：自动产出“需要人工审核的候选列表”，只把最可疑的 bag 给 A。产出：`review_candidates.csv`。

### B-2. 固化 split 与评估协议
5. 目标：避免病灶级泄漏。做法：train/test 固定按 `lesion_id` 分组切分，默认沿用当前 `seed=42` 和 7:3 比例。产出：主 split 文件。
6. 目标：避免单次切分偶然性。做法：追加 grouped 5-fold CV 文件，fold 级别也不允许共享 `lesion_id`。产出：5 份 fold 索引或一个统一 fold 表。
7. 目标：让所有对照实验可直接复用同一评估框架。做法：规定每个实验都必须输出 `bag-level metrics`、`bag-level predictions`、`error by source_bucket`。产出：统一评估模板。

### B-3. 实现 UNI/DINOv2 特征缓存
8. 目标：先把强特征提取落地，再做弱监督。做法：对每个实例先统一转 `RGB`，保留现有 TIFF 读法，不改原始数据。产出：标准化后的特征提取入口。
9. 目标：兼顾全局与局部形态。做法：每张视图固定提取 `1` 个全局特征，外加最多 `4` 个局部 patch 特征；ROI crop 视图优先作为局部实例保留。产出：每个实例对应的 feature 文件。
10. 目标：让 CLAM 训练完全脱离原图 IO。做法：把所有 UNI 特征离线缓存，并建立 `feature_index` 记录 `instance_id -> feature_path` 的映射。产出：`feature cache + feature index`。
11. 目标：保证主线不被模型下载或权重问题卡住。做法：优先用 UNI；如果权重不可得，临时 fallback 到通用 DINOv2，但实验命名必须区分。产出：明确的 encoder 版本记录。

### B-4. 实现 lesion-level CLAM 主实验
12. 目标：把当前任务从单图分类升级为 bag-level 弱监督。做法：默认使用 `CLAM single-branch`，输入是离线 UNI 特征，不直接读像素。产出：可运行训练入口。
13. 目标：让模型知道倍率和通道信息。做法：在 UNI 特征之外追加小型 `magnification embedding` 和 `channel embedding`。产出：实例级融合输入。
14. 目标：保持第一版结构可控。做法：第一版不做复杂 transformer fusion，不做端到端解冻，只训 CLAM 头。产出：稳定可复现实验。
15. 目标：增强可解释性。做法：保存每个 bag 的 attention 排名，输出“最重要的倍率/通道/patch”。产出：`attention summary`。
16. 目标：验证主线是否有效。做法：先运行 `uni_clam_lesion_bag`，与当前 `improved_hybrid` 正面对照。产出：主结果表。

### B-5. 完成错误分析与 hard-case 管理
17. 目标：把当前核心痛点量化出来。做法：专门汇总 `multichannel_kc_adm` 的假阳性数量、错误置信度、被关注最多的通道。产出：`hard-case analysis`。
18. 目标：为 GAN 支线准备数据，而不是盲目生成。做法：从训练集 hard-case bag 中筛出 patch，优先保留边界模糊、容易误判的 ADM patch。产出：`gan_train_patch_list`。
19. 目标：减少 A 的人工筛选量。做法：自动给每个候选 patch 附上来源 bag、来源通道、来源倍率和误判背景说明，只把最终 shortlist 给 A。产出：`gan_review_shortlist`。

## 协作时间线

- 第 1 周前半：B 完成 `bag/instance manifest`、质检统计、候选审核表；A 做第一次规则确认。
- 第 1 周后半：B 完成 UNI 特征缓存和 split 文件
- 第 2 周前半：B 跑通 `UNI+CLAM` 第一版并输出错误分析；A 看重点 hard-case。
- 第 2 周后半：若 `UNI+CLAM` 已稳定，A 再做 StyleGAN3,审核增强源和最终结果。
- 任一阶段如果前一步未稳定，后一步自动顺延，尤其 GAN 不能反向阻塞主线。

## Acceptance Criteria

- `UNI+CLAM` 必须保持 `sensitivity >= 0.95`。
- `UNI+CLAM` 必须优先改善当前 `multichannel_kc_adm` 假阳性，或提升整体特异度。
- `StyleGAN3` 只有在不降低敏感度的前提下进一步改善 hard-case，才保留。
- 如果 GAN 无稳定收益，正式 v2 主方案直接定为 `UNI+CLAM lesion bag`。

## Assumptions

- 当前仍无 patient/mouse/mutation 标签，因此全部设计都以 `lesion-level` 为上限，不扩展到 `TERT` 预测。
