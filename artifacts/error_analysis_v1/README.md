# Error Analysis V1

## Overview

This directory contains error analysis outputs from the CLAM model evaluation with the hard-case split.

## Experiments

### exp_clam_hard_case_v1

**Split**: `hard_case_split.csv` - Forces multichannel_kc_adm bags into test set

**Key Results**:
- Accuracy: 81.82%
- Sensitivity: 100%
- Specificity: 60%
- **2 False Positives from multichannel_kc_adm bags**

This experiment confirms the core problem: **multichannel_kc_adm ADM bags are misclassified as PanIN** even with bag-level attention aggregation.

## Files

### error_by_source_bucket.csv

Error breakdown by source bucket:

| source_bucket | bag_count | error_count | false_positive_count | accuracy |
|---------------|-----------|-------------|---------------------|----------|
| multichannel_kc_adm | 2 | 2 | 2 | 0.0 |
| multichannel_adm | 1 | 1 | 1 | 0.0 |
| caerulein_adm | 3 | 0 | 0 | 1.0 |
| KC | 5 | 0 | 0 | 1.0 |
| KPC | 1 | 0 | 0 | 1.0 |

### hard_case_analysis.json

Per-bag analysis of hard-case bags (multichannel_kc_adm):

| bag_id | predicted_correctly | error_type | positive_score | top_attention_channels |
|--------|---------------------|------------|----------------|------------------------|
| LESION_MC_001 | False | false_positive | 0.866 | ck19 (dominant) |
| LESION_MC_002 | False | false_positive | 0.995 | merge, amylase |

### gan_patch_candidates.csv

Empty - no candidates extracted because:
1. With hard_case_split, MC bags are in TEST set
2. GAN training requires candidates from TRAIN set only

To extract GAN candidates, run analysis on the original split where MC bags are in TRAIN.

## Key Findings

1. **Bag-level aggregation does NOT solve the multichannel_kc_adm problem**
   - Both MC bags (LESION_MC_001, LESION_MC_002) are misclassified as PanIN
   - High confidence (0.87, 0.99) despite being ADM

2. **ck19 channel appears to drive misclassification**
   - LESION_MC_001's top attention instances all have ck19 channel
   - ck19-KC-adm-IF-40x was the known FP from single-image experiments

3. **Attention patterns reveal focus on ck19**
   - Model focuses attention on ck19 channel instances
   - These instances may have PanIN-like morphological features

## Recommendations

1. **GAN Augmentation**: Use training set ADM instances from multichannel_kc_adm bags
   - Priority: ck19 channel, 40x magnification
   - Focus on high-attention instances

2. **Feature Engineering**: Consider separate channel embeddings or channel-specific attention

3. **Data Collection**: Need more ADM examples from KC background to balance training

## Comparison with Original Split

| Metric | Original Split (MC in train) | Hard-case Split (MC in test) |
|--------|------------------------------|------------------------------|
| Accuracy | 100% | 81.82% |
| Sensitivity | 100% | 100% |
| Specificity | 100% | 60% |
| MC False Positives | N/A (not in test) | 2 |

The original split achieved perfect metrics because MC bags were never evaluated in test set.