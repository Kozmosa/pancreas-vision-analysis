# CLAM Hard-case Experiment V1

## Configuration

- **Split**: `hard_case_split.csv` - Forces multichannel_kc_adm bags into test set
- **Model**: CLAM Single Branch (hidden=256, attention=128)
- **Training**: 20 epochs, LR=1e-4, CPU

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 81.82% |
| Sensitivity | 100% |
| Specificity | 60% |
| ROC-AUC | 83.33% |
| True Positive | 6 |
| True Negative | 3 |
| False Positive | 2 |
| False Negative | 0 |

## Key Finding

**Both multichannel_kc_adm ADM bags are misclassified as PanIN:**

| bag_id | true_label | predicted | positive_score | instance_count |
|--------|------------|-----------|----------------|----------------|
| LESION_MC_001 | ADM | PanIN | 0.866 | 60 |
| LESION_MC_002 | ADM | PanIN | 0.995 | 30 |

This confirms the core problem: **Bag-level attention aggregation does not resolve the multichannel_kc_adm false positive issue.**

The high positive scores (0.87, 0.99) indicate the model is confidently wrong about these ADM bags.

## Comparison

| Experiment | Test Split | Accuracy | MC Bags in Test | MC Errors |
|------------|------------|----------|-----------------|-----------|
| exp_clam_v1 | main_split | 100% | 0 | N/A |
| exp_clam_hard_case_v1 | hard_case_split | 81.82% | 2 | 2 FP |

The 100% accuracy in exp_clam_v1 was misleading because MC bags were never evaluated.

## Files

- `bag_metrics.json` - Evaluation metrics
- `bag_predictions.csv` - Per-bag predictions
- `attention_summary.json` - Attention weights per bag
- `training_history.json` - Loss per epoch
- `model.pt` - Model weights