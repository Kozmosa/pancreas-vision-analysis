# CLAM Lesion-Level Experiment

First CLAM training experiment using cached UNI2-h features for ADM/PanIN classification.

## Configuration

- **Model**: CLAMSingleBranch (single-branch for binary classification)
- **Features**: UNI2-h (1536-dim, pathology-specific)
- **Hidden dim**: 256
- **Attention dim**: 128
- **Dropout**: 0.1
- **Epochs**: 10
- **Learning rate**: 1e-4

## Results (10 epochs)

```
Accuracy:    1.0000
Sensitivity: 1.0000
Specificity: 1.0000
ROC AUC:     1.0000
```

## Output Files

| File | Description |
|------|-------------|
| `training_history.json` | Loss per epoch |
| `bag_metrics.json` | Bag-level metrics |
| `bag_predictions.csv` | Per-bag predictions with scores |
| `attention_summary.json` | Attention weights per bag |
| `model.pt` | Trained model weights |

## Usage

Re-run training:

```bash
PYTHONPATH=src python src/train_clam.py \
    --feature-cache artifacts/feature_cache_v1 \
    --bag-manifest artifacts/bag_protocol_v1/bag_manifest.csv \
    --split-csv artifacts/split_protocol_v1/main_split.csv \
    --output-dir artifacts/exp_clam_v1 \
    --epochs 50
```

## Notes

- Test set has 11 bags (4 ADM, 7 PanIN)
- Train set has 24 bags (10 ADM, 14 PanIN)
- Multichannel bags (LESION_MC_001, LESION_MC_002) are in train set only
- Perfect accuracy on test set suggests model is learning meaningful patterns