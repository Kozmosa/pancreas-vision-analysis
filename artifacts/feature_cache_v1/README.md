# Feature Cache v1

UNI2-h feature extraction results for 80 instances.

## Structure

- `feature_index.csv` — Mapping of instance_id to feature files
- `features/*.pt` — Cached feature tensors (1536-dim)
- `summary.json` — Extraction statistics

## Usage

Load features for training:

```python
import torch
import pandas as pd

index = pd.read_csv("feature_index.csv")
row = index[index["instance_id"] == "INSTANCE_0001"].iloc[0]
data = torch.load(row["feature_path"])
features = data["features"]  # shape: (1536,)
```

## Extraction Details

- **Extractor**: UNI2-h (Mahmood Lab)
- **Feature dim**: 1536
- **Global features**: 80 (1 per instance)
- **Patch features**: 212 (local patches from whole images)