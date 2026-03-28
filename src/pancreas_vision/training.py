from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from pancreas_vision.data import MicroscopyDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class EvaluationMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass
class PredictionRecord:
    image_path: str
    true_label: int
    predicted_label: int
    positive_score: float
    correct: bool
    bag_id: str | None = None
    source_bucket: str | None = None
    label_name: str | None = None
    sample_type: str | None = None


@dataclass
class TrainingHistory:
    epoch: int
    train_loss: float
    train_accuracy: float
    learning_rate: float


@dataclass
class BagPredictionRecord:
    bag_id: str
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    positive_score: float
    correct: bool
    source_buckets: str
    instance_count: int
    dominant_channel: str
    dominant_magnification: str


@dataclass
class SourceBucketErrorRecord:
    source_bucket: str
    bag_count: int
    error_count: int
    false_positive_count: int
    false_negative_count: int
    accuracy: float


def build_transforms(
    image_size: int = 224,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    return build_model_with_backbone(
        backbone_name="resnet18",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=0.0,
    )


def build_model_with_backbone(
    backbone_name: str,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        classifier_attr = "fc"
        classifier_in_features = model.fc.in_features
    elif backbone_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        classifier_attr = "fc"
        classifier_in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith(f"{classifier_attr}."):
                parameter.requires_grad = False

    classifier_layers: list[nn.Module] = []
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(classifier_in_features, num_classes))
    setattr(model, classifier_attr, nn.Sequential(*classifier_layers))
    return model


def create_dataloaders(
    train_records,
    test_records,
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(image_size=image_size)
    train_dataset = MicroscopyDataset(train_records, transform=train_transform)
    test_dataset = MicroscopyDataset(test_records, transform=eval_transform)
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        label_counts = Counter(record.label_index for record in train_records)
        sample_weights = [
            1.0 / label_counts[record.label_index] for record in train_records
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    output_path: Path | None = None,
) -> list[TrainingHistory]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )
    model.to(device)
    history: list[TrainingHistory] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        total_correct = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        history.append(
            TrainingHistory(
                epoch=epoch,
                train_loss=running_loss / max(total_examples, 1),
                train_accuracy=total_correct / max(total_examples, 1),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        scheduler.step()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(item) for item in history], indent=2),
            encoding="utf-8",
        )
    return history


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[EvaluationMetrics, list[PredictionRecord]]:
    model.eval()
    predicted_labels = []
    true_labels = []
    positive_scores = []
    image_paths = []

    for images, labels, paths in data_loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

        predicted_labels.extend(predicted.cpu().numpy().tolist())
        true_labels.extend(labels.numpy().tolist())
        positive_scores.extend(probabilities[:, 1].cpu().numpy().tolist())
        image_paths.extend(paths)

    tn, fp, fn, tp = confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    ).ravel()
    accuracy = accuracy_score(true_labels, predicted_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    roc_auc = (
        roc_auc_score(true_labels, positive_scores)
        if len(set(true_labels)) > 1
        else float("nan")
    )

    metrics = EvaluationMetrics(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        roc_auc=float(roc_auc),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
    predictions = [
        PredictionRecord(
            image_path=image_path,
            true_label=true_label,
            predicted_label=predicted_label,
            positive_score=float(positive_score),
            correct=true_label == predicted_label,
        )
        for image_path, true_label, predicted_label, positive_score in zip(
            image_paths, true_labels, predicted_labels, positive_scores
        )
    ]
    return metrics, predictions


def aggregate_predictions_to_bags(
    predictions: list[PredictionRecord],
    record_lookup: dict[str, object],
) -> tuple[EvaluationMetrics, list[BagPredictionRecord], list[SourceBucketErrorRecord]]:
    bag_groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        record = record_lookup[prediction.image_path]
        prediction.bag_id = getattr(record, "lesion_id")
        prediction.source_bucket = getattr(record, "source_bucket")
        prediction.label_name = getattr(record, "label_name")
        prediction.sample_type = getattr(record, "sample_type")
        bag_groups[prediction.bag_id].append(prediction)

    bag_predictions: list[BagPredictionRecord] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    positive_scores: list[float] = []

    for bag_id, bag_items in sorted(bag_groups.items()):
        first_record = record_lookup[bag_items[0].image_path]
        mean_positive_score = float(
            sum(item.positive_score for item in bag_items) / len(bag_items)
        )
        predicted_label = 1 if mean_positive_score >= 0.5 else 0
        true_label = int(getattr(first_record, "label_index"))
        channel_counts = Counter(
            getattr(record_lookup[item.image_path], "channel_name") for item in bag_items
        )
        magnification_counts = Counter(
            getattr(record_lookup[item.image_path], "magnification") for item in bag_items
        )
        source_buckets = sorted(
            {getattr(record_lookup[item.image_path], "source_bucket") for item in bag_items}
        )
        bag_predictions.append(
            BagPredictionRecord(
                bag_id=bag_id,
                true_label=true_label,
                true_label_name=getattr(first_record, "label_name"),
                predicted_label=predicted_label,
                predicted_label_name="PanIN" if predicted_label == 1 else "ADM",
                positive_score=mean_positive_score,
                correct=predicted_label == true_label,
                source_buckets="|".join(source_buckets),
                instance_count=len(bag_items),
                dominant_channel=channel_counts.most_common(1)[0][0],
                dominant_magnification=magnification_counts.most_common(1)[0][0],
            )
        )
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        positive_scores.append(mean_positive_score)

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    bag_metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        sensitivity=float(tp / (tp + fn) if (tp + fn) else 0.0),
        specificity=float(tn / (tn + fp) if (tn + fp) else 0.0),
        roc_auc=float(
            roc_auc_score(true_labels, positive_scores)
            if len(set(true_labels)) > 1
            else float("nan")
        ),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )

    bucket_error_rows: list[SourceBucketErrorRecord] = []
    bucket_groups: dict[str, list[BagPredictionRecord]] = defaultdict(list)
    for row in bag_predictions:
        for source_bucket in row.source_buckets.split("|"):
            bucket_groups[source_bucket].append(row)
    for source_bucket, rows in sorted(bucket_groups.items()):
        fp_count = sum(
            row.true_label == 0 and row.predicted_label == 1 for row in rows
        )
        fn_count = sum(
            row.true_label == 1 and row.predicted_label == 0 for row in rows
        )
        error_count = sum(not row.correct for row in rows)
        bucket_error_rows.append(
            SourceBucketErrorRecord(
                source_bucket=source_bucket,
                bag_count=len(rows),
                error_count=error_count,
                false_positive_count=fp_count,
                false_negative_count=fn_count,
                accuracy=float(sum(row.correct for row in rows) / len(rows)),
            )
        )
    return bag_metrics, bag_predictions, bucket_error_rows


def save_metrics(metrics: EvaluationMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def save_experiment_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_predictions(predictions: list[PredictionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in predictions], indent=2),
        encoding="utf-8",
    )


def save_bag_predictions(
    predictions: list[BagPredictionRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in predictions], indent=2),
        encoding="utf-8",
    )


def save_source_bucket_errors(
    rows: list[SourceBucketErrorRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in rows], indent=2),
        encoding="utf-8",
    )


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
