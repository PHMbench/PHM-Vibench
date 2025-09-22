"""Utilities for building common evaluation metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import List, Any


class ContrastiveAccuracy(torchmetrics.Metric):
    """
    Contrastive learning accuracy metric.

    Measures the accuracy of contrastive learning by checking if the
    most similar positive sample is correctly identified for each anchor.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> None:
        """
        Update the metric with new predictions.

        Args:
            z_anchor: [B, D] anchor representations
            z_positive: [B, D] positive representations
        """
        with torch.no_grad():
            # L2 normalize representations
            z_anchor = F.normalize(z_anchor, dim=1)
            z_positive = F.normalize(z_positive, dim=1)

            # Compute similarity matrix
            similarity_matrix = torch.mm(z_anchor, z_positive.t())

            # Find the most similar positive for each anchor
            _, predicted = torch.max(similarity_matrix, dim=1)

            # Correct matches are on the diagonal
            correct_matches = torch.arange(
                similarity_matrix.shape[0],
                device=predicted.device
            )

            # Count correct predictions
            self.correct += (predicted == correct_matches).float().sum()
            self.total += z_anchor.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy."""
        return self.correct / self.total

    def reset(self) -> None:
        """Reset the metric state."""
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)


def get_metrics(metric_names: List[str], metadata: Any) -> nn.ModuleDict:
    """Create metrics according to ``metric_names`` for each dataset.

    Parameters
    ----------
    metric_names : list of str
        Metric identifiers such as ``acc`` or ``f1``.
    metadata : Any
        Dataset metadata used to infer the number of classes.
    """
    metric_classes = {
        "acc": torchmetrics.Accuracy,
        "f1": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auroc": torchmetrics.AUROC,
        "contrastive_acc": ContrastiveAccuracy,
    }

    metrics = nn.ModuleDict()

    unique_ids = set()
    max_labels = {}
    for item_id, item_data in metadata.items():
        if "Name" in item_data:
            data_id = item_data["Name"]
            unique_ids.add(data_id)
            if "Label" in item_data:
                current_label = item_data["Label"]
                if data_id not in max_labels or current_label > max_labels[data_id]:
                    max_labels[data_id] = current_label

    for data_name, n_class in max_labels.items():
        if n_class is None:
            raise ValueError(f"数据集 '{data_name}' 的配置缺少 'n_classes'")

        task_type = "multiclass" if n_class >= 2 else "binary" # fix
        data_metrics = nn.ModuleDict()
        for stage in ["train", "val", "test"]:
            for metric_name in metric_names:
                key = metric_name.lower()
                if key in metric_classes:
                    if key == "contrastive_acc":
                        # ContrastiveAccuracy doesn't need num_classes
                        data_metrics[f"{stage}_{key}"] = metric_classes[key]()
                    else:
                        # Standard torchmetrics
                        data_metrics[f"{stage}_{key}"] = metric_classes[key](
                            task=task_type,
                            num_classes=int(n_class) + 1,
                        )
                else:
                    print(f"警告: 不支持的指标类型 '{metric_name}'，已跳过。")
        metrics[data_name] = data_metrics

    return metrics


if __name__ == "__main__":
    # Minimal demonstration
    dummy_meta = {1: {"Name": "ds", "Label": 2}}
    m = get_metrics(["acc", "f1"], dummy_meta)
    print("Built metrics:", list(m["ds"].keys()))
