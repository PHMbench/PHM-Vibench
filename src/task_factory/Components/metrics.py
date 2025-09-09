"""Utilities for building common evaluation metrics."""

import torch.nn as nn
import torchmetrics
from typing import List, Any


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
        # Classification metrics
        "acc": torchmetrics.Accuracy,
        "f1": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auroc": torchmetrics.AUROC,
        # Regression metrics
        "mse": torchmetrics.MeanSquaredError,
        "mae": torchmetrics.MeanAbsoluteError,
        "r2": torchmetrics.R2Score,
        "mape": torchmetrics.MeanAbsolutePercentageError,
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
                    # Classification metrics need task and num_classes
                    if key in ["acc", "f1", "precision", "recall", "auroc"]:
                        data_metrics[f"{stage}_{key}"] = metric_classes[key](
                            task=task_type,
                            num_classes=int(n_class) + 1,
                        )
                    # Regression metrics don't need these parameters
                    else:
                        data_metrics[f"{stage}_{key}"] = metric_classes[key]()
                else:
                    print(f"警告: 不支持的指标类型 '{metric_name}'，已跳过。")
        metrics[data_name] = data_metrics

    return metrics


if __name__ == "__main__":
    # Minimal demonstration
    dummy_meta = {1: {"Name": "ds", "Label": 2}}
    m = get_metrics(["acc", "f1"], dummy_meta)
    print("Built metrics:", list(m["ds"].keys()))
