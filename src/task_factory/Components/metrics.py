import torch.nn as nn
import torchmetrics
from typing import Dict, List, Any

def get_metrics(metric_names: List[str], metadata: Any) -> nn.ModuleDict:
    """根据配置获取评估指标集合"""
    metric_classes = {
        "acc": torchmetrics.Accuracy,
        "f1": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall
        # 可以根据需要添加更多指标
    }

    metrics = nn.ModuleDict()

    # Extract unique data IDs and find maximum labels for each
    unique_data_ids = set()
    max_labels = {}

    # Iterate through metadata to find unique data_ids and their maximum labels
    for item_id, item_data in metadata.items():
        if 'Name' in item_data:
            data_id = item_data['Name']
            unique_data_ids.add(data_id)
            
            # Track maximum label for each data_id
            if 'Label' in item_data:
                current_label = item_data['Label']
                if data_id not in max_labels or current_label > max_labels[data_id]:
                    max_labels[data_id] = current_label

    print(f"Found {len(unique_data_ids)} unique data IDs")
    print(f"Maximum labels per data ID: {max_labels}")

    # Assuming data_config is derived from metadata
    data_config = metadata

    for data_name, n_class in max_labels.items():

        if n_class is None:
             raise ValueError(f"数据集 '{data_name}' 的配置缺少 'n_classes'")

        task_type = "multiclass" if n_class > 2 else "binary"

        data_metrics = nn.ModuleDict()
        for stage in ["train", "val", "test"]:
            for metric_name in metric_names:
                metric_key = metric_name.lower()
                if metric_key in metric_classes:
                    data_metrics[f"{stage}_{metric_key}"] = metric_classes[metric_key](
                        task=task_type,
                        num_classes= int(n_class) + 1,
                    )
                else:
                    print(f"警告: 不支持的指标类型 '{metric_name}'，已跳过。") # 或者抛出错误

        metrics[data_name] = data_metrics

    return metrics