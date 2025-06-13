"""Common loss utilities used across PHM-Vibench tasks."""

import torch
import torch.nn as nn


def get_loss_fn(loss_name: str) -> nn.Module:
    """Return a loss module according to ``loss_name``.

    Parameters
    ----------
    loss_name: str
        Key identifying the loss type. Supported values are ``CE``, ``MSE``,
        ``MAE``, ``BCE`` and ``NLL``.
    """
    loss_mapping = {
        "CE": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "BCE": nn.BCEWithLogitsLoss(),
        "NLL": nn.NLLLoss(),
    }

    key = loss_name.upper()
    if key not in loss_mapping:
        raise ValueError(
            f"不支持的损失函数类型: {loss_name}，可选类型: {list(loss_mapping.keys())}"
        )
    return loss_mapping[key]


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(4, 5)
    target = torch.randint(0, 5, (4,))
    loss_fn = get_loss_fn("CE")
    print("Test CE loss:", loss_fn(pred, target))
