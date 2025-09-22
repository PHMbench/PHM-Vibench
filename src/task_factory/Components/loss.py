"""Common loss utilities used across PHM-Vibench tasks."""

import torch
import torch.nn as nn
from .metric_loss import MatchingLoss
from .prediction_loss import *
from .contrastive_loss import InfoNCELoss, SimCLRLoss, SupConLoss

def get_loss_fn(loss_name: str) -> nn.Module:
    """Return a loss module according to ``loss_name``.

    Parameters
    ----------
    loss_name: str
        Key identifying the loss type. Supported values are ``CE``, ``MSE``,
        ``MAE``, ``BCE`` and ``NLL``.
    """
    # loss_name = args.loss
    loss_mapping = {
        "CE": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "BCE": nn.BCEWithLogitsLoss(),
        "NLL": nn.NLLLoss(),
        "MATCHING": MatchingLoss,
        "SIGNAL_MASK_LOSS": Signal_mask_Loss,  # TODO Time Series Prediction
        "INFONCE": InfoNCELoss(),
        "SIMCLR": SimCLRLoss(),
        "SUPCON": SupConLoss(),
    }

    key = loss_name.upper()
    if key not in loss_mapping:
        raise ValueError(
            f"不支持的损失函数类型: {loss_name}，可选类型: {list(loss_mapping.keys())}"
        )
    if key == "Matching":
        return loss_mapping[key] # (args)

    return loss_mapping[key]


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(4, 5)
    target = torch.randint(0, 5, (4,))
    loss_fn = get_loss_fn("CE")
    print("Test CE loss:", loss_fn(pred, target))
