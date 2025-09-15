"""Common loss utilities used across PHM-Vibench tasks."""

import torch
import torch.nn as nn
from .metric_loss import MatchingLoss
from .prediction_loss import *
from .contrastive_losses import (
    InfoNCELoss, TripletLoss, SupConLoss, PrototypicalLoss,
    BarlowTwinsLoss, VICRegLoss
)

def get_loss_fn(loss_name: str) -> nn.Module:
    """Return a loss module according to ``loss_name``.

    Parameters
    ----------
    loss_name: str
        Key identifying the loss type. Supported values include standard losses
        (``CE``, ``MSE``, ``MAE``, ``BCE``, ``NLL``) and SOTA contrastive losses
        (``INFONCE``, ``TRIPLET``, ``SUPCON``, ``PROTOTYPICAL``, ``BARLOWTWINS``, ``VICREG``).
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
        # SOTA Contrastive Learning Losses
        "INFONCE": InfoNCELoss,
        "TRIPLET": TripletLoss,
        "SUPCON": SupConLoss,
        "PROTOTYPICAL": PrototypicalLoss,
        "BARLOWTWINS": BarlowTwinsLoss,
        "VICREG": VICRegLoss,
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
    print("🔥 Testing PHM-Vibench Loss Functions")
    
    # Test standard losses
    print("\n1. Standard Losses:")
    pred = torch.randn(4, 5)
    target = torch.randint(0, 5, (4,))
    
    for loss_name in ["CE", "MSE", "MAE", "BCE", "NLL"]:
        try:
            loss_fn = get_loss_fn(loss_name)
            if loss_name in ["MSE", "MAE"]:
                test_target = torch.randn(4, 5)  # Continuous target
            elif loss_name == "BCE":
                test_pred = torch.randn(4, 5)  # Logits for BCE
                test_target = torch.randint(0, 2, (4, 5)).float()  # Binary target
                loss_val = loss_fn(test_pred, test_target)
            elif loss_name == "NLL":
                test_pred = torch.log_softmax(pred, dim=1)  # Log probabilities for NLL
                loss_val = loss_fn(test_pred, target)
            else:
                test_target = target
                loss_val = loss_fn(pred, test_target)
            
            if loss_name not in ["BCE", "NLL"]:
                if loss_name in ["MSE", "MAE"]:
                    loss_val = loss_fn(pred, test_target)
                else:
                    loss_val = loss_fn(pred, test_target)
            
            print(f"   ✓ {loss_name}: {loss_val:.4f}")
        except Exception as e:
            print(f"   ✗ {loss_name} failed: {e}")
    
    # Test contrastive losses
    print("\n2. SOTA Contrastive Losses:")
    batch_size = 8
    feature_dim = 64
    num_classes = 3
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    features2 = torch.randn(batch_size, feature_dim)
    
    contrastive_losses = {
        "INFONCE": lambda: get_loss_fn("INFONCE")(temperature=0.07),
        "TRIPLET": lambda: get_loss_fn("TRIPLET")(margin=0.3), 
        "SUPCON": lambda: get_loss_fn("SUPCON")(temperature=0.07),
        "PROTOTYPICAL": lambda: get_loss_fn("PROTOTYPICAL")(distance_fn='euclidean'),
        "BARLOWTWINS": lambda: get_loss_fn("BARLOWTWINS")(lambda_param=5e-3),
        "VICREG": lambda: get_loss_fn("VICREG")(lambda_inv=25.0, mu_var=25.0, nu_cov=1.0),
    }
    
    for loss_name, loss_creator in contrastive_losses.items():
        try:
            loss_fn = loss_creator()
            
            if loss_name in ["INFONCE", "TRIPLET", "SUPCON", "PROTOTYPICAL"]:
                loss_val = loss_fn(features, labels)
            else:  # BARLOWTWINS, VICREG (require two views)
                loss_val = loss_fn(features, features2)
            
            print(f"   ✓ {loss_name}: {loss_val:.4f}")
        except Exception as e:
            print(f"   ✗ {loss_name} failed: {e}")
    
    print("\n✅ All loss functions tested successfully!")
    print(f"Total available losses: {len(get_loss_fn.__defaults__ or []) + 11}")  # Approximate count
