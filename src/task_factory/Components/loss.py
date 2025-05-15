import torch.nn as nn

def get_loss_fn(loss_name: str) -> nn.Module:
    """根据名称获取损失函数实例"""
    loss_mapping = {
        "CE": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
        "BCE": nn.BCEWithLogitsLoss()
        # 可以根据需要添加更多损失函数
    }
    loss_name_upper = loss_name.upper()
    if loss_name_upper not in loss_mapping:
        raise ValueError(f"不支持的损失函数类型: {loss_name}"
                         f"，可选类型: {list(loss_mapping.keys())}")
    return loss_mapping[loss_name_upper]