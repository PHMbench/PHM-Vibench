import torch
from typing import Dict, Any, Iterable

def calculate_regularization(reg_config: Dict[str, Any], params: Iterable[torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    """
    计算正则化损失。

    Args:
        reg_config: 正则化配置字典，例如 {'flag': True, 'method': {'l1': 0.01, 'l2': 0.005}}
        params: 需要计算正则化的模型参数迭代器。

    Returns:
        一个字典，包含每种正则化损失和总正则化损失 ('total')。
        如果不启用正则化或没有有效的正则化方法，则返回 {'total': tensor(0.0)}。
    """
    reg_losses = {}
    total_reg_loss = torch.tensor(0.0, device=next(iter(params)).device if params else 'cpu', dtype=torch.float32) # 获取设备信息

    if not reg_config or not reg_config.get('flag', False):
        reg_losses['total'] = total_reg_loss
        return reg_losses

    method_dict = reg_config.get('method', {})
    trainable_params = [p for p in params if p.requires_grad]

    if not trainable_params: # 如果没有可训练参数
        reg_losses['total'] = total_reg_loss
        return reg_losses

    for reg_type, weight in method_dict.items():
        if weight == 0:
            continue

        current_reg_loss = torch.tensor(0.0, device=total_reg_loss.device, dtype=torch.float32)
        reg_type_lower = reg_type.lower()

        if reg_type_lower == 'l1':
            for p in trainable_params:
                current_reg_loss += torch.norm(p, 1)
        elif reg_type_lower == 'l2':
             for p in trainable_params:
                current_reg_loss += torch.norm(p, 2).pow(2) # L2 正则化通常是范数的平方
        else:
            print(f"警告: 不支持的正则化类型: {reg_type}，已跳过。")
            continue

        weighted_loss = weight * current_reg_loss
        reg_losses[reg_type_lower] = weighted_loss
        total_reg_loss += weighted_loss

    reg_losses['total'] = total_reg_loss
    return reg_losses