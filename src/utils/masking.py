import torch

def add_mask(signal, forecast_part, mask_ratio, return_component_masks=False):
    """
    Applies forecasting and random masking to a signal tensor.

    Args:
        signal (torch.Tensor): The input signal tensor of shape (B, L, C).
        forecast_part (float): The fraction of the sequence to be masked for forecasting.
        mask_ratio (float): The ratio of random masking for imputation.
        return_component_masks (bool): If True, also returns individual mask components.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x_in (torch.Tensor): The masked signal tensor.
            - total_mask (torch.Tensor): The boolean mask tensor.
        If return_component_masks is True, returns:
            - x_in, total_mask, mask_rand, mask_pred_expanded
    """
    B, L, C = signal.shape
    device = signal.device

    L_f = int(L * forecast_part)
    L_o = L - L_f

    # Prediction mask for forecasting
    mask_pred = torch.zeros(L, dtype=torch.bool, device=device)
    mask_pred[L_o:] = True

    # Random mask for imputation
    mask_rand = (torch.rand((B, L_o, 1), device=device) < mask_ratio)
    mask_rand = torch.cat([mask_rand, torch.zeros(B, L_f, 1, device=device)], 1)
    mask_rand = mask_rand.bool().expand(-1, -1, C)

    # Combine masks
    mask_pred_expanded = mask_pred.unsqueeze(0).unsqueeze(2).expand(B, L, C)
    total_mask = mask_pred_expanded | mask_rand

    # Apply mask to signal
    x_in = signal.clone()
    x_in[total_mask] = 0.0
    
    if return_component_masks:
        return x_in, total_mask, mask_rand, mask_pred_expanded
    else:
        return x_in, total_mask
