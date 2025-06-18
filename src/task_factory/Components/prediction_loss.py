# signal_prediction_loss.py  ───────────────────────────────────
from __future__ import annotations
import torch, torch.nn as nn
from dataclasses import dataclass
from src.utils.masking import add_mask

EPS = 1e-8
# ───────────────────────── Loss Module ────────────────────────
class Signal_mask_Loss(nn.Module):
    r"""
    **Self-supervised signal loss** that blends **imputation** and **forecasting**.

    Given an episode signal \(X\in\mathbb R^{B\times L\times C}\):

    * **Random mask**  
      For each time-step \(t<L_f\) we drop it with prob. ρ.
    * **Prediction mask**  
      Entire last \(L_f=\lfloor L\cdot\lambda\rfloor\) steps are hidden.
    * The model receives \(\tilde X\) (zeros at masked positions)  
      and outputs \(\hat X=f_\theta(\tilde X)\).
    * Loss is computed **only** on the masked points
      \[
        \mathcal L=
        \begin{cases}
        \text{MSE: }\frac{1}{N}\sum_{m}(\hat x_m-x_m)^2 \\[4pt]
        \text{RelL2: }\dfrac{\lVert\hat X-X\rVert_2}{\lVert X\rVert_2}
        \end{cases}
      \]

    Returns `(loss, stats_dict)`
    """

    def __init__(self, cfg: SigPredCfg):
        super().__init__()
        self.cfg = cfg
        self.loss_type = 'rel_l2' # mse

    # ──────────────────────── forward ─────────────────────────
    def forward(self,
                model:  nn.Module,
                batch: torch.Tensor           # (B,L,C) ground-truth
                ) -> tuple[torch.Tensor, dict]:
        signal = batch['x']
        file_id = batch.get('file_id', None)

        x_in, total_mask, mask_rand, mask_pred = add_mask(signal, self.cfg.forecast_part, self.cfg.mask_ratio, return_component_masks=True)

        # 3️⃣ model prediction --------------------------------------
        with torch.set_grad_enabled(self.training):
            x_hat = model(x_in,file_id, task_id = 'prediction') 
            # x_hat = model(x_in)                            # (B,L,C)

        # 4️⃣ compute loss -----------------------------------------
        if self.loss_type == "mse":
            num = total_mask.sum().clamp(min=1)             # avoid /0
            loss = nn.MSELoss(reduction="sum")(x_hat[total_mask], signal[total_mask]) / num

        elif self.loss_type == "rel_l2":                # relative L2
            diff = (x_hat - signal)[total_mask]
            loss = diff.pow(2).sum() / (signal[total_mask].pow(2).sum() + EPS)



        # # # 5️⃣ stats --------------------------------------------------
        # stats = {
        #     "impute_frac": mask_rand.float().mean().item(),
        #     "forecast_frac": mask_pred.float().mean().item(),
        #     "mask_total_frac": total_mask.float().mean().item()
        # }
        return loss#, stats


if __name__ == "__main__":
    # ─────────────────────── Configuration dataclass ──────────────
    @dataclass
    class SigPredCfg:
        mask_ratio: float = 0.50      # Bernoulli-mask ratio on *observed* part
        loss_type:  str   = "rel_l2"     # "mse" | "rel_l2"
        forecast_part: float = 0.5    # fraction of sequence regarded as “future”
    # Dummy model that just copies input
    class Identity(nn.Module):
        def forward(self, x, file_id=None, task_id=None): return x

    cfg   = SigPredCfg(mask_ratio=0.5, loss_type="mse", forecast_part=0.5)
    crit  = Signal_mask_Loss(cfg).cuda()
    model = Identity().cuda()

    B, L, C = 8, 100, 3
    x = torch.randn(B, L, C, device="cuda")
    batch = {'x': x, 'file_id': None}  # Simulated batch with input signal

    loss, st = crit(model, batch)
    print(f"loss={loss.item():.4f}  stats={st}")
