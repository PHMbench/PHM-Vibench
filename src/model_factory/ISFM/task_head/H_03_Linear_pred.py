import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenProjectHead(nn.Module):
    """
    1) Flatten (B, P, C) -> (B, P*C)
    2) Hidden MLP -> (B, hidden)
    3) Universal projection kernel  hidden -> (max_len * max_out)
    4) Reshape & slice to (B, pred_len, out_dim)

    Runtime arguments
    -----------------
    pred_len : int    – desired temporal length
    out_dim  : int    – desired channel count   (<= max_out)
    """

    def __init__(
        self,
        P: int,                    # num_patches * patch_size
        in_dim: int,               # C
        hidden: int = 512,         # “bottleneck” width
        max_len: int = 2048,       # maximum horizon the kernel can cover
        max_out: int = 512,        # maximum channel count kernel covers
        act: str | None = "gelu",
    ):
        super().__init__()
        Act = {"relu": nn.ReLU, "gelu": nn.GELU, None: nn.Identity}[act]

        flat_dim = P * in_dim
        self.fc1   = nn.Linear(flat_dim, hidden)
        self.act   = Act()

        # Universal projection kernel  (hidden -> max_len * max_out)
        self.weight = nn.Parameter(torch.randn(hidden, max_len * max_out))
        self.bias   = nn.Parameter(torch.zeros(max_len * max_out))

        # store meta
        self.max_len = max_len
        self.max_out = max_out

    # ----------------------------------------------------------
    def forward(self,
                x: torch.Tensor,            # (B,P,C)
                pred_len: int,
                out_dim:  int) -> torch.Tensor:

        if pred_len > self.max_len or out_dim > self.max_out:
            raise ValueError(f"Requested ({pred_len}, {out_dim}) exceeds "
                             f"kernel capacity ({self.max_len}, {self.max_out})")
        if x.dim() != 3:
            raise ValueError("Input must be (B,P,C)")

        B = x.size(0)

        # ① flatten whole signal
        h = x.reshape(B, -1)                # (B, P*C)

        # ② hidden projection
        h = self.act(self.fc1(h))           # (B, hidden)

        # ③ universal projection
        univ = F.linear(h, self.weight.T, self.bias)   # (B, max_len*max_out)
        univ = univ.view(B, self.max_len, self.max_out)

        # ④ slice to desired block
        y = univ[:, :pred_len, :out_dim]    # (B, pred_len, out_dim)
        return y


# ---------------------------- demo -----------------------------
if __name__ == "__main__":
    num_patches, patch_size, C_in = 128, 256, 64
    P = num_patches * patch_size

    head = FlattenProjectHead(P=P,
                              in_dim=C_in,
                              hidden=256,
                              max_len=4096,
                              max_out=8).cuda()

    B = 4
    x = torch.randn(B, P, C_in, device="cuda")

    y = head(x, pred_len=4096, out_dim=3)
    print(y.shape)   # torch.Size([4, 4096, 128])
