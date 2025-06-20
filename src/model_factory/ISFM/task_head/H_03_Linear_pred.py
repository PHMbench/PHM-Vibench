import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class H_03_Linear_pred(nn.Module):
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

    def __init__(self, args):
        super().__init__()

        # flat_dim       = args.num_patches * args.d_model

        hidden  = getattr(args, "hidden_dim", 64)
        max_len = getattr(args, "max_len", 4096)
        max_out = getattr(args, "max_out", 3)  # 
        actname = getattr(args, "act", "relu")

        Act  = {"relu": nn.ReLU, "gelu": nn.GELU, None: nn.Identity}[actname]


        self.fc1   = nn.Linear(args.output_dim, hidden)
        self.fc2   = nn.Linear(args.num_patches, hidden)
        self.act   = Act()

        # Universal projection kernel  (hidden -> max_len * max_out)
        self.weight = nn.Parameter(torch.randn(int(hidden**2), max_len * max_out))
        self.bias   = nn.Parameter(torch.zeros(max_len * max_out))

        # store meta
        self.max_len = max_len
        self.max_out = max_out

    # ----------------------------------------------------------
    def forward(self,
                x: torch.Tensor,            # (B,L,C)
                shape: tuple = None, **kwargs) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be (B,L,C)")
        pred_len = min(shape[0], self.max_len)
        out_dim  = min(shape[1], self.max_out)

        if pred_len > self.max_len or out_dim > self.max_out:
            raise ValueError(f"Requested ({pred_len}, {out_dim}) exceeds "
                             f"kernel capacity ({self.max_len}, {self.max_out})")
        B = x.size(0)

        # ① flatten whole signal
        # h = x.reshape(B, -1)                # (B, P*C)

        # ② hidden projection
         # (B, P*C)
        h = self.act(self.fc1(x))           # (B, hidden)
        h = rearrange(h, "B L C -> B C L") 
        h = self.act(self.fc2(h))           # (B, hidden)
        h = h.view(B, -1)                # (B, hidden ** 2)

        # ③ universal projection
        univ = F.linear(h, self.weight.T, self.bias)   # (B, max_len*max_out)
        univ = univ.view(B, self.max_len, self.max_out)

        # ④ slice to desired block
        y = univ[:, :pred_len, :out_dim]    # (B, pred_len, out_dim)
        return y


# ---------------------------- demo -----------------------------
if __name__ == "__main__":
    class Args:
        num_patches  = 128
        patch_size_L = 256
        in_dim       = 64
        hidden       = 256
        max_len      = 4096
        max_out      = 8
        act          = "gelu"
        output_dim      = 64

    args = Args()
    head = H_03_Linear_pred(args).cuda()

    B = 4
    # P = args.num_patches * args.patch_size_L
    x = torch.randn(B, args.num_patches, args.output_dim, device="cuda")
    shape = (4096, 3)  # desired output shape
    y = head(x, shape=shape)  # (B, pred_len, out_dim)
    print(y.shape)        # torch.Size([4, 4096, 3])
