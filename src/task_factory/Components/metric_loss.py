# matching_loss.py  ─────────────────────────────────────────────
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

EPS = 1e-8
# ────────────────────────── Matching-Loss ──────────────────────
class MatchingLoss(nn.Module):
    r"""
    PyTorch criterion implementing **Matching Networks** episodic loss.

    Let support set \(S=\{(x_s,y_s)\}_{s=1}^{k n}\),
        query set \(Q=\{(x_q,y_q)\}_{q=1}^{k q}\).

    Steps
    -----
    1. Distance \(d_{qs}=1-\frac{x_q\cdot x_s}{\lVert x_q\rVert\lVert x_s\rVert}\)
    2. Attention \(a_{qs}=\frac{e^{-d_{qs}}}{\sum_{s'}e^{-d_{qs'}}}\)
    3. Prediction \(\hat y_q=a_q Y_S^{\text{one-hot}}\)
    4. Loss \(\mathcal L=-\sum_q\log\hat y_q^{\,y_q}\)

    Returns `(loss, accuracy)` for the query set.
    """

    def __init__(self, cfg: MatchCfg):
        super().__init__()
        self.cfg = cfg
        self.nll = nn.NLLLoss()
        self.metric = cfg.metric.lower()

    # ───────────────────── forward (public) ────────────────────
    def forward(
        self,
        emb: torch.Tensor,          # (B, D)   episode embeddings
        lbl: torch.Tensor,          # (B,)     episode integer labels
        n_support: int | None = None,
        n_way:     int | None = None,
        n_query:   int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        If `n_support / n_way / n_query` are **None** they default to `self.cfg`.
        """
        n_support = n_support or self.cfg.num_support
        n_way     = n_way     or self.cfg.num_labels
        n_query   = n_query   or self.cfg.num_query

        device = emb.device

        # 1️⃣ split indices ------------------------------------------------
        classes = torch.unique(lbl, sorted=True)
        if len(classes) != n_way:
            raise ValueError(f"Episode has {len(classes)} classes; "
                             f"expected n_way={n_way}")

        sup_idx, qry_idx = self._support_query_indices(lbl, classes, n_support)
        support, query = emb[sup_idx], emb[qry_idx]          # (k·n,D), (k·q,D)

        # 2️⃣ pair-wise distances -----------------------------------------
        dist = self._pairwise_dist(query, support, self.metric)  # (k·q,k·n)

        # 3️⃣ attention weights -------------------------------------------
        attn = (-dist).softmax(dim=-1)                          # (k·q,k·n)

        # 4️⃣ predictions --------------------------------------------------
        y_pred = self._predict(attn, n_support, n_way, n_query) # (k·q,k)

        # 5️⃣ loss & accuracy ---------------------------------------------
        loss = self.nll(y_pred.clamp(EPS, 1-EPS).log(),
                        lbl[qry_idx].long().to(device))

        acc = (y_pred.argmax(1).cpu()
               == lbl[qry_idx].cpu()).float().mean().item()

        return loss, acc

    # ───────────────── helper: build indices ──────────────────
    @staticmethod
    def _support_query_indices(
        labels:  torch.Tensor,
        classes: torch.Tensor,
        n_sup:   int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sup, qry = [], []
        for c in classes:
            idx = labels.eq(c).nonzero(as_tuple=False).squeeze(1)
            sup.append(idx[:n_sup])
            qry.append(idx[n_sup:])
        return torch.cat(sup), torch.cat(qry)

    # ───────────────── helper: distances ───────────────────────
    @staticmethod
    def _pairwise_dist(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
        if metric == "cosine":
            x_n = x / (x.norm(dim=1, keepdim=True) + EPS)
            y_n = y / (y.norm(dim=1, keepdim=True) + EPS)
            return 1 - x_n @ y_n.T
        if metric == "l2":
            x2, y2 = x.pow(2).sum(1, keepdim=True), y.pow(2).sum(1).unsqueeze(0)
            return x2 + y2 - 2 * (x @ y.T)
        if metric == "dot":
            return -(x @ y.T)
        raise ValueError(f"Unsupported metric '{metric}'")

    # ───────────────── helper: prediction ──────────────────────
    @staticmethod
    def _predict(attn: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    # attn: 注意力权重张量，形状为 (k * q, k * n)
    #       k * q 是查询样本的总数 (k 个类别，每个类别 q 个查询样本)
    #       k * n 是支持样本的总数 (k 个类别，每个类别 n 个支持样本)
    #       attn[i, j] 表示第 i 个查询样本对第 j 个支持样本的注意力权重。
    #       每一行 attn[i, :] 的和为 1 (因为它是 softmax 的输出)。
    # n: 每个类别的支持样本数
    # k: 类别数 (way)
    # q: 每个类别的查询样本数

    # 1. 检查注意力张量的形状是否符合预期
        if attn.shape != (k * q, k * n):
            raise ValueError(f"Attention shape {attn.shape} "
                             f"vs expected {(k*q, k*n)}")
    # 2. 创建支持集样本的标签 (整数形式)
    #    假设支持集样本是按类别顺序排列的，例如：
    #    所有类别0的支持样本，然后是所有类别1的支持样本，以此类推。
    #    torch.arange(k): 生成 [0, 1, ..., k-1]，代表 k 个类别。
    #    .repeat_interleave(n, output_size=k*n): 将每个类别索引重复 n 次。
    #    例如，如果 k=3, n=2，labels 会是 [0, 0, 1, 1, 2, 2]。
    #    这个张量的长度是 k*n，对应支持集中的每一个样本。
        labels = torch.arange(k).repeat_interleave(n, output_size=k*n).to(attn.device)
    # labels 的形状是 (k*n,)

    # 3. 将支持集标签转换为 one-hot 编码
    #    F.one_hot(labels, num_classes=k): 将整数标签转换为 one-hot 向量。
    #    例如，如果 labels = [0, 0, 1, 1, 2, 2] 且 k=3，Y_onehot 会是:
    #    [[1, 0, 0],  // 第1个支持样本，属于类别0
    #     [1, 0, 0],  // 第2个支持样本，属于类别0
    #     [0, 1, 0],  // 第3个支持样本，属于类别1
    #     [0, 1, 0],  // 第4个支持样本，属于类别1
    #     [0, 0, 1],  // 第5个支持样本，属于类别2
    #     [0, 0, 1]]  // 第6个支持样本，属于类别2
    #    .float(): 转换为浮点数类型，以便进行矩阵乘法。
        Y_onehot = F.one_hot(labels, num_classes=k).float()   # (k·n,k)
    # Y_onehot 的每一行代表一个支持样本，每一列代表一个类别。

    # 4. 计算预测结果
    #    这是核心步骤：将注意力权重矩阵 attn 与支持集标签的 one-hot 编码矩阵 Y_onehot 相乘。
    #    attn: (k*q, k*n)
    #    Y_onehot: (k*n, k)
    #    结果 y_pred 的形状是 (k*q, k)
    #
    #    理解这个矩阵乘法：
    #    对于第 i 个查询样本 (对应 attn 的第 i 行，attn[i, :])
    #    以及第 c 个类别 (对应 Y_onehot 的第 c 列，Y_onehot[:, c])
    #
    #    y_pred[i, c] = sum_{j=0}^{k*n-1} (attn[i, j] * Y_onehot[j, c])
    #
    #    attn[i, j] 是第 i 个查询样本对第 j 个支持样本的注意力。
    #    Y_onehot[j, c] 是 1 如果第 j 个支持样本属于类别 c，否则是 0。
    #
    #    所以，y_pred[i, c] 实质上是第 i 个查询样本对所有“属于类别 c 的支持样本”的注意力权重之和。
    #    这可以被看作是第 i 个查询样本属于类别 c 的“证据”或“分数”。
        return attn @ Y_onehot                                # (k·q,k)


if __name__ == "__main__":

    # ────────────────────────── Config dataclass ───────────────────
    @dataclass
    class MatchCfg:
        num_support: int  # n  (shot)
        num_query:   int  # q
        num_labels:  int  # k  (way)
        metric:      str = "cosine"  # 'cosine' | 'l2' | 'dot'
    cfg = MatchCfg(num_support=5, num_query=5, num_labels=5)
    criterion = MatchingLoss(cfg).cuda()

    k, n, q = cfg.num_labels, cfg.num_support, cfg.num_query
    emb = torch.randn(k*(n+q), 64, device='cuda')           # 50×64
    lbl = torch.arange(k, device='cuda').repeat_interleave(n+q)

    loss, acc = criterion(emb, lbl)
    print(f"loss={loss.item():.4f}, acc={acc:.2%}")
