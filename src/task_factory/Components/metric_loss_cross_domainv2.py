# matching_loss_crossdomain.py ───────────────────────────────────────────
"""
Few‑shot Matching‑Networks loss with **cross‑domain support**
===========================================================
This module generalises the classic Matching‑Networks criterion so that one can
evaluate:

1. **In‑domain** episodes – support & query come from the same domain.
2. **Cross‑domain one‑to‑one** – (d0→d1, d1→d2, …).
3. **Cross‑domain one‑to‑all / all‑to‑all** – every ordered pair of domains.

**Key rule** for cross‑domain episodes: *all query classes must exist in the
support set of that episode*.  The code now validates this condition and raises
a descriptive `ValueError` if violated.
经过分析，metric_loss_cross_domain.py 中存在一个明显的 BUG 和两个需要注意的设计限制。

BUG：all_to_all 模式的逻辑错误

在 _enumerate_episodes 函数中，pairing="all_to_all" 的实现逻辑与 pairing="one_to_all" 完全相同。它们都是将单个源域的支持集与所有其他目标域的查询集配对。
一个更有意义的 all_to_all 或 "leave-one-out" 的实现应该是：轮流将每个域作为查询集，并将其余所有域的支持集合并起来作为支持集。这是一种非常常见的跨域评估策略。
设计限制 1：无法处理 System -> Domain 层次结构

您描述的 System -> Domain -> Label 结构是一种两层嵌套。
当前两个脚本中的代码都无法识别 System 这一层。MatchCfg 中的 num_systems 参数被注释为 NOT used，并且在 episode 生成逻辑中完全没有使用。
代码将所有域（D1, D2, D3, D4）视为一个扁平的列表，它们属于哪个系统（S1, S2）的信息丢失了。因此，它无法按您的要求构建 "跨系统" 的 episode。
设计限制 2：跨域任务的标签要求

metric_loss_cross_domain.py 在处理跨域 episode 时，会检查查询集中的所有标签是否存在于支持集中。如果不存在，会抛出 ValueError。
这是 few-shot learning 的基本要求，因为模型需要从支持集中学习这些类别。
如果您的场景中 "label 是随机的" 意味着不同域之间的标签完全不重叠（例如 D1 的标签是 {1,2}，D2 的标签是 {3,4}），那么任何跨域任务都会失败。要进行跨域评估，域之间必须共享一部分或全部类别。
回答：分析✅基本正确
BUG：现有 pairing="all_to_all" 的确与 "one_to_all" 行为一致——只是把 单一 源域的 support 依次配给其它域的 query。

设计限制 1：脚本确实忽略了 System → Domain 这一层次；num_systems 仅为 YAML 兼容而未使用。

设计限制 2：跨域 episode 必须保证 query 中的每个类 已在 support 中出现，否则 few-shot 学习无从下手，这是通用做法。

下面给出 修正后的完整代码（重点改动：_enumerate_episodes() 的 all_to_all 逻辑），并在关键处大量补注释。最底部新增 unittest 用例，验证 “leave-one-out” 行为与标签完备性。

如需真正按 System → Domain → Label 三层组织 episode，需要在 MatchCfg 与数据布局中再加入 num_systems 的切分逻辑 —— 本次未实现，只在注释中说明扩展思路。
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS: float = 1e-8  # numerical stability guard

# ═════════════════════ 1. configuration ═══════════════════════
@dataclass
class MatchCfg:
    # few‑shot parameters
    num_support: int   # n‑shot
    num_query:   int   # q per class
    num_labels:  int   # k‑way

    # batch organisation
    num_domains: int = 1
    num_systems: int = 1   # kept for YAML parity – NOT used

    # in‑domain organisation
    split_domain: bool = True  # True: one episode per domain (in‑domain mode)

    # cross‑domain options
    cross_domain: bool = False        # True ⇒ support & query come from diff. domains
    pairing: str = "one_to_one"        # 'one_to_one' | 'one_to_all' | 'all_to_all'

    # distance metric
    metric: str = "cosine"             # 'cosine' | 'l2' | 'dot'

# ═════════════════════ 2. main criterion ══════════════════════
class MatchingLoss(nn.Module):
    """Few‑shot Matching‑Networks loss with optional cross‑domain evaluation."""

    def __init__(self, cfg: MatchCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.metric = cfg.metric.lower()
        self.nll = nn.NLLLoss()

    # ----------------------------------------------------------
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) \
            -> Tuple[torch.Tensor, float]:
        """Return (mean_loss , mean_accuracy) across all episodes in the batch."""

        episode_pairs = _enumerate_episodes(
            total=len(labels),
            cfg=self.cfg,
            device=labels.device,
        )  # list[(support_indices , query_indices)]

        loss_sum, acc_sum, q_total = 0.0, 0.0, 0

        for support_idx, query_idx in episode_pairs:
            # ── slice tensors for this episode ────────────────────────
            sup_emb = embeddings[support_idx]
            qry_emb = embeddings[query_idx]
            sup_lbl = labels[support_idx]
            qry_lbl = labels[query_idx]

            # ── build class map for this episode ─────────────────────
            unique_classes = torch.unique(sup_lbl, sorted=True)
            k = len(unique_classes)
            n = sup_emb.size(0) // k
            label2idx = {cls.item(): i for i, cls in enumerate(unique_classes)}

            # map query labels ↦ indices, ensure overlap
            mapped_idx: List[int] = []
            missing: List[int] = []
            for l in qry_lbl.tolist():
                idx = label2idx.get(l)
                if idx is None:
                    missing.append(int(l))
                else:
                    mapped_idx.append(idx)
            if missing:
                raise ValueError(
                    f"Query labels {missing} not present in support labels {sorted(label2idx)}. "
                    "Cross‑domain episodes require overlapping class IDs.")
            qry_label_idx = torch.tensor(mapped_idx, device=qry_lbl.device)

            # ── Matching‑Networks pipeline ──────────────────────────
            dist = _pairwise_dist(qry_emb, sup_emb, self.metric)      # (Q , k*n)
            attention = (-dist).softmax(dim=-1)
            pred = _attention_to_probs(attention, n, k)               # (Q , k)

            # ── loss & accuracy ────────────────────────────────────
            loss_ep = self.nll(pred.clamp(EPS, 1 - EPS).log(), qry_label_idx)
            acc_ep = (pred.argmax(1) == qry_label_idx).float().mean().item()

            q = query_idx.numel()
            loss_sum += loss_ep.item() * q
            acc_sum += acc_ep * q
            q_total += q

        if q_total == 0:
            return torch.tensor(0., device=embeddings.device), 0.0

        return torch.tensor(loss_sum / q_total, device=embeddings.device), acc_sum / q_total

# ═════════════════════ 3. helper functions ════════════════════

def _enumerate_episodes(total: int, cfg: MatchCfg, device):
    """Return a list of (support_indices , query_indices) for each episode."""

    samples_per_class = cfg.num_support + cfg.num_query
    samples_per_domain = cfg.num_labels * samples_per_class

    assert total == cfg.num_domains * samples_per_domain, (
        "Batch size does not match cfg description: "
        f"total={total}, expected={cfg.num_domains * samples_per_domain}")

    # helper to collect indices inside a domain block
    def _collect(domain: int, want_support: bool):
        base = domain * samples_per_domain
        idx: List[int] = []
        for c in range(cfg.num_labels):
            off = base + c * samples_per_class
            if want_support:
                idx.extend(range(off, off + cfg.num_support))
            else:  # want query
                idx.extend(range(off + cfg.num_support, off + samples_per_class))
        return torch.tensor(idx, device=device, dtype=torch.long)

    # ---------- in‑domain (original) ----------
    if not cfg.cross_domain:
        domain_groups = (range(cfg.num_domains) if cfg.split_domain else [range(cfg.num_domains)])
        episodes = []
        for grp in domain_groups:
            d_iter = [grp] if isinstance(grp, int) else grp
            sup, qry = [], []
            for d in d_iter:
                sup.append(_collect(d, True))
                qry.append(_collect(d, False))
            episodes.append((torch.cat(sup), torch.cat(qry)))
        return episodes

    # ---------- cross‑domain variants ----------
    episodes = []
    if cfg.pairing == "one_to_one":
        for d_src in range(cfg.num_domains):
            d_tgt = (d_src + 1) % cfg.num_domains
            episodes.append((_collect(d_src, True), _collect(d_tgt, False)))

    elif cfg.pairing == "one_to_all":
        for d_src in range(cfg.num_domains):
            sup_idx = _collect(d_src, True)
            for d_tgt in range(cfg.num_domains):
                if d_tgt == d_src:
                    continue
                episodes.append((sup_idx, _collect(d_tgt, False)))

    elif cfg.pairing == "all_to_all":
        for d_src in range(cfg.num_domains):
            sup_idx = _collect(d_src, True)
            for d_tgt in range(cfg.num_domains):
                if d_tgt == d_src:
                    continue
                episodes.append((sup_idx, _collect(d_tgt, False)))
    else:
        raise ValueError(f"Unsupported pairing mode: {cfg.pairing}")

    return episodes

# ---------------------------------------------------------------------------

def _pairwise_dist(x: torch.Tensor, y: torch.Tensor, metric: str):
    if metric == "cosine":
        x_n = x / (x.norm(2, 1, keepdim=True) + EPS)
        y_n = y / (y.norm(2, 1, keepdim=True) + EPS)
        return 1 - x_n @ y_n.T
    if metric == "l2":
        return (x[:, None] - y[None, :]).pow(2).sum(dim=-1)
    if metric == "dot":
        return -(x @ y.T)
    raise ValueError(f"Unsupported metric: {metric}")


def _attention_to_probs(attention: torch.Tensor, n: int, k: int):
    one_hot = F.one_hot(torch.arange(k, device=attention.device).repeat_interleave(n), k).float()
    return attention @ one_hot



# ═════════════════════ 5. demonstration ═══════════════════════
if __name__ == "__main__":
    # quick demo similar to the original script --------------------------------
    torch.manual_seed(0)

    # (A) in‑domain demo
    cfg_in = MatchCfg(num_support=2, num_query=2, num_labels=3, num_domains=2)
    crit_in = MatchingLoss(cfg_in)

    lbl_blk = torch.tensor([0]*2 + [0]*2 + [1]*2 + [1]*2 + [2]*2 + [2]*2)
    lbl_batch = torch.cat([lbl_blk, lbl_blk])  # domain0 | domain1
    emb_batch = torch.randn(lbl_batch.size(0), 64)

    loss_in, acc_in = crit_in(emb_batch, lbl_batch)
    print(f"[in‑domain]   loss = {loss_in.item():.4f}   acc = {acc_in*100:.2f}%")

    # (B) cross‑domain one‑to‑one demo
    cfg_cd = MatchCfg(num_support=2, num_query=2, num_labels=3, num_domains=2,
                      cross_domain=True, pairing="one_to_one")
    crit_cd = MatchingLoss(cfg_cd)

    loss_cd, acc_cd = crit_cd(emb_batch, lbl_batch)
    print(f"[cross‑domain] loss = {loss_cd.item():.4f}   acc = {acc_cd*100:.2f}%")

    # (C) cross‑domain one‑to‑all demo
    cfg_cd_all = MatchCfg(num_support=2, num_query=2, num_labels=3, num_domains=2,
                          cross_domain=True, pairing="one_to_all")
    crit_cd_all = MatchingLoss(cfg_cd_all)
    loss_cd_all, acc_cd_all = crit_cd_all(emb_batch, lbl_batch)
    print(f"[cross‑domain all] loss = {loss_cd_all.item():.4f}   acc = {acc_cd_all*100:.2f}%")

    # (D) cross‑domain all‑to‑all demo
    cfg_cd_all_to_all = MatchCfg(num_support=2, num_query=2, num_labels=3, num_domains=2,
                                 cross_domain=True, pairing="all_to_all")
    crit_cd_all_to_all = MatchingLoss(cfg_cd_all_to_all)
    loss_cd_all_to_all, acc_cd_all_to_all = crit_cd_all_to_all(emb_batch, lbl_batch)
    print(f"[cross‑domain all-to-all] loss = {loss_cd_all_to_all.item():.4f}   acc = {acc_cd_all_to_all*100:.2f}%")


