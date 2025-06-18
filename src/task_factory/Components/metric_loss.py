# matching_loss.py ─────────────────────────────────────────────
"""
Clean, well-documented Matching-Networks loss.

Batch layout (fixed)
────────────────────
domain ──► class ──► [ support | query ]

Key features
────────────
* arbitrary integer labels; mapping handled automatically
* optional `split_domain`:  True  → 1 domain = 1 episode  
                            False → merge all domains
* variables spelled out for readability (e.g. `support_indices`)
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS: float = 1e-8  # small number for numerical stability
# ══════════════════════════════════════════════════════════════





# ───────────────────── 2. main criterion ─────────────────────
class MatchingLoss(nn.Module):
    """Few-shot Matching-Networks loss (domain grouping only)."""

    def __init__(self, cfg: MatchCfg) -> None:
        super().__init__()
        self.cfg: MatchCfg = cfg
        self.metric: str   = cfg.matching_metric.lower()
        self.nll: nn.Module = nn.NLLLoss()

    # ----------------------------------------------------------
    def forward(self,
                embeddings: torch.Tensor,   # (B, D)
                labels: torch.Tensor        # (B,)
                ) -> tuple[torch.Tensor, float]:

        episode_pairs = _enumerate_episodes(       # [(support_indices, query_indices), …]
            total=len(labels),
            cfg=self.cfg,
            device=labels.device
        )

        loss_accumulator: float = 0.0
        acc_accumulator:  float = 0.0
        total_query_samples: int = 0

        for support_indices, query_indices in episode_pairs:  # 

            # ── gather embeddings & labels for this episode ────────────
            support_embeddings = embeddings[support_indices]
            query_embeddings   = embeddings[query_indices]
            support_labels     = labels[support_indices]
            query_labels       = labels[query_indices]

            # ── derive the class set of THIS episode ───────────────────
            #    (labels can be 101, 202, … any integers)
            unique_classes = torch.unique(support_labels, sorted=True)
            classes_in_episode: int = len(unique_classes)
            supports_per_class: int = support_embeddings.size(0) // classes_in_episode

            # map original integer labels → [0 … k-1] for NLL / one-hot
            label_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
            query_label_indices = torch.tensor(
                [label_to_index[int(l)] for l in query_labels],
                device=query_labels.device
            )

            # ── Matching-Networks pipeline ─────────────────────────────
            pairwise_distance = _pairwise_dist(query_embeddings,
                                               support_embeddings,
                                               self.metric)          # shape (Q , k·n)
            attention = (-pairwise_distance).softmax(dim=-1)          # softmax over supports
            prediction = _attention_to_probs(attention,
                                             supports_per_class,
                                             classes_in_episode)      # shape (Q , k)

            # ── loss & accuracy for this episode ──────────────────────
            loss_episode = self.nll(prediction.clamp(EPS, 1-EPS).log(),
                                    query_label_indices)
            accuracy_episode = (prediction.argmax(1) == query_label_indices) \
                               .float().mean().item()

            num_query_episode = query_indices.numel()
            loss_accumulator += loss_episode.item() * num_query_episode
            acc_accumulator  += accuracy_episode   * num_query_episode
            total_query_samples += num_query_episode

        # aggregate across all episodes in the batch
        if total_query_samples == 0:
            return torch.tensor(0., device=embeddings.device), 0.0

        mean_loss = torch.tensor(loss_accumulator / total_query_samples,
                                 device=embeddings.device)
        mean_acc  = acc_accumulator / total_query_samples
        return mean_loss, mean_acc


# ───────────────────── 3. helper functions ───────────────────
def _enumerate_episodes(total: int, cfg: MatchCfg, device) \
        -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Produce a list of (support_indices , query_indices) for each episode.

    Batch layout is **fixed** as::

        domain0 :  class0  support/query
                   class1  support/query
                   ...
        domain1 :  class0  ...
        ...

    If `split_domain=False`, all domains are merged into **one** episode.
    """
    samples_per_class  = cfg.num_support + cfg.num_query
    samples_per_domain = cfg.num_labels * samples_per_class
    assert total == cfg.num_domains * samples_per_domain, \
        "Batch size does not match cfg descriptors."

    # decide whether each domain forms its own episode
    domain_groups = range(cfg.num_domains) # if cfg.domain_wise_metric else [range(cfg.num_domains)]
    episode_list: list[tuple[torch.Tensor, torch.Tensor]] = []

    for domain_selection in domain_groups:

        if isinstance(domain_selection, int):
            domain_iterable = [domain_selection]
        else:
            domain_iterable = domain_selection  # merged range

        support_indices, query_indices = [], []

        for d in domain_iterable:
            domain_base = d * samples_per_domain # domain offset
            for c in range(cfg.num_labels):
                offset = domain_base + c * samples_per_class
                # collect support indices for this (domain, class)
                support_indices.extend(range(offset,                  # support offset + cfg.num_support
                                             offset + cfg.num_support))
                # collect query indices for this (domain, class)
                query_indices.extend(range(offset + cfg.num_support,  # query offset + cfg.num_query
                                           offset + samples_per_class))

        episode_list.append((
            torch.tensor(support_indices, device=device, dtype=torch.long),
            torch.tensor(query_indices,  device=device, dtype=torch.long)
        ))
    return episode_list


def _pairwise_dist(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    """Compute pairwise distance / similarity matrix."""
    x = x.squeeze()
    y = y.squeeze()
    if metric == "cosine":
        x_normed = x / (x.norm(2, -1, keepdim=True) + EPS)
        y_normed = y / (y.norm(2, -1, keepdim=True) + EPS)
        return 1 - x_normed @ y_normed.T
    if metric == "l2":
        return (x[:, None] - y[None, :]).pow(2).sum(dim=-1)
    if metric == "dot":
        return -(x @ y.T)
    raise ValueError(f"Unsupported metric: {metric}")


def _attention_to_probs(attention: torch.Tensor,
                        supports_per_class: int,
                        num_classes: int) -> torch.Tensor:
    """
    Convert attention matrix to class-probabilities.

    Parameters
    ----------
    attention           : shape (Q , k*n)
    supports_per_class  : n
    num_classes         : k
    """
    one_hot_support_labels = F.one_hot(
        torch.arange(num_classes, device=attention.device)
              .repeat_interleave(supports_per_class),      # length k*n
        num_classes
    ).float()                                              # shape (k*n , k)

    # Y_pred = attention  @  one_hot_labels
    return attention @ one_hot_support_labels              # shape (Q , k)


# ───────────────────── 4. demonstration ──────────────────────
if __name__ == "__main__":
# ───────────────────── 1. configuration ──────────────────────
    @dataclass
    class MatchCfg:
        # few-shot parameters
        num_support:  int                 # shots per class (n-shot)
        num_query:    int                 # queries per class
        num_labels:   int                 # classes per domain (k-way)

        # batch organisation
        num_domains:  int = 1             # how many domains in this batch
        num_systems:  int = 1             # kept for YAML parity – NOT used

        # options
        split_domain: bool = True         # True: one episode per domain
        matching_metric:       str  = "cosine"     # 'cosine' | 'l2' | 'dot'
    # YAML-style parameters
    cfg = MatchCfg(
        num_support=2,
        num_query=2,
        num_labels=3,
        num_domains=2,
        num_systems=2,          # ignored
    )

    criterion = MatchingLoss(cfg)

    # build a toy batch: 2 domains × 3 classes × (2+2) = 24 samples
    # labels purposely non-contiguous: domain0 = {10, 20, 30},
    #                                  domain1 = {110, 120, 130}
    domain0_labels = torch.tensor([10]*2 + [10]*2 +
                                  [20]*2 + [20]*2 +
                                  [30]*2 + [30]*2)
    domain1_labels = domain0_labels + 100
    batch_labels   = torch.cat([domain0_labels, domain1_labels])        # (24,)

    batch_embeddings = torch.randn(24, 64)

    loss_value, acc_value = criterion(batch_embeddings, batch_labels)
    print(f"loss = {loss_value:.4f} ; accuracy = {acc_value:.2%}")
