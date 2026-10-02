"""P4 research implementation, not a recovered historical XOAN checkpoint.

Input is a dimensionless, training-standardized (batch, length, channels) tensor.
Every selectable edge executes tanh(operator(h)); the bounded coordinates and
linear Avg/Var head are explicit P4 changes needed by the extraction bound.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Any
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

OPERATORS = ("I", "D1", "ABS", "SQUARE", "MA3", "HT")
LIPSCHITZ = (1.0, 2.0, 1.0, 2.0, 1.0, math.sqrt(2.0))
# Relative operation budgets, not measured FLOPs or hardware latency.
COST = (1, 2, 2, 2, 3, 8)
FAMILY = {"I": "identity", "D1": "difference", "ABS": "pointwise_amplitude",
          "SQUARE": "pointwise_amplitude", "MA3": "smoothing", "HT": "analytic_envelope"}


def sparsemax(scores: Tensor) -> Tensor:
    """Euclidean simplex projection along the final axis (Martins & Astudillo)."""
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    sorted_scores = shifted.sort(dim=-1, descending=True).values
    k = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
    cumulative = sorted_scores.cumsum(-1)
    support = (1 + k * sorted_scores > cumulative).sum(-1, keepdim=True)
    threshold = (cumulative.gather(-1, support - 1) - 1) / support.to(scores.dtype)
    return (shifted - threshold).clamp_min(0)


def operator(x: Tensor, name: str) -> Tensor:
    """Execute the named discrete operator; HT is an envelope, not signed Hilbert."""
    if name == "I":
        return x
    if name == "D1":
        return torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
    if name == "ABS":
        return x.abs()
    if name == "SQUARE":
        return x.square()
    if name in {"MA3", "MA5"}:
        size = int(name[2:])
        padded = F.pad(x.transpose(1, 2), (size // 2, size // 2), mode="replicate")
        return F.avg_pool1d(padded, size, stride=1).transpose(1, 2)
    if name == "HT":
        length = x.shape[1]
        multiplier = x.new_zeros(length)
        multiplier[0] = 1
        multiplier[1:(length + 1) // 2] = 2
        if length % 2 == 0:
            multiplier[length // 2] = 1
        analytic = torch.fft.ifft(torch.fft.fft(x, dim=1) * multiplier[None, :, None], dim=1)
        return analytic.abs()
    raise ValueError(f"Unknown operator {name!r}; no replacement is selected automatically.")


def pool(x: Tensor) -> Tensor:
    return torch.cat((x.mean(1), x.var(1, unbiased=False)), dim=1)


class Trace(NamedTuple):
    states: list[Tensor]
    weights: list[Tensor]
    branches: list[Tensor]


class OperatorNet(nn.Module):
    """Three serial stages; all downstream gates can be frozen without freezing signals."""
    def __init__(self, channels: int, classes: int, mode: str = "sparse", stages: int = 3,
                 learned: bool = False, bounded: bool = True) -> None:
        super().__init__()
        if mode not in {"sparse", "dense", "uniform"}:
            raise ValueError("mode must be sparse, dense, or uniform")
        if channels < 1 or classes < 2 or stages < 1:
            raise ValueError("Positive channels/stages and at least two classes are required")
        self.mode, self.stages, self.bounded, self.learned = mode, stages, bounded, learned
        self.gates = nn.ModuleList(nn.Sequential(nn.Linear(2 * channels, 16), nn.Tanh(),
                                                 nn.Linear(16, 6)) for _ in range(stages))
        for gate in self.gates:
            nn.init.normal_(gate[-1].weight, std=0.02)
            nn.init.zeros_(gate[-1].bias)
        self.transitions = nn.ModuleList(
            nn.ModuleList(nn.Conv1d(channels, channels, 3, padding=1) for _ in OPERATORS)
            for _ in range(stages)) if learned else None
        self.head = nn.Linear(2 * channels, classes)

    def initial(self, x: Tensor) -> Tensor:
        return x.tanh() if self.bounded else x

    def branch(self, h: Tensor, stage: int, edge: int, replacement: str | None = None) -> Tensor:
        if replacement is not None:
            z = operator(h, replacement)
        elif self.learned:
            z = self.transitions[stage][edge](h.transpose(1, 2)).transpose(1, 2)
        else:
            z = operator(h, OPERATORS[edge])
        return z.tanh() if self.bounded else z

    def forward(self, x: Tensor, frozen: list[Tensor] | None = None,
                intervention: tuple[int, int, str] | None = None,
                policy_removal: bool = False) -> tuple[Tensor, Trace]:
        if x.ndim != 3 or x.shape[1] < 5:
            raise ValueError("Expected (batch,length>=5,channels)")
        if frozen is not None and len(frozen) != self.stages:
            raise ValueError("Supply one reference allocation for every stage")
        if policy_removal and frozen is not None:
            raise ValueError("Policy removal recomputes routing; it cannot also freeze routing")
        if intervention is not None:
            s, j, replacement = intervention
            if not (0 <= s < self.stages and 0 <= j < 6):
                raise ValueError("Intervention stage/edge is outside the executed graph")
            if replacement not in {*OPERATORS, "MA5", "ZERO"}:
                raise ValueError("Unrecognized intervention")
        h = self.initial(x)
        states, allocations, all_branches = [h], [], []
        for stage in range(self.stages):
            z = torch.stack([self.branch(h, stage, j) for j in range(6)], dim=1)
            if frozen is None:
                scores = self.gates[stage](pool(h))
                if policy_removal and intervention is not None and stage == intervention[0]:
                    # Exclude the edge before projecting; do not relabel this as fixed knockout.
                    keep = [j for j in range(6) if j != intervention[1]]
                    selected = scores[:, keep]
                    weights_kept = (torch.softmax(selected, -1) if self.mode == "dense" else
                                    torch.ones_like(selected) / 5 if self.mode == "uniform" else sparsemax(selected))
                    weights = torch.zeros_like(scores)
                    weights[:, keep] = weights_kept
                else:
                    weights = (torch.softmax(scores, -1) if self.mode == "dense" else
                               torch.ones_like(scores) / 6 if self.mode == "uniform" else sparsemax(scores))
            else:
                weights = frozen[stage]
                if weights.shape != (x.shape[0], 6):
                    raise ValueError("Frozen allocation shape must equal (batch,6)")
            if intervention is not None and stage == intervention[0] and not policy_removal:
                edge, name = intervention[1], intervention[2]
                z = z.clone()
                z[:, edge] = torch.zeros_like(h) if name == "ZERO" else self.branch(h, stage, edge, name)
            h = (weights[:, :, None, None] * z).sum(dim=1)
            allocations.append(weights)
            all_branches.append(z)
            states.append(h)
        return self.head(pool(h)), Trace(states, allocations, all_branches)

    def discrete(self, x: Tensor, path: tuple[int, ...]) -> Tensor:
        if len(path) != self.stages or any(j not in range(6) for j in path):
            raise ValueError("One valid edge per stage is required")
        h = self.initial(x)
        for stage, edge in enumerate(path):
            h = self.branch(h, stage, edge)
        return self.head(pool(h))

    def residual_loss(self, trace: Trace) -> Tensor:
        """Piecewise differentiable nearest-operator residual, not a proof of task improvement."""
        defects = [(z - trace.states[s + 1][:, None]).square().mean((2, 3)).amin(1)
                   for s, z in enumerate(trace.branches)]
        return torch.stack(defects).sum(0).mean()

    @staticmethod
    def routing_concentration_loss(trace: Trace) -> Tensor:
        """Same-architecture endpoint penalty; not a Fair DARTS reproduction."""
        return torch.stack([(a * (1 - a)).sum(-1).mean() for a in trace.weights]).mean()

    def bound(self, trace: Trace, path: tuple[int, ...]) -> Tensor:
        """Real-arithmetic, input-wise logit bound for the declared bounded fixed operators."""
        if self.learned or not self.bounded:
            raise ValueError("Analytic operator bound does not apply to learned/unbounded controls")
        batch, length, _ = trace.states[0].shape
        discrepancy = trace.states[0].new_zeros(batch)
        for s, j in enumerate(path):
            # Residual evaluated on the reference prefix; propagation uses the selected path's L.
            residual = (trace.states[s + 1] - trace.branches[s][:, j]).flatten(1).norm(dim=1)
            discrepancy = LIPSCHITZ[j] * discrepancy + residual
        head_l = self.head.weight.norm(dim=1).amax() * math.sqrt(5 / length)
        return head_l * discrepancy

    @torch.no_grad()
    def extract(self, x: Tensor, budget: int = 238, relative_tolerance: float = 0.49,
                strategy: str = "cost") -> dict:
        """Select with a total forward-call budget, including reference/ranking calls.

        The perturbation comparator ranks frozen-reference edge knockouts and
        searches paths by summed edge effect, without retraining. It is a local
        adaptation, not a reproduction of DARTS-PT. Candidate calls reuse an
        already executed argmax path. Timing includes all CUDA synchronization.
        """
        if x.shape[0] != 1 or self.training:
            raise ValueError("extract requires model.eval() and a single sample")
        if not (0 < relative_tolerance < 0.5) or budget < 2:
            raise ValueError("Use total budget>=2 and 0<tolerance<0.5")
        if strategy not in {"cost", "perturbation"}:
            raise ValueError("strategy must be cost or perturbation")
        def sync() -> None:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
        sync()
        started = time.perf_counter()
        logits, trace = OperatorNet.forward(self, x)
        top = logits.topk(2, dim=-1)
        winner, rival = (int(i) for i in top.indices[0])
        margin = float(top.values[0, 0] - top.values[0, 1])
        costs = (1,) * 6 if self.learned else COST
        labels = tuple(f"learned_{j}" for j in range(6)) if self.learned else OPERATORS
        paths = list(itertools.product(range(6), repeat=self.stages))
        argmax_path = tuple(int(a[0].argmax()) for a in trace.weights)
        argmax_logits = self.discrete(x, argmax_path)
        cache = {argmax_path: argmax_logits}
        result = dict(accepted=False, path=None, queries=0, candidate_queries=0,
                      ranking_queries=0, reference_queries=1, argmax_queries=1,
                      total_queries=2, strategy=strategy, total_query_budget=budget,
                      margin=margin, prediction=int(logits.argmax(-1)),
                      argmax_agrees=bool(logits.argmax(-1) == argmax_logits.argmax(-1)),
                      argmax_gap=float((logits - argmax_logits).abs().max()),
                      analytic_bound=None, analytic_sufficient=False, exact_gap=None,
                      search_complete=False, cost=None, replay_seconds=None,
                      replay_queries=0, extraction_seconds=None,
                      minimum_declared_cost=False)
        def finish() -> dict:
            sync()
            result["extraction_seconds"] = time.perf_counter() - started
            return result
        effects = x.new_zeros((self.stages, 6))
        if strategy == "perturbation":
            # Partial rankings change the comparator, so fail closed on its budget.
            if budget < 2 + 6 * self.stages:
                raise ValueError("Perturbation ranking needs reference+argmax+6*stages calls")
            frozen = [a.detach() for a in trace.weights]
            for stage in range(self.stages):
                for edge in range(6):
                    modified, _ = OperatorNet.forward(
                        self, x, frozen=frozen, intervention=(stage, edge, "ZERO"))
                    effects[stage, edge] = margin - (modified[0, winner] - modified[0, rival])
                    result["ranking_queries"] += 1
                    result["total_queries"] += 1
            paths.sort(key=lambda p: (-sum(float(effects[s, j]) for s, j in enumerate(p)),
                                     sum(costs[j] for j in p), p))
        else:
            paths.sort(key=lambda p: (sum(costs[j] for j in p), p))
        tested = 0
        for path in paths:
            if path in cache:
                candidate = cache[path]
            else:
                if result["total_queries"] >= budget:
                    break
                candidate = self.discrete(x, path)
                result["candidate_queries"] += 1
                result["queries"] += 1
                result["total_queries"] += 1
            tested += 1
            gap = float((logits - candidate).abs().max())
            if margin > 0 and gap <= relative_tolerance * margin:
                analytic = float(self.bound(trace, path)) if self.bounded and not self.learned else None
                result.update(accepted=True, path=[labels[j] for j in path],
                              cost=sum(costs[j] for j in path), exact_gap=gap,
                              analytic_bound=analytic,
                              analytic_sufficient=analytic is not None and 2 * analytic < margin,
                              minimum_declared_cost=strategy == "cost",
                              search_complete=tested == len(paths))
                # Replay is separately measured and charged against the same budget.
                if result["total_queries"] < budget:
                    sync()
                    replay_started = time.perf_counter()
                    self.discrete(x, path)
                    sync()
                    result["replay_seconds"] = time.perf_counter() - replay_started
                    result["replay_queries"] = 1
                    result["total_queries"] += 1
                return finish()
        result["search_complete"] = tested == len(paths)
        return finish()


class ConvControl(nn.Module):
    """Task-utility control; it has no named-operator explanation claim."""
    def __init__(self, channels: int, classes: int, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, width, 7, padding=3), nn.GELU(),
                                      nn.Conv1d(width, width, 5, padding=2), nn.Tanh())
        self.head = nn.Linear(2 * width, classes)

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        features = self.features(x.transpose(1, 2)).transpose(1, 2)
        return self.head(pool(features)), None


class Model(OperatorNet):
    """PHMFactory logits interface for the bounded P07 network.

    Factory CE training is the sparse baseline. Evidence-bearing residual or
    concentration training uses the P07 runner and this same OperatorNet class;
    the shared task must not silently omit the declared regularizer.
    """
    def __init__(self, args: Any, metadata: Any = None) -> None:
        del metadata
        if getattr(args, "residual_weight", 0) or getattr(args, "concentration_weight", 0):
            raise ValueError("Regularized P07 training requires the explicit P07 runner objective")
        super().__init__(channels=int(args.in_channels), classes=int(args.num_classes),
                         mode=getattr(args, "mode", "sparse"), stages=int(getattr(args, "stages", 3)))

    def forward(self, x: Tensor, data_id: Any = None, task_id: Any = None) -> Tensor:
        del data_id, task_id
        return OperatorNet.forward(self, x)[0]
