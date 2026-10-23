"""Frozen TSPN + configured operator branches + one probability coefficient.

Model forward is deployment-only. Train with forward_details and TSPNFusionLoss,
whose nonzero tau is intentionally different from the assessed deployment alpha.
The reference is evaluated once, without hooks or a copied reference backbone.
All frequencies below are cycles/sample; v2 assumes a fixed sampling convention
per task and supplies no extra RPM/condition classifier to the new branches.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import SimpleNamespace

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _bounded_logit(initial: Tensor, limits: Tensor) -> Tensor:
    fraction = (initial - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    if not torch.all((fraction > 0) & (fraction < 1)):
        raise ValueError("Initial filter parameters must be strictly inside their bounds.")
    return torch.logit(fraction)


class GaussianBands(nn.Module):
    """Real-frequency masks with bounded learned centers and standard deviations.

    Three-width support is a declared numerical convention, not compact support.
    Bounds guarantee this support throughout learning, rather than repairing it.
    """
    def __init__(self, centers, widths, center_bounds, width_bounds):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float32)
        widths = torch.as_tensor(widths, dtype=torch.float32)
        cb = torch.as_tensor(center_bounds, dtype=torch.float32)
        wb = torch.as_tensor(width_bounds, dtype=torch.float32)
        if centers.ndim != 1 or centers.numel() == 0 or widths.shape != centers.shape:
            raise ValueError("centers/widths must be nonempty equal-length vectors.")
        if cb.shape != (len(centers), 2) or wb.shape != cb.shape:
            raise ValueError("Each band needs its own lower/upper parameter bounds.")
        if not all(torch.isfinite(t).all() for t in (centers, widths, cb, wb)):
            raise ValueError("Filter parameters must be finite.")
        if not torch.all((cb[:, 1] > cb[:, 0]) & (wb[:, 1] > wb[:, 0]) & (wb[:, 0] > 0)):
            raise ValueError("Invalid filter bounds.")
        if not torch.all((cb[:, 0] - 3*wb[:, 1] > 0) & (cb[:, 1] + 3*wb[:, 1] < .5)):
            raise ValueError("The full allowed three-width support must lie in (0, Nyquist).")
        self.register_buffer("center_bounds", cb)
        self.register_buffer("width_bounds", wb)
        self.center_parameter = nn.Parameter(_bounded_logit(centers, cb))
        self.width_parameter = nn.Parameter(_bounded_logit(widths, wb))

    @property
    def count(self) -> int:
        return self.center_parameter.numel()

    def parameters_in_frequency(self) -> tuple[Tensor, Tensor]:
        def transform(raw, limits):
            return limits[:, 0] + (limits[:, 1] - limits[:, 0]) * raw.sigmoid()
        return transform(self.center_parameter, self.center_bounds), transform(self.width_parameter, self.width_bounds)

    def forward(self, frequency: Tensor) -> Tensor:
        center, width = self.parameters_in_frequency()
        return torch.exp(-.5 * ((frequency[None, :] - center[:, None]) / width[:, None]).square())

    def check_grid(self, length: int) -> None:
        # A numerical sampling rule, not a physical resolution guarantee.
        if torch.any(2 * self.width_bounds[:, 0] * length < 2):
            raise ValueError("Under-sampled band: increase measured window length or declare wider bands.")


def analytic_envelope(x: Tensor) -> Tensor:
    """Hilbert magnitude on the last axis, with correct odd/even multipliers."""
    n = x.shape[-1]
    mask = x.new_zeros(n)
    mask[0] = 1
    if n % 2 == 0:
        mask[1:n//2] = 2
        mask[n//2] = 1
    else:
        mask[1:(n+1)//2] = 2
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * mask, dim=-1).abs()


class EnvelopeBranch(nn.Module):
    """1D: Gaussian carrier bank -> envelope -> modulation-band energy.

    Input B,L,C; named carrier/modulation statistics. The FFT boundary is periodic.
    Optional impulsiveness, energy and lag operators reuse the same filtered signal.
    Amplitude is preserved through the mean envelope; no per-sample whitening.
    """
    def __init__(self, in_channels: int, carrier: Mapping, modulation: Mapping,
                 diagnostics: Mapping | None = None):
        super().__init__()
        self.carrier = GaussianBands(**carrier)
        self.modulation = GaussianBands(**modulation)
        self.diagnostics = None if diagnostics is None else copy.deepcopy(dict(diagnostics))
        names = ['log_mean_envelope'] + [f'log_modulation_energy_{j}' for j in range(self.modulation.count)]
        if self.diagnostics is not None:
            if set(self.diagnostics) != {'lags', 'epsilon'}:
                raise ValueError("Envelope diagnostics require exactly lags and epsilon.")
            lags, epsilon = self.diagnostics['lags'], self.diagnostics['epsilon']
            if (not isinstance(lags, (list, tuple)) or not lags or
                    any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags) or
                    len(set(lags)) != len(lags)):
                raise ValueError("Declare unique positive integer lags in samples, not shaft orders.")
            if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("Declare a positive fixed diagnostic epsilon.")
            names += ['log_band_kurtosis', 'log_envelope_cv2', 'asinh_teager_ratio']
            names += [f'envelope_correlation_lag_{lag}' for lag in lags]
        self.feature_names = [f'channel_{c}/carrier_{k}/{name}'
                              for c in range(in_channels) for k in range(self.carrier.count) for name in names]
        self.output_dim = len(self.feature_names)

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[1]
        self.carrier.check_grid(n)
        self.modulation.check_grid(n)
        if self.diagnostics is not None and (n < 3 or max(self.diagnostics['lags']) > n - 2):
            raise ValueError("Each declared lag needs at least two observed pairs; no circular lag padding.")
        frequency = torch.fft.rfftfreq(n, device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(x.transpose(1, 2), dim=-1, norm="ortho")
        masks = self.carrier(frequency)
        filtered = torch.fft.irfft(spectrum[:, :, None, :] * masks[None, None, :, :],
                                  n=n, dim=-1, norm="ortho")
        envelope = analytic_envelope(filtered)
        mean = envelope.mean(-1, keepdim=True)
        centered = envelope - mean
        power = torch.fft.rfft(centered, dim=-1, norm="ortho").abs().square()
        weights = self.modulation(frequency)
        weights = weights / weights.sum(-1, keepdim=True)
        energies = torch.einsum("bckf,mf->bckm", power, weights)
        values = [torch.log1p(mean), torch.log1p(energies)]
        if self.diagnostics is not None:
            # Explicit regularized operators; epsilon is fixed in signal-squared
            # units, not fitted from a target batch. Exact scale invariance is not claimed.
            eps = self.diagnostics['epsilon']
            yc = filtered - filtered.mean(-1, keepdim=True)
            variance = yc.square().mean(-1, keepdim=True)
            kurtosis = yc.pow(4).mean(-1, keepdim=True) / (variance + eps).square()
            cv2 = centered.square().mean(-1, keepdim=True) / (mean.square() + eps)
            teager = (filtered[..., 1:-1].square() - filtered[..., :-2] * filtered[..., 2:]).mean(-1, keepdim=True)
            teager = teager / (filtered.square().mean(-1, keepdim=True) + eps)
            values.extend([torch.log1p(kurtosis), torch.log1p(cv2), torch.asinh(teager)])
            for lag in self.diagnostics['lags']:
                left, right = centered[..., :-lag], centered[..., lag:]
                denominator = ((left.square().mean(-1, keepdim=True) + eps) *
                               (right.square().mean(-1, keepdim=True) + eps)).sqrt()
                values.append((left * right).mean(-1, keepdim=True) / denominator)
        return torch.cat(values, dim=-1).flatten(1)


def mixture_log_probs(raw_logits: Tensor, candidate_logits: Tensor, alpha: float,
                      temperature: float = 1.) -> Tensor:
    """Probability mixture, NOT a convex mixture of logits. No probability clipping."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("alpha must be in [0,1] and reference temperature must be positive.")
    log_p = F.log_softmax(raw_logits / temperature, dim=-1)
    if alpha == 0:
        return log_p
    log_q = F.log_softmax(candidate_logits, dim=-1)
    if alpha == 1:
        return log_q
    return torch.logaddexp(log_p + math.log1p(-alpha), log_q + math.log(alpha))


class OperatorResidualHead(nn.Module):
    """One zero-anchored nonlinear function per physical branch, not cross-branch mixing.

    The last matrix starts at zero: initialization exactly preserves the calibrated
    TSPN logits. Additivity explains logit changes, not unique physical causation.
    """
    def __init__(self, dimension: int, hidden: int, classes: int):
        super().__init__()
        self.hidden = nn.Linear(dimension, hidden)
        nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        self.output = nn.Linear(hidden, classes, bias=False)
        nn.init.zeros_(self.output.weight)

    def effective_weights(self, cap: Tensor) -> tuple[Tensor, Tensor]:
        root = cap.sqrt()
        return tuple(w / (w.norm() / root).clamp_min(1.)
                     for w in (self.hidden.weight, self.output.weight))

    def forward(self, z: Tensor, cap: Tensor) -> Tensor:
        w1, w2 = self.effective_weights(cap)
        h = F.relu(F.linear(z, w1, self.hidden.bias)) - F.relu(self.hidden.bias)
        value = F.linear(h, w2)
        # Softmax is unchanged by a common class shift. Center to give one
        # identifiable class-contrast convention; zero features give zero evidence.
        return value - value.mean(-1, keepdim=True)


class TSPNFusion(nn.Module):
    """One frozen reference and a configuration-assembled operator family.

    Each named branch produces its own vector; no TF maps are resized to another
    branch's coordinate grid. Empty branches are allowed only for the refit-head
    control. They are not presented as an operator-enhanced model.
    """
    def __init__(self, reference: nn.Module, *, in_channels: int, num_classes: int,
                 branches, use_reference_features: bool = True,
                 reference_temperature: float = 1., head_frobenius_cap: float = 5.,
                 head_type: str = "linear", head_hidden_dim: int = 16):
        super().__init__()
        if int(reference.args.num_classes) != num_classes or int(reference.args.in_channels) != in_channels:
            raise ValueError("Reference and new predictor must share channels and labels.")
        if not math.isfinite(reference_temperature) or reference_temperature <= 0:
            raise ValueError("Invalid frozen reference temperature.")
        if not math.isfinite(head_frobenius_cap) or head_frobenius_cap <= 0:
            raise ValueError("head_frobenius_cap must be positive.")
        if not isinstance(branches, (list, tuple)):
            raise ValueError("branches must be an explicitly ordered sequence.")
        if not branches and not use_reference_features:
            raise ValueError("The candidate needs at least one feature source.")
        if head_type not in {"linear", "mlp", "operator_residual"}:
            raise ValueError("head_type must be linear, mlp or operator_residual.")
        if head_type in {"mlp", "operator_residual"} and (isinstance(head_hidden_dim, bool) or
                not isinstance(head_hidden_dim, int) or head_hidden_dim < 1):
            raise ValueError("An MLP needs an explicit positive integer hidden width.")
        self.head_type = head_type
        self.reference = reference.requires_grad_(False).eval()
        self.in_channels, self.num_classes = int(in_channels), int(num_classes)
        self.use_reference_features = bool(use_reference_features)
        self.branch_specs = copy.deepcopy(list(branches))
        self.branches = nn.ModuleDict()
        self.feature_dims = {}
        if self.use_reference_features:
            self.feature_dims["reference"] = int(reference.channel_for_classifier)
        for item in branches:
            spec = dict(item)
            name, kind = spec.pop("name"), spec.pop("type")
            if not isinstance(name, str) or not name or "." in name or name == "reference" or name in self.branches:
                raise ValueError("Branch names must be nonempty, unique and not 'reference'.")
            if kind == "envelope":
                branch = EnvelopeBranch(in_channels, **spec)
            else:
                from .TSPN_tf_operators import TimeFrequencyBranch
                branch = TimeFrequencyBranch(kind, in_channels, **spec)
            self.branches[name] = branch
            self.feature_dims[name] = branch.output_dim
        dimension = sum(self.feature_dims.values())
        # Preserve legacy linear state keys. The MLP is a readout comparator,
        # not another encoder or a new signal-processing branch.
        if head_type == "operator_residual":
            self.operator_heads = nn.ModuleDict({name: OperatorResidualHead(d, head_hidden_dim, num_classes)
                                                 for name, d in self.feature_dims.items()})
        else:
            if head_type == "mlp":
                self.candidate_hidden = nn.Linear(dimension, head_hidden_dim)
                nn.init.kaiming_uniform_(self.candidate_hidden.weight, nonlinearity="relu")
                nn.init.zeros_(self.candidate_hidden.bias)
            self.candidate_head = nn.Linear(head_hidden_dim if head_type == "mlp" else dimension, num_classes)
            nn.init.normal_(self.candidate_head.weight, mean=0., std=.01)
            nn.init.zeros_(self.candidate_head.bias)
        self.register_buffer("head_frobenius_cap", torch.tensor(float(head_frobenius_cap)))
        self.register_buffer("alpha", torch.tensor(0., dtype=torch.float64))
        self.register_buffer("reference_temperature", torch.tensor(reference_temperature, dtype=torch.float64))

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    @torch.no_grad()
    def _reference_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self.reference.eval()
        z = x
        for layer in self.reference.signal_processing_layers:
            z = layer(z)
        z = self.reference.feature_extractor_layers(z)
        return z, self.reference.clf(z)

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.in_channels or not x.is_floating_point():
            raise ValueError("Expected finite real floating input B,L,C.")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite input; preprocessing is not silently changed.")

    @torch.no_grad()
    def set_alpha(self, alpha: float) -> None:
        """Assign a fixed deployment coefficient; assignment alone is not certification."""
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1].")
        self.alpha.fill_(alpha)

    def effective_head_weight(self) -> Tensor:
        if self.head_type == "operator_residual":
            raise ValueError("The operator-residual readout has separate branch matrices, not a joint head.")
        w = self.candidate_head.weight
        cap = self.head_frobenius_cap.sqrt() if self.head_type == "mlp" else self.head_frobenius_cap
        return w/(torch.linalg.vector_norm(w)/cap).clamp_min(1.)

    def forward_details(self, x: Tensor) -> dict:
        self._check_input(x)
        z0, raw_logits = self._reference_features(x)
        feature_dict = {}
        if self.use_reference_features:
            feature_dict["reference"] = torch.asinh(z0)
        for name, branch in self.branches.items():
            feature_dict[name] = branch(x)
        contributions = {}
        if self.head_type == "operator_residual":
            scale = math.sqrt(len(feature_dict))
            contributions = {name: self.operator_heads[name](z / math.sqrt(self.feature_dims[name]),
                                                            self.head_frobenius_cap) / scale
                             for name, z in feature_dict.items()}
            logits = raw_logits / self.reference_temperature.item() + torch.stack(list(contributions.values())).sum(0)
        else:
            features = torch.cat([z/math.sqrt(self.feature_dims[name]) for name, z in feature_dict.items()], -1)
            if self.head_type == "mlp":
                w = self.candidate_hidden.weight
                cap = self.head_frobenius_cap.sqrt()
                w = w/(torch.linalg.vector_norm(w)/cap).clamp_min(1.)
                features = F.relu(F.linear(features, w, self.candidate_hidden.bias))
            logits = F.linear(features, self.effective_head_weight(), self.candidate_head.bias)
        p0 = F.softmax(raw_logits/self.reference_temperature.item(), -1)
        q = F.softmax(logits, -1)
        return {"raw_logits": raw_logits, "candidate_logits": logits,
                "raw_probs": p0, "candidate_probs": q, "correction": q-p0,
                "raw_features": z0, "branch_features": feature_dict,
                "reference_temperature": self.reference_temperature,
                **({"branch_logit_contributions": contributions} if contributions else {})}

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        if self.training:
            raise RuntimeError("Use eval() for deployment; forward_details for training.")
        self._check_input(x)
        alpha = self.alpha.item()
        if alpha == 0:
            _, logits = self._reference_features(x)
            return F.softmax(logits/self.reference_temperature.item(), -1)
        out = self.forward_details(x)
        return (1-alpha)*out["raw_probs"]+alpha*out["candidate_probs"]

    def forward(self, x: Tensor, data_id=None, task_id=None) -> Tensor:
        if self.training:
            raise RuntimeError("Train with forward_details + TSPNFusionLoss; alpha is not tau.")
        self._check_input(x)
        if self.alpha.item() == 0:
            _, logits = self._reference_features(x)
            return F.log_softmax(logits/self.reference_temperature.item(), -1)
        out = self.forward_details(x)
        return mixture_log_probs(out["raw_logits"], out["candidate_logits"], self.alpha.item(), self.reference_temperature.item())


class Model(TSPNFusion):
    """Existing PHMFactory constructor and strict canonical checkpoint loader.

    The upstream loader accepts bare model states and the platform's network.
    Lightning prefix. No guessed prefixes or partial weight loading are added.
    """
    def __init__(self, args, metadata=None):
        from .TSPN import Model as OriginalTSPN
        from ..model_factory import load_ckpt
        if args.checkpoint_kind not in {"reference", "fusion"}:
            raise ValueError("checkpoint_kind must be reference or fusion.")
        raw = copy.deepcopy(args.reference_config)
        raw = vars(raw) if not isinstance(raw, Mapping) else dict(raw)
        if int(args.num_classes) != int(raw["num_classes"]):
            raise ValueError("Top-level and reference class spaces must agree.")
        raw["device"] = args.device
        reference = OriginalTSPN(SimpleNamespace(**raw))
        if args.checkpoint_kind == "reference":
            load_ckpt(reference, args.checkpoint_path, strict=True)
        super().__init__(reference, in_channels=int(raw["in_channels"]), num_classes=int(raw["num_classes"]),
                         branches=args.branches, use_reference_features=bool(getattr(args, "use_reference_features", True)),
                         reference_temperature=float(args.reference_temperature),
                         head_frobenius_cap=float(getattr(args, "head_frobenius_cap", 5.)),
                         head_type=getattr(args, "head_type", "linear"),
                         head_hidden_dim=getattr(args, "head_hidden_dim", 16))
        if args.checkpoint_kind == "fusion":
            load_ckpt(self, args.checkpoint_path, strict=True)
        self.to(args.device)
