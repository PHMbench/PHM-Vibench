"""Four-arm model binding for the frozen P08 E1 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from typing import Final

import torch
import torch.nn.functional as F

from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model
from src.task_factory.Components.contrastive_losses import InfoNCELoss


ARM_IDS: Final = ("P08-DN", "P08-M", "P08-BG", "P08-NC")


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    use_physical_duration: bool
    use_band_projection: bool
    use_prompt: bool
    physical_patch_duration_s: float
    fixed_raw_token_points: int
    global_resample_numerator_hz: int | None = None
    global_resample_denominator: int | None = None

    @property
    def global_resample_target_hz(self) -> float | None:
        if self.global_resample_numerator_hz is None:
            return None
        if self.global_resample_denominator is None:
            raise RuntimeError("incomplete global-resampling rational")
        return self.global_resample_numerator_hz / self.global_resample_denominator

    def to_dict(self) -> dict[str, object]:
        # Exact numerator/denominator fields are the only serialized authority.
        # A redundant float display is omitted so selection traces contain no
        # ambiguous "target" field and remain safe for source-only auditing.
        return asdict(self)


def arm_spec(
    arm_id: str,
    *,
    duration_ms: float | None = None,
    global_resample_numerator_hz: int | None = None,
    global_resample_denominator: int | None = None,
) -> ArmSpec:
    """Resolve one protocol arm and fail on undeclared degrees of freedom."""

    if arm_id not in ARM_IDS:
        raise ValueError(f"arm_id must be one of {ARM_IDS}, got {arm_id!r}")
    if arm_id in {"P08-DN", "P08-M"}:
        if duration_ms not in {5.0, 10.0, 15.0}:
            raise ValueError("P08-DN/P08-M duration_ms must be 5, 10, or 15")
        if (
            global_resample_numerator_hz is not None
            or global_resample_denominator is not None
        ):
            raise ValueError("native-rate arms cannot set a global resampling rate")
        return ArmSpec(
            arm_id=arm_id,
            use_physical_duration=True,
            use_band_projection=True,
            use_prompt=arm_id == "P08-M",
            physical_patch_duration_s=float(duration_ms) / 1000.0,
            fixed_raw_token_points=128,
        )
    if arm_id == "P08-BG":
        if (
            isinstance(global_resample_numerator_hz, bool)
            or isinstance(global_resample_denominator, bool)
            or not isinstance(global_resample_numerator_hz, int)
            or not isinstance(global_resample_denominator, int)
        ):
            raise ValueError("P08-BG requires integer target numerator and denominator")
        candidate = (global_resample_numerator_hz, global_resample_denominator)
        allowed = {(51_200, 3), (25_600, 1), (51_200, 1)}
        if candidate not in allowed or math.gcd(*candidate) != 1:
            raise ValueError("P08-BG target rate is outside the frozen three-rate grid")
        if duration_ms is not None:
            raise ValueError("P08-BG cannot set a physical duration")
        return ArmSpec(
            arm_id=arm_id,
            use_physical_duration=False,
            use_band_projection=False,
            use_prompt=False,
            physical_patch_duration_s=0.01,
            fixed_raw_token_points=256,
            global_resample_numerator_hz=candidate[0],
            global_resample_denominator=candidate[1],
        )
    if (
        duration_ms is not None
        or global_resample_numerator_hz is not None
        or global_resample_denominator is not None
    ):
        raise ValueError("P08-NC has no representation-selection candidate")
    return ArmSpec(
        arm_id="P08-NC",
        use_physical_duration=False,
        use_band_projection=False,
        use_prompt=False,
        physical_patch_duration_s=0.01,
        fixed_raw_token_points=128,
    )


def model_args(spec: ArmSpec, *, dropout: float = 0.1) -> SimpleNamespace:
    return SimpleNamespace(
        embedding="HSE_prompt",
        backbone="B_04_Dlinear",
        task_head="H_11_Unified_cla",
        training_stage="pretrain",
        use_physical_duration=spec.use_physical_duration,
        fixed_raw_token_points=spec.fixed_raw_token_points,
        physical_patch_duration_s=spec.physical_patch_duration_s,
        physical_patch_points=256,
        patch_size_L=256,
        patch_size_C=1,
        num_patches=32,
        output_dim=128,
        shared_band_hz=6000.0,
        use_band_projection=spec.use_band_projection,
        use_prompt=spec.use_prompt,
        prompt_dim=128,
        prompt_combination="add",
        prompt_reference_fs_hz=10000.0,
        prompt_reference_duration_s=0.01,
        freeze_prompts_in_finetuning=False,
        dropout=float(dropout),
        unified_num_classes=4,
        num_classes=4,
    )


def build_model(
    spec: ArmSpec,
    *,
    seed: int,
    device: torch.device,
    dropout: float = 0.1,
) -> Model:
    """Build an arm with paired shared-component initialization."""

    torch.manual_seed(int(seed))
    model = Model(model_args(spec, dropout=dropout), metadata=None)

    def reset_embedding_component(module: torch.nn.Module, component_seed: int) -> None:
        """Reset one component without depending on optional-module draw order."""

        torch.manual_seed(component_seed)
        with torch.no_grad():
            for child in module.modules():
                if isinstance(child, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        child.bias.zero_()
                elif isinstance(child, torch.nn.LayerNorm) and child.elementwise_affine:
                    child.weight.fill_(1.0)
                    child.bias.zero_()

    # Optional prompt modules consume random draws during construction.  Reset
    # every shared stochastic component from a component-specific seed so arm
    # comparisons remain exactly paired for a given experiment seed.
    reset_embedding_component(model.embedding.patch_encoder, int(seed) + 10_003)
    reset_embedding_component(model.embedding.band_encoder, int(seed) + 20_003)
    if not spec.use_band_projection:
        for parameter in model.embedding.band_encoder.parameters():
            parameter.requires_grad = False
    if spec.use_prompt:
        reset_embedding_component(model.embedding.prompt_encoder, int(seed) + 30_003)
        prompt_projection = getattr(model.embedding, "prompt_proj", None)
        if isinstance(prompt_projection, torch.nn.Module):
            reset_embedding_component(prompt_projection, int(seed) + 40_003)

    # DLinear intends an averaging initialization.  Make its otherwise-random
    # biases identical across arms and candidates.
    with torch.no_grad():
        for linear in (
            model.backbone.Linear_Seasonal,
            model.backbone.Linear_Trend,
        ):
            linear.weight.fill_(1.0 / 32.0)
            if linear.bias is not None:
                linear.bias.zero_()

    # Prompt construction consumes random draws only in P08-M.  Reset the
    # shared classifier from a component-specific seed to preserve arm pairing.
    torch.manual_seed(int(seed) + 100_003)
    model.task_head.classifier.reset_parameters()
    # Leave the global generator in an arm-independent state for paired
    # dropout and feature-view noise when training starts immediately.
    torch.manual_seed(int(seed) + 900_003)
    return model.to(device)


def pretraining_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    feature_noise_std: float = 0.1,
    classification_weight: float = 0.1,
    contrastive_weight: float = 0.4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match the existing hse_contrastive feature-view InfoNCE contract."""

    if logits.shape[0] != labels.shape[0] or features.shape[0] != labels.shape[0]:
        raise ValueError("pretraining batch dimensions do not match")
    if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
        raise FloatingPointError("non-finite pretraining output")
    ce = F.cross_entropy(logits, labels)
    second_view = features + torch.randn_like(features) * float(feature_noise_std)
    contrastive = InfoNCELoss(temperature=float(temperature))(
        torch.cat((features, second_view), dim=0)
    )
    if classification_weight < 0.0 or contrastive_weight < 0.0:
        raise ValueError("pretraining loss weights must be nonnegative")
    if classification_weight == 0.0 and contrastive_weight == 0.0:
        raise ValueError("at least one pretraining loss weight must be positive")
    loss = classification_weight * ce + contrastive_weight * contrastive
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite pretraining loss")
    return loss, {
        "classification_loss": float(ce.detach().cpu()),
        "contrastive_loss": float(contrastive.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }


__all__ = [
    "ARM_IDS",
    "ArmSpec",
    "arm_spec",
    "build_model",
    "model_args",
    "pretraining_loss",
]
