import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.model_factory.ISFM.system_utils import normalize_fs


def _deterministic_start_indices(
    max_start: int,
    batch_size: int,
    num_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the historical sequential patch grid for deterministic evaluation."""
    step = max(
        1,
        max_start // (num_patches - 1) if num_patches > 1 else 1,
    )
    starts = torch.arange(
        0,
        min(max_start + 1, num_patches * step),
        step,
        device=device,
    )
    if len(starts) < num_patches:
        repeats = (num_patches + len(starts) - 1) // len(starts)
        starts = starts.repeat(repeats)[:num_patches]
    return starts.unsqueeze(0).expand(batch_size, -1)


class E_01_HSE(nn.Module):
    """
    Divide an input signal into HSE patches and mix them with linear layers.

    Training samples patch starts randomly. Evaluation uses deterministic,
    evenly-spaced starts unless an evaluator explicitly supplies starts. Inputs use
    shape ``[B, L, C]``. Patch sizes larger than the observed signal are invalid;
    HSE never repeats, pads, or duplicates scientific inputs.
    """

    def __init__(self, args):
        super(E_01_HSE, self).__init__()
        self.patch_size_L = args.patch_size_L
        self.patch_size_C = args.patch_size_C
        self.num_patches = args.num_patches
        self.output_dim = args.output_dim

        self.linear1 = nn.Linear(
            self.patch_size_L * (self.patch_size_C * 2),
            self.output_dim,
        )
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor, fs, **kwargs) -> torch.Tensor:
        """Return HSE features with shape ``[B, num_patches, output_dim]``."""
        if not torch.is_tensor(x) or x.ndim != 3:
            observed = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            raise ValueError(f"HSE input must have shape [B,L,C], got {observed}")

        B, L, C = x.size()
        device = x.device
        if self.patch_size_L > L:
            raise ValueError(
                "HSE patch_size_L exceeds the observed signal length: "
                f"patch_size_L={self.patch_size_L}, L={L}. Reduce patch_size_L "
                "or provide longer windows; HSE does not repeat or pad time."
            )
        if self.patch_size_C > C:
            raise ValueError(
                "HSE patch_size_C exceeds the observed channel count: "
                f"patch_size_C={self.patch_size_C}, C={C}. Reduce patch_size_C "
                "or provide the requested channels; HSE does not duplicate or "
                "pad channels."
            )

        fs_tensor = normalize_fs(
            fs,
            batch_size=B,
            device=device,
            as_column=True,
        )
        T = 1.0 / fs_tensor
        t = torch.arange(L, device=device, dtype=torch.float32).view(1, L)
        t = t * T

        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        explicit_L = kwargs.get("start_indices_L")
        explicit_C = kwargs.get("start_indices_C")
        if (explicit_L is None) != (explicit_C is None):
            raise ValueError(
                "start_indices_L and start_indices_C must be supplied together"
            )

        if explicit_L is None:
            if self.training:
                start_indices_L = torch.randint(
                    0,
                    max_start_L + 1,
                    (B, self.num_patches),
                    device=device,
                )
                start_indices_C = torch.randint(
                    0,
                    max_start_C + 1,
                    (B, self.num_patches),
                    device=device,
                )
            else:
                start_indices_L = _deterministic_start_indices(
                    max_start_L,
                    B,
                    self.num_patches,
                    device,
                )
                start_indices_C = _deterministic_start_indices(
                    max_start_C,
                    B,
                    self.num_patches,
                    device,
                )
        else:
            expected_shape = (B, self.num_patches)
            start_indices_L = torch.as_tensor(
                explicit_L,
                dtype=torch.long,
                device=device,
            )
            start_indices_C = torch.as_tensor(
                explicit_C,
                dtype=torch.long,
                device=device,
            )
            if tuple(start_indices_L.shape) != expected_shape:
                raise ValueError(
                    "start_indices_L must have shape "
                    f"{expected_shape}, observed={tuple(start_indices_L.shape)}"
                )
            if tuple(start_indices_C.shape) != expected_shape:
                raise ValueError(
                    "start_indices_C must have shape "
                    f"{expected_shape}, observed={tuple(start_indices_C.shape)}"
                )
            if (start_indices_L < 0).any() or (
                start_indices_L > max_start_L
            ).any():
                raise ValueError("start_indices_L contains an out-of-range start")
            if (start_indices_C < 0).any() or (
                start_indices_C > max_start_C
            ).any():
                raise ValueError("start_indices_C contains an out-of-range start")

        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)
        idx_L = start_indices_L.unsqueeze(-1) + offsets_L
        idx_C = start_indices_C.unsqueeze(-1) + offsets_C

        idx_L = idx_L.unsqueeze(-1)
        idx_C = idx_C.unsqueeze(-2)

        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(
            3,
            idx_C.expand(-1, -1, self.patch_size_L, -1),
        )

        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))
        t_patches = t_patches.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            self.patch_size_C,
        )

        patches = torch.cat([patches, t_patches], dim=-1)
        patches = rearrange(patches, "b p l c -> b p (l c)")
        out = self.linear1(patches)
        out = F.silu(out)
        return self.linear2(out)


class E_01_HSE_abalation(nn.Module):
    """
    Hierarchical Signal Embedding (HSE) module.

    Supports configurable sampling, mixing, linear depth, patch scale, and
    activation ablations while retaining the same no-repeat input boundary as
    the maintained HSE implementation.
    """

    def __init__(self, args, args_d):
        super(E_01_HSE_abalation, self).__init__()
        self.patch_size_L = args.patch_size_L
        self.patch_size_C = args.patch_size_C
        self.num_patches = args.num_patches
        self.output_dim = args.output_dim
        self.args_d = args_d

        self.sampling_mode = getattr(args, "sampling_mode", "random")
        self.apply_mixing = getattr(args, "apply_mixing", True)

        if hasattr(args, "linear_config"):
            if (
                isinstance(args.linear_config, (list, tuple))
                and len(args.linear_config) == 2
            ):
                self.linear_config = tuple(args.linear_config)
            else:
                self.linear_config = (1, 1)
        else:
            self.linear_config = (1, 1)

        if hasattr(args, "patch_scale"):
            if (
                isinstance(args.patch_scale, (list, tuple))
                and len(args.patch_scale) == 3
            ):
                self.patch_scale = tuple(args.patch_scale)
            else:
                self.patch_scale = (1, 1, 1)
        else:
            self.patch_scale = (1, 1, 1)

        self.patch_size_L *= self.patch_scale[0]
        self.patch_size_C *= self.patch_scale[1]
        self.num_patches *= self.patch_scale[2]

        self.activation_type = getattr(args, "activation_type", "silu")
        self.activation_fn = self._get_activation_fn(self.activation_type)
        self._setup_linear_layers()

    def _get_activation_fn(self, activation_type):
        activation_map = {
            "silu": F.silu,
            "relu": F.relu,
            "gelu": F.gelu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return activation_map.get(activation_type.lower(), F.silu)

    def _setup_linear_layers(self):
        layer1_depth, layer2_depth = self.linear_config

        if layer1_depth == 1:
            self.linear1 = nn.Linear(
                self.patch_size_L * (self.patch_size_C * 2),
                self.output_dim,
            )
        else:
            layers = [
                nn.Linear(
                    self.patch_size_L * (self.patch_size_C * 2),
                    self.output_dim,
                )
            ]
            for _ in range(layer1_depth - 1):
                layers.extend(
                    [
                        nn.LayerNorm(self.output_dim),
                        nn.Linear(self.output_dim, self.output_dim),
                    ]
                )
            self.linear1 = nn.Sequential(*layers)

        if not self.apply_mixing:
            self.linear2 = nn.Identity()
        elif layer2_depth == 1:
            self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        else:
            layers = [nn.Linear(self.output_dim, self.output_dim)]
            for _ in range(layer2_depth - 1):
                layers.extend(
                    [
                        nn.LayerNorm(self.output_dim),
                        nn.Linear(self.output_dim, self.output_dim),
                    ]
                )
            self.linear2 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, data_name) -> torch.Tensor:
        if not torch.is_tensor(x) or x.ndim != 3:
            observed = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            raise ValueError(f"HSE input must have shape [B,L,C], got {observed}")

        B, L, C = x.size()
        device = x.device
        if self.patch_size_L > L:
            raise ValueError(
                "HSE patch_size_L exceeds the observed signal length: "
                f"patch_size_L={self.patch_size_L}, L={L}. Reduce patch_size_L "
                "or provide longer windows; HSE does not repeat or pad time."
            )
        if self.patch_size_C > C:
            raise ValueError(
                "HSE patch_size_C exceeds the observed channel count: "
                f"patch_size_C={self.patch_size_C}, C={C}. Reduce patch_size_C "
                "or provide the requested channels; HSE does not duplicate or "
                "pad channels."
            )

        fs = self.args_d.task[data_name]["f_s"]
        T = 1.0 / fs
        t = torch.arange(L, device=device, dtype=torch.float32) * T
        t = t.unsqueeze(0).expand(B, -1)

        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C

        if self.sampling_mode == "random" and self.training:
            start_indices_L = torch.randint(
                0,
                max_start_L + 1,
                (B, self.num_patches),
                device=device,
            )
            start_indices_C = torch.randint(
                0,
                max_start_C + 1,
                (B, self.num_patches),
                device=device,
            )
        else:
            start_indices_L = _deterministic_start_indices(
                max_start_L,
                B,
                self.num_patches,
                device,
            )
            start_indices_C = _deterministic_start_indices(
                max_start_C,
                B,
                self.num_patches,
                device,
            )

        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)
        idx_L = start_indices_L.unsqueeze(-1) + offsets_L
        idx_C = start_indices_C.unsqueeze(-1) + offsets_C

        idx_L = idx_L.unsqueeze(-1)
        idx_C = idx_C.unsqueeze(-2)

        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(
            3,
            idx_C.expand(-1, -1, self.patch_size_L, -1),
        )

        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))
        t_patches = t_patches.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            self.patch_size_C,
        )

        patches = torch.cat([patches, t_patches], dim=-1)
        patches = rearrange(patches, "b p l c -> b p (l c)")
        out = self.linear1(patches)

        if self.apply_mixing:
            out = self.activation_fn(out)
            out = self.linear2(out)

        return out
