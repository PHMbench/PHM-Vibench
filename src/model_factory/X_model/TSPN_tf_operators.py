"""Finite differentiable time–frequency maps, with explicit coordinates.

All frequencies are cycles/sample. These are sampled, finite-support analyses,
not an inverse-transform API. No resizing, hidden boundary padding or conversion
between scale and frequency axes is performed. Fixed transforms backpropagate to
the input; their scientific definitions are not changed by the optimizer.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _positive_vector(values, name: str) -> Tensor:
    value = torch.as_tensor(values, dtype=torch.float64)
    if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError(f"{name} must be a finite positive vector.")
    return value


class STFTMap(nn.Module):
    """Unit-energy Hann-window complex STFT; win_length is independent of n_fft.

    center=False and unfold use only measured samples. Increasing n_fft without
    increasing win_length only refines the frequency grid.
    """
    def __init__(self, win_length: int, n_fft: int, hop_length: int):
        super().__init__()
        if not (4 <= win_length <= n_fft and 1 <= hop_length <= win_length):
            raise ValueError("Require 4 <= win_length <= n_fft and 1 <= hop <= win_length.")
        self.win_length, self.n_fft, self.hop_length = int(win_length), int(n_fft), int(hop_length)
        window = torch.hann_window(win_length, periodic=True, dtype=torch.float64)
        self.register_buffer("window", window / torch.linalg.vector_norm(window))
        self.register_buffer("frequency", torch.fft.rfftfreq(n_fft, dtype=torch.float64))
        self.row_count = n_fft // 2 + 1
        self.coordinate_kind = "frequency_cycles_per_sample"

    def coefficients(self, x: Tensor) -> Tensor:
        if x.shape[1] < self.win_length:
            raise ValueError("Input is shorter than the measured STFT window.")
        frames = x.transpose(1, 2).unfold(-1, self.win_length, self.hop_length)
        values = torch.fft.rfft(frames * self.window.to(x), n=self.n_fft, dim=-1)
        return values.transpose(-1, -2)

    def times(self, n: int, *, device=None, dtype=torch.float32) -> Tensor:
        return torch.arange(0, n-self.win_length+1, self.hop_length, device=device, dtype=dtype) + (self.win_length-1)/2

    def enbw(self) -> float:
        return float(self.window.square().sum()/self.window.sum().square())


class _FiniteKernelMap(nn.Module):
    """Valid cross-correlation with explicit conjugated analysis atoms."""
    def _set_kernels(self, kernels: Tensor, frequencies: Tensor, stride: int):
        if stride < 1:
            raise ValueError("stride must be positive.")
        self.stride = int(stride)
        self.radius = (kernels.shape[-1]-1)//2
        self.register_buffer("kernel_real", kernels.real.contiguous())
        self.register_buffer("kernel_imag", kernels.imag.contiguous())
        self.register_buffer("frequency", frequencies)
        self.row_count = kernels.shape[0]

    def coefficients(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        length = 2*self.radius+1
        if n < length:
            raise ValueError(f"Measured signal needs at least {length} samples for the configured kernel support.")
        signal = x.transpose(1, 2).reshape(b*c, 1, n)
        real = F.conv1d(signal, self.kernel_real.to(x)[:, None], stride=self.stride)
        imag = F.conv1d(signal, self.kernel_imag.to(x)[:, None], stride=self.stride)
        return torch.complex(real, imag).reshape(b, c, self.row_count, -1)

    def times(self, n: int, *, device=None, dtype=torch.float32) -> Tensor:
        return torch.arange(self.radius, n-self.radius, self.stride, device=device, dtype=dtype)


class CWTMap(_FiniteKernelMap):
    """Sampled CWT at explicit scales using Morlet or Mexican-hat mother wavelets.

    C[a,b] = sum_n x[n] conj(psi((n-b)/a))/sqrt(a).
    All scales use a declared finite maximum lag, with valid centers only. No
    per-scale discrete renormalization is applied: that would change this formula.
    Morlet includes its continuous zero-mean correction; truncation can leave a
    small residual DC response. 'frequency' is the mother-wavelet pseudo-frequency.
    """
    def __init__(self, scales, wavelet: str = "morlet", omega0: float = 6.,
                 truncate: float = 4., stride: int = 1):
        super().__init__()
        scales = _positive_vector(scales, "scales")
        if wavelet not in {"morlet", "mexican_hat"} or not math.isfinite(truncate) or truncate < 3:
            raise ValueError("Use morlet or mexican_hat, with truncate >= 3.")
        if wavelet == "morlet" and (not math.isfinite(omega0) or omega0 < 5):
            raise ValueError("Morlet omega0 must be at least 5 in this definition.")
        spectral_center = omega0 if wavelet == "morlet" else math.sqrt(2.)
        if ((spectral_center+3)/(2*math.pi*scales) >= .5).any():
            raise ValueError("Wavelet scales place the main spectral support too near/above Nyquist.")
        radius = math.ceil(float(scales.max())*truncate)
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        u = lag[None]/scales[:, None]
        if wavelet == "morlet":
            psi = math.pi**(-.25)*(torch.exp(1j*omega0*u)-math.exp(-omega0**2/2))*torch.exp(-u.square()/2)
        else:
            psi = (2/(math.sqrt(3)*math.pi**.25))*(1-u.square())*torch.exp(-u.square()/2)
            psi = torch.complex(psi, torch.zeros_like(psi))
        kernels = psi.conj()/scales[:, None].sqrt()
        self._set_kernels(kernels, spectral_center/(2*math.pi*scales), stride)
        self.register_buffer("scales", scales)
        self.wavelet, self.omega0, self.truncate = wavelet, omega0, truncate
        self.coordinate_kind = "scale_samples_with_pseudo_frequency"


class StockwellMap(_FiniteKernelMap):
    """Finite positive-frequency S transform, sampled at specified frequencies.

    S(b,f)=sum x[n] f/(k sqrt(2pi)) exp(-(n-b)^2 f^2/(2k^2)) exp(-i2pi f n).
    The absolute phase exp(-i2pi f b) is included. f=0, an inverse transform,
    complete-spectrum energy claims and periodic-boundary implementations are
    deliberately outside this selected-frequency forward operator.
    """
    def __init__(self, frequencies, width_factor: float = 1., truncate: float = 4., stride: int = 1):
        super().__init__()
        f = _positive_vector(frequencies, "frequencies")
        if (f >= .5).any() or width_factor <= 0 or truncate < 3 or not math.isfinite(width_factor+truncate):
            raise ValueError("Positive sub-Nyquist frequencies, width_factor > 0 and truncate >= 3 required.")
        sigma = width_factor/f
        if (f+3/(2*math.pi*sigma) >= .5).any():
            raise ValueError("Configured S-transform spectral support approaches Nyquist.")
        radius = math.ceil(float(sigma.max())*truncate)
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        kernels = torch.exp(-.5*(lag[None]/sigma[:, None]).square())/(math.sqrt(2*math.pi)*sigma[:, None])
        kernels = kernels*torch.exp(-2j*math.pi*f[:, None]*lag[None])
        self._set_kernels(kernels, f, stride)
        self.width_factor, self.truncate = float(width_factor), float(truncate)
        self.coordinate_kind = "frequency_cycles_per_sample"

    def coefficients(self, x: Tensor) -> Tensor:
        relative = super().coefficients(x)
        b = self.times(x.shape[1], device=x.device, dtype=x.dtype)
        phase = torch.exp(-2j*math.pi*self.frequency.to(x)[:, None]*b[None])
        return relative*phase[None, None]


class ChirpletMap(_FiniteKernelMap):
    """Gaussian-window chirplet analysis at one configured chirp rate.

    Atom: pi^(-1/4)/sqrt(sigma) exp(-r^2/(2sigma^2))
          exp(i2pi(f*r + rate*r^2/2)).
    Multiple named branches may use different rates. This is not synchrosqueezing.
    """
    def __init__(self, frequencies, window_sigma: float, chirp_rate: float = 0.,
                 truncate: float = 4., stride: int = 1):
        super().__init__()
        f = _positive_vector(frequencies, "frequencies")
        if window_sigma <= 0 or truncate < 3 or not all(map(math.isfinite, [window_sigma, chirp_rate, truncate])):
            raise ValueError("Invalid chirplet window or rate.")
        radius = math.ceil(window_sigma*truncate)
        spectral_margin = abs(chirp_rate)*radius+3/(2*math.pi*window_sigma)
        if (f-spectral_margin <= 0).any() or (f+spectral_margin >= .5).any():
            raise ValueError("Chirplet instantaneous-frequency support crosses DC or Nyquist.")
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        env = math.pi**(-.25)/math.sqrt(window_sigma)*torch.exp(-.5*(lag/window_sigma).square())
        kernels = env[None]*torch.exp(-2j*math.pi*(f[:, None]*lag[None]+chirp_rate*lag.square()[None]/2))
        self._set_kernels(kernels, f, stride)
        self.window_sigma, self.chirp_rate, self.truncate = float(window_sigma), float(chirp_rate), float(truncate)
        self.coordinate_kind = "center_frequency_cycles_per_sample_at_fixed_chirp_rate"


class TimeFrequencyReadout(nn.Module):
    """Ordered row groups and time bins -> log-power mean and centered RMS.

    A row group is a declared contiguous set of that operator's original rows,
    not an interpolation to another operator's coordinates. Centered RMS rather
    than variance admits a simple global feature-response bound.
    """
    def __init__(self, row_count: int, in_channels: int, time_bins: int,
                 log_floor: float, row_groups: int | None = None):
        super().__init__()
        groups = row_count if row_groups is None else int(row_groups)
        if not 1 <= groups <= row_count or time_bins < 1 or log_floor <= 0 or not math.isfinite(log_floor):
            raise ValueError("Invalid ordered readout partition or log floor.")
        self.row_groups, self.time_bins = groups, int(time_bins)
        self.output_dim = int(in_channels)*groups*time_bins*2
        self.register_buffer("log_floor", torch.tensor(log_floor, dtype=torch.float64))

    def forward(self, coeff: Tensor) -> Tensor:
        if coeff.shape[-1] < self.time_bins:
            raise ValueError("Not enough valid TF time positions for time_bins.")
        power = coeff.abs().square()
        grouped = torch.stack([part.mean(-2) for part in torch.tensor_split(power, self.row_groups, -2)], -2)
        log_power = torch.log(self.log_floor.to(power)+grouped)
        features = []
        for part in torch.tensor_split(log_power, self.time_bins, -1):
            mean = part.mean(-1)
            rms = torch.linalg.vector_norm(part-mean[..., None], dim=-1)/math.sqrt(part.shape[-1])
            features.append(torch.stack((mean, rms), -1))
        return torch.stack(features, -2).flatten(1)


class TimeFrequencyBranch(nn.Module):
    """A concrete configured analysis plus its readout; no runtime registry."""
    def __init__(self, kind: str, in_channels: int, transform: Mapping, readout: Mapping):
        super().__init__()
        if kind == "stft":
            self.transform = STFTMap(**transform)
        elif kind == "cwt":
            self.transform = CWTMap(**transform)
        elif kind == "stockwell":
            self.transform = StockwellMap(**transform)
        elif kind == "chirplet":
            self.transform = ChirpletMap(**transform)
        else:
            raise ValueError(f"Unknown TF operator {kind!r}; supported: stft, cwt, stockwell, chirplet.")
        self.readout = TimeFrequencyReadout(self.transform.row_count, in_channels, **readout)
        self.output_dim = self.readout.output_dim
        self.kind = kind

    def forward(self, x: Tensor) -> Tensor:
        return self.readout(self.transform.coefficients(x))
