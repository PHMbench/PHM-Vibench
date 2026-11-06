"""Finite time–frequency analyses with optionally source-learned physical coordinates.

The original fixed-operator API and state_dict remain unchanged by default.
Learnable variants are global source-trained maps with predeclared parameter
bounds, fixed support geometry, and a frozen map at evaluation; no per-sample
router or target-time adaptation is implemented. Frequencies are cycles/sample.
A parameterized but non-trainable transform uses the identical kernel and
support as its trainable counterpart. Initial buffers (frequency, window,
scales and kernels) retain their checkpoint meaning. Read learned coordinates
through the effective/current accessors after restoring the selected model.
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


def _learning_mode(learnable, trainable):
    if type(learnable) is not bool or type(trainable) is not bool:
        raise TypeError("learnable and trainable must be YAML booleans.")
    return learnable


def _bounded_initial(value, bounds, name):
    """Logit-coordinate initialization strictly inside predeclared bounds."""
    initial = torch.as_tensor(value, dtype=torch.float64)
    interval = torch.as_tensor(bounds, dtype=torch.float64)
    if (interval.shape != initial.shape + (2,) or not torch.isfinite(interval).all()
            or not torch.isfinite(initial).all() or not torch.all(interval[..., 1] > interval[..., 0])
            or not torch.all((initial > interval[..., 0]) & (initial < interval[..., 1]))):
        raise ValueError(f"{name} needs finite bounds strictly containing each initial value.")
    ratio = (initial - interval[..., 0]) / (interval[..., 1] - interval[..., 0])
    return torch.logit(ratio), interval


def _bounded_value(parameter: Tensor, bounds: Tensor) -> Tensor:
    return bounds[..., 0] + torch.sigmoid(parameter) * (bounds[..., 1] - bounds[..., 0])


class STFTMap(nn.Module):
    """Unit-energy Hann-window complex STFT; win_length is independent of n_fft.

    center=False and unfold use only measured samples. Increasing n_fft without
    increasing win_length only refines the frequency grid.
    """
    def __init__(self, win_length: int, n_fft: int, hop_length: int, *,
                 learnable: bool = False, trainable: bool = True, window_exponent: float = 1.,
                 window_exponent_bounds=None):
        super().__init__()
        if not (4 <= win_length <= n_fft and 1 <= hop_length <= win_length):
            raise ValueError("Require 4 <= win_length <= n_fft and 1 <= hop <= win_length.")
        self.win_length, self.n_fft, self.hop_length = int(win_length), int(n_fft), int(hop_length)
        window = torch.hann_window(win_length, periodic=True, dtype=torch.float64)
        self.register_buffer("window", window / torch.linalg.vector_norm(window))
        self.register_buffer("frequency", torch.fft.rfftfreq(n_fft, dtype=torch.float64))
        self.row_count = n_fft // 2 + 1
        self.coordinate_kind = "frequency_cycles_per_sample"
        self.learnable = _learning_mode(learnable, trainable)
        if self.learnable:
            param, bounds = _bounded_initial(window_exponent, window_exponent_bounds, "window_exponent")
            if not torch.all(bounds > 0):
                raise ValueError("Window exponent bounds must be positive.")
            self.window_exponent_parameter = nn.Parameter(param, requires_grad=trainable)
            self.register_buffer("window_exponent_bounds", bounds)
        elif window_exponent != 1. or window_exponent_bounds is not None:
            raise ValueError("Nondefault STFT window shape requires learnable=True and explicit bounds.")

    def effective_window_exponent(self):
        if not self.learnable:
            return self.window.new_tensor(1.)
        return _bounded_value(self.window_exponent_parameter, self.window_exponent_bounds)

    def current_frequency(self):
        return self.frequency

    def _current_window(self):
        if not self.learnable:
            return self.window
        # The constant normalization of the stored Hann cancels after raising
        # it to a positive exponent and normalizing the result to unit energy.
        w = self.window.pow(self.effective_window_exponent())
        return w / torch.linalg.vector_norm(w)

    def coefficients(self, x: Tensor) -> Tensor:
        if x.shape[1] < self.win_length:
            raise ValueError("Input is shorter than the measured STFT window.")
        frames = x.transpose(1, 2).unfold(-1, self.win_length, self.hop_length)
        values = torch.fft.rfft(frames * self._current_window().to(x), n=self.n_fft, dim=-1)
        return values.transpose(-1, -2)

    def times(self, n: int, *, device=None, dtype=torch.float32) -> Tensor:
        return torch.arange(0, n-self.win_length+1, self.hop_length, device=device, dtype=dtype) + (self.win_length-1)/2

    @torch.no_grad()
    def enbw(self) -> float:
        window = self._current_window()
        return float(window.square().sum()/window.sum().square())


class _FiniteKernelMap(nn.Module):
    """Valid cross-correlation with explicit conjugated analysis atoms."""
    def _set_kernels(self, kernels: Tensor, frequencies: Tensor, stride: int, lag: Tensor):
        if stride < 1:
            raise ValueError("stride must be positive.")
        self.stride = int(stride)
        self.radius = (kernels.shape[-1]-1)//2
        self.register_buffer("kernel_real", kernels.real.contiguous())
        self.register_buffer("kernel_imag", kernels.imag.contiguous())
        self.register_buffer("frequency", frequencies)
        self.row_count = kernels.shape[0]
        if self.learnable:
            # Bounds fix the sampled lag grid for every fit and its matched
            # frozen control. The grid follows module device/dtype, not a batch.
            self.register_buffer("_lag", lag, persistent=False)

    def _effective_kernels(self, x: Tensor):
        return self.kernel_real.to(x), self.kernel_imag.to(x)

    def coefficients(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        length = 2*self.radius+1
        if n < length:
            raise ValueError(f"Measured signal needs at least {length} samples for the configured kernel support.")
        signal = x.transpose(1, 2).reshape(b*c, 1, n)
        kernel_real, kernel_imag = self._effective_kernels(x)
        real = F.conv1d(signal, kernel_real[:, None], stride=self.stride)
        imag = F.conv1d(signal, kernel_imag[:, None], stride=self.stride)
        return torch.complex(real, imag).reshape(b, c, self.row_count, -1)

    def times(self, n: int, *, device=None, dtype=torch.float32) -> Tensor:
        return torch.arange(self.radius, n-self.radius, self.stride, device=device, dtype=dtype)


def _cwt_kernel(scales, lag, wavelet, omega0):
    u = lag[None] / scales[:, None]
    if wavelet == "morlet":
        psi = math.pi**(-.25) * (torch.exp(1j*omega0*u) - math.exp(-omega0**2/2)) * torch.exp(-u.square()/2)
    else:
        psi = (2/(math.sqrt(3)*math.pi**.25)) * (1-u.square()) * torch.exp(-u.square()/2)
        psi = torch.complex(psi, torch.zeros_like(psi))
    return psi.conj() / scales[:, None].sqrt()


class CWTMap(_FiniteKernelMap):
    """Sampled CWT at explicit scales using Morlet or Mexican-hat mother wavelets.

    C[a,b] = sum_n x[n] conj(psi((n-b)/a))/sqrt(a).
    All scales use a declared finite maximum lag, with valid centers only. No
    per-scale discrete renormalization is applied: that would change this formula.
    Morlet includes its continuous zero-mean correction; truncation can leave a
    small residual DC response. 'frequency' is the mother-wavelet pseudo-frequency.
    """
    def __init__(self, scales, wavelet: str = "morlet", omega0: float = 6.,
                 truncate: float = 4., stride: int = 1, *, learnable: bool = False,
                 trainable: bool = True, scale_bounds=None):
        super().__init__()
        scales = _positive_vector(scales, "scales")
        if wavelet not in {"morlet", "mexican_hat"} or not math.isfinite(truncate) or truncate < 3:
            raise ValueError("Use morlet or mexican_hat, with truncate >= 3.")
        if wavelet == "morlet" and (not math.isfinite(omega0) or omega0 < 5):
            raise ValueError("Morlet omega0 must be at least 5 in this definition.")
        spectral_center = omega0 if wavelet == "morlet" else math.sqrt(2.)
        if ((spectral_center+3)/(2*math.pi*scales) >= .5).any():
            raise ValueError("Wavelet scales place the main spectral support too near/above Nyquist.")
        self.learnable = _learning_mode(learnable, trainable)
        if self.learnable:
            param, bounds = _bounded_initial(scales, scale_bounds, "CWT scales")
            if (bounds[..., 0] <= 0).any() or ((spectral_center+3)/(2*math.pi*bounds[..., 0]) >= .5).any():
                raise ValueError("CWT scale bounds violate positivity or Nyquist.")
            self.scale_parameter = nn.Parameter(param, requires_grad=trainable)
            self.register_buffer("scale_bounds", bounds)
            max_support_scale = float(bounds[..., 1].max())
        else:
            if scale_bounds is not None:
                raise ValueError("CWT scale_bounds require learnable=True.")
            max_support_scale = float(scales.max())
        radius = math.ceil(max_support_scale*truncate)
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        kernels = _cwt_kernel(scales, lag, wavelet, omega0)
        self._set_kernels(kernels, spectral_center/(2*math.pi*scales), stride, lag)
        self.register_buffer("scales", scales)
        self.wavelet, self.omega0, self.truncate = wavelet, omega0, truncate
        self.coordinate_kind = "scale_samples_with_pseudo_frequency"

    def effective_scales(self):
        return (_bounded_value(self.scale_parameter, self.scale_bounds)
                if self.learnable else self.scales)

    def current_frequency(self):
        center = self.omega0 if self.wavelet == "morlet" else math.sqrt(2.)
        return center/(2*math.pi*self.effective_scales())

    def _effective_kernels(self, x):
        if not self.learnable:
            return super()._effective_kernels(x)
        kernels = _cwt_kernel(self.effective_scales(), self._lag, self.wavelet, self.omega0)
        return kernels.real.to(x), kernels.imag.to(x)


def _stockwell_kernel(f, lag, width_factor):
    sigma = width_factor/f
    window = torch.exp(-.5*(lag[None]/sigma[:, None]).square())/(math.sqrt(2*math.pi)*sigma[:, None])
    return window*torch.exp(-2j*math.pi*f[:, None]*lag[None])


class StockwellMap(_FiniteKernelMap):
    """Finite positive-frequency S transform, sampled at specified frequencies.

    S(b,f)=sum x[n] f/(k sqrt(2pi)) exp(-(n-b)^2 f^2/(2k^2)) exp(-i2pi f n).
    The absolute phase exp(-i2pi f b) is included. f=0, an inverse transform,
    complete-spectrum energy claims and periodic-boundary implementations are
    deliberately outside this selected-frequency forward operator.
    """
    def __init__(self, frequencies, width_factor: float = 1., truncate: float = 4.,
                 stride: int = 1, *, learnable: bool = False,
                 trainable: bool = True, frequency_bounds=None):
        super().__init__()
        f = _positive_vector(frequencies, "frequencies")
        if (f >= .5).any() or width_factor <= 0 or truncate < 3 or not math.isfinite(width_factor+truncate):
            raise ValueError("Positive sub-Nyquist frequencies, width_factor > 0 and truncate >= 3 required.")
        sigma = width_factor/f
        if (f+3/(2*math.pi*sigma) >= .5).any():
            raise ValueError("Configured S-transform spectral support approaches Nyquist.")
        self.learnable = _learning_mode(learnable, trainable)
        if self.learnable:
            param, bounds = _bounded_initial(f, frequency_bounds, "Stockwell frequencies")
            if (bounds[..., 0] <= 0).any() or (bounds[..., 1]*(1+3/(2*math.pi*width_factor)) >= .5).any():
                raise ValueError("Stockwell frequency bounds violate positivity or Nyquist.")
            self.frequency_parameter = nn.Parameter(param, requires_grad=trainable)
            self.register_buffer("frequency_bounds", bounds)
            max_sigma = float((width_factor / bounds[..., 0]).max())
        else:
            if frequency_bounds is not None:
                raise ValueError("Stockwell frequency_bounds require learnable=True.")
            max_sigma = float(sigma.max())
        radius = math.ceil(max_sigma*truncate)
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        kernels = _stockwell_kernel(f, lag, width_factor)
        self._set_kernels(kernels, f, stride, lag)
        self.width_factor, self.truncate = float(width_factor), float(truncate)
        self.coordinate_kind = "frequency_cycles_per_sample"

    def current_frequency(self):
        return (_bounded_value(self.frequency_parameter, self.frequency_bounds)
                if self.learnable else self.frequency)

    def _effective_kernels(self, x):
        if not self.learnable:
            return super()._effective_kernels(x)
        kernels = _stockwell_kernel(self.current_frequency(), self._lag, self.width_factor)
        return kernels.real.to(x), kernels.imag.to(x)

    def coefficients(self, x: Tensor) -> Tensor:
        relative = super().coefficients(x)
        b = self.times(x.shape[1], device=x.device, dtype=x.dtype)
        phase = torch.exp(-2j*math.pi*self.current_frequency().to(x)[:, None]*b[None])
        return relative*phase[None, None]


def _chirplet_kernel(f, rate, lag, sigma):
    env = math.pi**(-.25)/math.sqrt(sigma)*torch.exp(-.5*(lag/sigma).square())
    return env[None]*torch.exp(-2j*math.pi*(f[:, None]*lag[None]+rate*lag.square()[None]/2))


class ChirpletMap(_FiniteKernelMap):
    """Gaussian-window chirplet analysis at one configured chirp rate.

    Atom: pi^(-1/4)/sqrt(sigma) exp(-r^2/(2sigma^2))
          exp(i2pi(f*r + rate*r^2/2)).
    Multiple named branches may use different rates. This is not synchrosqueezing.
    """
    def __init__(self, frequencies, window_sigma: float, chirp_rate: float = 0.,
                 truncate: float = 4., stride: int = 1, *, learnable: bool = False,
                 trainable: bool = True, frequency_bounds=None, chirp_rate_bounds=None):
        super().__init__()
        f = _positive_vector(frequencies, "frequencies")
        if window_sigma <= 0 or truncate < 3 or not all(map(math.isfinite, [window_sigma, chirp_rate, truncate])):
            raise ValueError("Invalid chirplet window or rate.")
        radius = math.ceil(window_sigma*truncate)
        spectral_margin = abs(chirp_rate)*radius+3/(2*math.pi*window_sigma)
        if (f-spectral_margin <= 0).any() or (f+spectral_margin >= .5).any():
            raise ValueError("Chirplet instantaneous-frequency support crosses DC or Nyquist.")
        self.learnable = _learning_mode(learnable, trainable)
        if self.learnable:
            fp, fb = _bounded_initial(f, frequency_bounds, "chirplet frequencies")
            cp, cb = _bounded_initial(chirp_rate, chirp_rate_bounds, "chirplet chirp rate")
            extreme_rate = float(cb.abs().max())
            margin = extreme_rate*radius+3/(2*math.pi*window_sigma)
            if (fb[..., 0] <= margin).any() or (fb[..., 1] >= .5-margin).any():
                raise ValueError("Chirplet parameter bounds violate DC/Nyquist support.")
            self.frequency_parameter = nn.Parameter(fp, requires_grad=trainable)
            self.register_buffer("frequency_bounds", fb)
            self.chirp_rate_parameter = nn.Parameter(cp, requires_grad=trainable)
            self.register_buffer("chirp_rate_bounds", cb)
        elif frequency_bounds is not None or chirp_rate_bounds is not None:
            raise ValueError("Chirplet parameter bounds require learnable=True.")
        lag = torch.arange(-radius, radius+1, dtype=torch.float64)
        kernels = _chirplet_kernel(f, chirp_rate, lag, window_sigma)
        self._set_kernels(kernels, f, stride, lag)
        self.window_sigma, self.chirp_rate, self.truncate = float(window_sigma), float(chirp_rate), float(truncate)
        self.coordinate_kind = ("center_frequency_cycles_per_sample_at_bounded_chirp_rate"
                                if self.learnable else "center_frequency_cycles_per_sample_at_fixed_chirp_rate")

    def current_frequency(self):
        return (_bounded_value(self.frequency_parameter, self.frequency_bounds)
                if self.learnable else self.frequency)

    def current_chirp_rate(self):
        return (_bounded_value(self.chirp_rate_parameter, self.chirp_rate_bounds)
                if self.learnable else self.chirp_rate)

    def _effective_kernels(self, x):
        if not self.learnable:
            return super()._effective_kernels(x)
        kernels = _chirplet_kernel(self.current_frequency(), self.current_chirp_rate(),
                                   self._lag, self.window_sigma)
        return kernels.real.to(x), kernels.imag.to(x)


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
