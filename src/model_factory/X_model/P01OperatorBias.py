"""P01: TSPN-style physical operators with same-forward additive contributions.

Factory entry: Model(args, metadata=None). Input is (batch, time, channels).
This is a dedicated variant, not a replacement for TSPN or TSPN_UXFD.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn
import torch.nn.functional as F


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def center_classes(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def envelope(x: torch.Tensor) -> torch.Tensor:
    """Hilbert magnitude along time; keep odd and even signal lengths unchanged."""
    n = x.shape[1]
    multiplier = x.new_zeros(n)
    multiplier[0] = 1
    if n % 2 == 0:
        multiplier[1:n // 2] = 2
        multiplier[n // 2] = 1
    else:
        multiplier[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(torch.fft.fft(x, dim=1) * multiplier[None, :, None], dim=1)
    return analytic.abs()


def statistics(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """RMS, mean absolute amplitude and standard deviation; no per-sample fitting."""
    rms = (x.square().mean(dim=dim) + 1e-8).sqrt()
    return torch.cat((rms, x.abs().mean(dim=dim), x.var(dim=dim, unbiased=False).add(1e-8).sqrt()), dim=-1)


class PhysicalBands(nn.Module):
    """Gaussian frequency responses with bounded learned coordinates.

    Center and bandwidth use order units for 'order', Hz for 'hz'.
    Bounds come from a source-defined physical range, never the target batch.
    A band with center+3*width >= Nyquist is rejected, not silently truncated.
    """
    def __init__(self, centers, widths, center_bounds, width_bounds, coordinate):
        super().__init__()
        if coordinate not in {'order', 'hz'}:
            raise ValueError('coordinate must be order or hz')
        self.coordinate = coordinate
        low, high = map(float, center_bounds)
        wlow, whigh = map(float, width_bounds)
        if not (0 < low < high and 0 < wlow < whigh):
            raise ValueError('band bounds must be positive increasing pairs')
        centers = torch.as_tensor(centers, dtype=torch.float32)
        widths = torch.as_tensor(widths, dtype=torch.float32)
        if centers.ndim != 1 or centers.numel() == 0 or widths.shape != centers.shape:
            raise ValueError('centers and widths must be nonempty equal-length vectors')
        if not ((centers > low).all() and (centers < high).all() and
                (widths > wlow).all() and (widths < whigh).all()):
            raise ValueError('initial bands must lie strictly inside configured bounds')
        self.register_buffer('center_limits', torch.tensor([low, high]))
        self.register_buffer('width_limits', torch.tensor([wlow, whigh]))
        self.center_raw = nn.Parameter(torch.logit((centers - low) / (high - low)))
        self.width_raw = nn.Parameter(torch.logit((widths - wlow) / (whigh - wlow)))

    def coordinates(self):
        lo, hi = self.center_limits
        wl, wh = self.width_limits
        return lo + (hi - lo) * self.center_raw.sigmoid(), wl + (wh - wl) * self.width_raw.sigmoid()

    def physical(self, fs, speed_hz):
        center, width = self.coordinates()
        scale = speed_hz if self.coordinate == 'order' else torch.ones_like(speed_hz)
        fc, bw = scale[:, None] * center[None], scale[:, None] * width[None]
        if bool(((fc - 3 * bw <= 0) | (fc + 3 * bw >= fs[:, None] / 2)).any()):
            raise ValueError('physical band exceeds observable support: require 0 < fc-3*bw < fc+3*bw < fs/2')
        return fc, bw

    def response(self, n, fs, speed_hz):
        fc, bw = self.physical(fs, speed_hz)
        frequency = torch.fft.rfftfreq(n, device=fs.device, dtype=fs.dtype)[None, :, None] * fs[:, None, None]
        return torch.exp(-0.5 * ((frequency - fc[:, None]) / bw[:, None]).square())

    def forward(self, x, fs, speed_hz):
        n = x.shape[1]
        response = self.response(n, fs, speed_hz)
        spectrum = torch.fft.rfft(x, dim=1)
        filtered = torch.fft.irfft(spectrum[:, :, :, None] * response[:, :, None, :], n=n, dim=1)
        return filtered.flatten(2)  # B,L,(channel,band), fixed ordering


class Model(nn.Module):
    """Order bands + fixed-Hz envelope bands + frequency-resolved STFT + raw.

    binding='physical' is the proposed structure. 'fixed' fixes periodic bands
    at reference speed. 'wrong_resonance' additionally scales the carrier with
    speed: a deliberately incorrect mechanism control, never the main model.
    """
    requires_physical_metadata = True

    def __init__(self, args, metadata=None):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.num_classes = int(_get(args, 'num_classes', 0))
        self.in_channels = int(_get(args, 'in_channels', 1))
        self.binding = str(_get(args, 'binding', 'physical'))
        self.head = str(_get(args, 'head', 'additive'))
        self.reference_speed_hz = float(_get(args, 'reference_speed_hz', 10.0))
        self.n_fft = int(_get(args, 'n_fft', 128))
        self.hop = int(_get(args, 'hop_length', self.n_fft // 4))
        self.rpm_field = str(_get(args, 'rpm_field', 'rotation_speed_rpm'))
        self.fs_field = str(_get(args, 'fs_field', 'sample_rate_hz'))
        self.paths = tuple(_get(args, 'paths', ['raw', 'periodic', 'envelope', 'stft']))
        if self.num_classes < 2 or self.in_channels < 1:
            raise ValueError('num_classes >= 2 and in_channels >= 1 are required')
        if self.binding not in {'physical', 'fixed', 'wrong_resonance'} or self.head not in {'additive', 'mlp'}:
            raise ValueError('invalid binding or head')
        if not self.paths or len(set(self.paths)) != len(self.paths) or set(self.paths) - {'raw', 'periodic', 'envelope', 'stft'}:
            raise ValueError('paths must be unique members of raw, periodic, envelope, stft')
        if self.reference_speed_hz <= 0 or not (2 <= self.n_fft and 0 < self.hop <= self.n_fft):
            raise ValueError('invalid reference speed or STFT dimensions')
        q = _get(args, 'order_centers', [3.0, 5.0, 7.0])
        bw = _get(args, 'order_widths', [0.3, 0.3, 0.3])
        qb = _get(args, 'order_center_bounds', [1.5, 8.0])
        wb = _get(args, 'order_width_bounds', [0.05, 0.4])
        self.bands = nn.ModuleDict()
        for path in self.paths:
            if path != 'raw':
                self.bands[path] = PhysicalBands(q, bw, qb, wb, 'order')
        if 'envelope' in self.paths:
            self.carrier = PhysicalBands(
                _get(args, 'carrier_centers_hz', [320.0]),
                _get(args, 'carrier_widths_hz', [35.0]),
                _get(args, 'carrier_center_bounds_hz', [260.0, 380.0]),
                _get(args, 'carrier_width_bounds_hz', [10.0, 40.0]), 'hz')
        band_count = len(q)
        carrier_count = len(_get(args, 'carrier_centers_hz', [320.0]))
        widths = {'raw': 6 * self.in_channels,
                  'periodic': 3 * self.in_channels * band_count,
                  'envelope': 3 * self.in_channels * carrier_count * band_count,
                  'stft': 3 * self.in_channels * band_count}
        # Deployable metadata are made equally available to every ablation.
        self.feature_widths = {p: widths[p] + 2 for p in self.paths}
        if self.head == 'additive':
            self.readouts = nn.ModuleDict({p: nn.Linear(self.feature_widths[p], self.num_classes, bias=False) for p in self.paths})
            self.bias = nn.Parameter(torch.zeros(self.num_classes))
        else:
            hidden = int(_get(args, 'hidden_dim', 32))
            self.classifier = nn.Sequential(nn.Linear(sum(self.feature_widths.values()), hidden), nn.ReLU(), nn.Linear(hidden, self.num_classes))
        self.register_buffer('stft_window', torch.hann_window(self.n_fft))

    def _physical_metadata(self, x, data_id, explicit):
        batch = x.shape[0]
        values = {}
        for name, column in [('sample_rate_hz', self.fs_field), ('rotation_speed_rpm', self.rpm_field)]:
            if explicit is not None and name in explicit:
                value = explicit[name]
            else:
                if self.metadata is None or data_id is None:
                    raise ValueError(f'{name} must be supplied per sample or resolved from declared metadata column')
                ids = data_id.detach().cpu().reshape(-1).tolist() if isinstance(data_id, torch.Tensor) else list(data_id)
                if len(ids) != batch:
                    raise ValueError('one file_id per sample is required; file IDs are not broadcast')
                value = [self.metadata[i][column] for i in ids]
            tensor = torch.as_tensor(value, dtype=x.dtype, device=x.device).reshape(-1)
            if tensor.numel() != batch or not torch.isfinite(tensor).all() or (tensor <= 0).any():
                raise ValueError(f'{name} must contain {batch} finite positive per-sample values')
            values[name] = tensor
        return values['sample_rate_hz'], values['rotation_speed_rpm'] / 60.0

    def _features(self, x, fs, speed):
        periodic_speed = speed if self.binding != 'fixed' else torch.full_like(speed, self.reference_speed_hz)
        metadata_features = torch.stack((torch.log(fs / 1000.0), torch.log(speed / self.reference_speed_hz)), dim=-1)
        result = {}
        for path in self.paths:
            if path == 'raw':
                # Two temporal segments retain coarse timing rather than only global energy.
                features = torch.cat([statistics(piece) for piece in torch.tensor_split(x, 2, dim=1)], dim=-1)
            elif path == 'periodic':
                features = statistics(self.bands[path](x, fs, periodic_speed))
            elif path == 'envelope':
                if self.binding == 'wrong_resonance':
                    fc, bw = self.carrier.physical(fs, speed)
                    fc = fc * (speed / self.reference_speed_hz)[:, None]
                    bw = bw * (speed / self.reference_speed_hz)[:, None]
                    if ((fc - 3*bw <= 0) | (fc + 3*bw >= fs[:, None]/2)).any():
                        raise ValueError('wrong-resonance control exceeds observable support')
                    f = torch.fft.rfftfreq(x.shape[1], device=x.device, dtype=x.dtype)[None, :, None] * fs[:, None, None]
                    h = torch.exp(-0.5*((f-fc[:, None])/bw[:, None]).square())
                    carrier = torch.fft.irfft(torch.fft.rfft(x, dim=1)[:, :, :, None]*h[:, :, None], n=x.shape[1], dim=1).flatten(2)
                else:
                    carrier = self.carrier(x, fs, speed)
                env = envelope(carrier)
                env = env - env.mean(dim=1, keepdim=True)  # remove DC before order analysis
                features = statistics(self.bands[path](env, fs, periodic_speed))
            else:
                if x.shape[1] < self.n_fft:
                    raise ValueError('signal shorter than n_fft; do not silently pad or resize')
                b, n, c = x.shape
                spectrum = torch.stft(x.transpose(1, 2).reshape(b*c, n), n_fft=self.n_fft,
                                      hop_length=self.hop, window=self.stft_window,
                                      center=False, normalized=True, return_complex=True)
                power = spectrum.abs().square().reshape(b, c, self.n_fft//2+1, -1)
                response = self.bands[path].response(self.n_fft, fs, periodic_speed)
                # Keep learned physical band index and frame statistics; never average F and T into one scalar.
                band_frames = torch.einsum('bcft,bfk->btck', power, response).flatten(2)
                features = statistics(torch.log1p(band_frames))
            result[path] = torch.cat((torch.asinh(features), metadata_features), dim=-1)
        return result

    def forward_details(self, x, data_id=None, task_id=None, *, physical_metadata=None):
        del task_id
        if x.ndim != 3 or x.shape[2] != self.in_channels or x.shape[1] < 4:
            raise ValueError(f'expected B,L,{self.in_channels} input, got {tuple(x.shape)}')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('signal must be finite floating point')
        fs, speed = self._physical_metadata(x, data_id, physical_metadata)
        features = self._features(x, fs, speed)
        if self.head == 'mlp':
            return {'logits': self.classifier(torch.cat(list(features.values()), dim=-1)), 'features': features}
        path_values = [center_classes(self.readouts[p](features[p])) for p in self.paths]
        bias = center_classes(self.bias).expand(x.shape[0], -1)
        contributions = torch.stack([bias, *path_values], dim=1)
        return {'logits': contributions.sum(dim=1), 'contributions': contributions,
                'path_names': ('bias', *self.paths), 'features': features}

    def forward(self, x, data_id=None, task_id=None, *, physical_metadata=None):
        return self.forward_details(x, data_id, task_id, physical_metadata=physical_metadata)['logits']
