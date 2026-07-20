"""
Physical Layer Alignment for 1D-2D Fusion Fault Diagnosis

This module implements physical alignment mechanisms that ensure the 1D time-domain
and 2D frequency-domain representations maintain consistent physical properties,
specifically focusing on energy distribution and spectral characteristics.

Key concepts:
- Energy distribution consistency between time and frequency domains
- Spectral envelope alignment
- Physical constraint preservation (Parseval's theorem)
- Signal integrity verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class PhysicalAlignmentLoss(nn.Module):
    """
    Physical alignment loss that enforces consistency between 1D and 2D representations
    based on physical properties of signals.
    """

    def __init__(self,
                 energy_weight: float = 1.0,
                 spectral_weight: float = 0.5,
                 parseval_weight: float = 0.3,
                 eps: float = 1e-8):
        """
        Initialize physical alignment loss.

        Args:
            energy_weight: Weight for energy distribution alignment
            spectral_weight: Weight for spectral envelope alignment
            parseval_weight: Weight for Parseval's theorem consistency
            eps: Small epsilon to avoid numerical issues
        """
        super().__init__()
        self.energy_weight = energy_weight
        self.spectral_weight = spectral_weight
        self.parseval_weight = parseval_weight
        self.eps = eps

    def forward(self,
                feat_1d: torch.Tensor,
                feat_2d: torch.Tensor,
                signal_1d: Optional[torch.Tensor] = None,
                signal_2d: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute physical alignment loss.

        Args:
            feat_1d: 1D features [B, C1, L1] or [B, C1]
            feat_2d: 2D features [B, C2, H2, W2] or [B, C2]
            signal_1d: Original 1D signals [B, L] (optional)
            signal_2d: Original 2D representations [B, H, W] (optional)

        Returns:
            Dictionary containing individual loss components and total loss
        """
        losses = {}

        # Energy distribution alignment
        energy_loss = self._energy_distribution_alignment(feat_1d, feat_2d)
        losses['energy'] = self.energy_weight * energy_loss

        # Spectral envelope alignment (if we have spatial information)
        if feat_1d.dim() >= 3 and feat_2d.dim() >= 4:
            spectral_loss = self._spectral_envelope_alignment(feat_1d, feat_2d)
            losses['spectral'] = self.spectral_weight * spectral_loss
        else:
            losses['spectral'] = torch.tensor(0.0, device=feat_1d.device)

        # Parseval's theorem consistency (if original signals provided)
        if signal_1d is not None and signal_2d is not None:
            parseval_loss = self._parseval_consistency(signal_1d, signal_2d)
            losses['parseval'] = self.parseval_weight * parseval_loss
        else:
            losses['parseval'] = torch.tensor(0.0, device=feat_1d.device)

        # Total physical alignment loss
        losses['total'] = losses['energy'] + losses['spectral'] + losses['parseval']

        return losses

    def _energy_distribution_alignment(self,
                                     feat_1d: torch.Tensor,
                                     feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Align energy distribution between 1D and 2D features.
        Uses KL divergence between normalized energy distributions.
        """
        # Compute energy distributions
        energy_1d = torch.sum(feat_1d ** 2, dim=-1, keepdim=True)  # [B, C1, 1] or [B, C1]
        energy_2d = torch.sum(feat_2d ** 2, dim=(-2, -1), keepdim=True)  # [B, C2, 1, 1]

        # Normalize to probability distributions
        energy_1d_norm = energy_1d / (torch.sum(energy_1d, dim=1, keepdim=True) + self.eps)
        energy_2d_norm = energy_2d / (torch.sum(energy_2d, dim=1, keepdim=True) + self.eps)

        # Align dimensions by pooling/interpolation
        min_channels = min(energy_1d_norm.size(1), energy_2d_norm.size(1))
        energy_1d_aligned = F.adaptive_avg_pool1d(
            energy_1d_norm.transpose(1, 2), min_channels
        ).transpose(1, 2) if energy_1d_norm.dim() == 3 else energy_1d_norm[:, :min_channels]

        energy_2d_aligned = F.adaptive_avg_pool2d(
            energy_2d_norm, (min_channels, 1)
        ).squeeze(-1) if energy_2d_norm.dim() == 4 else energy_2d_norm[:, :min_channels]

        # Compute KL divergence (symmetrized)
        kl_1d_to_2d = F.kl_div(
            torch.log(energy_1d_aligned + self.eps),
            energy_2d_aligned,
            reduction='batchmean'
        )
        kl_2d_to_1d = F.kl_div(
            torch.log(energy_2d_aligned + self.eps),
            energy_1d_aligned,
            reduction='batchmean'
        )

        return 0.5 * (kl_1d_to_2d + kl_2d_to_1d)

    def _spectral_envelope_alignment(self,
                                   feat_1d: torch.Tensor,
                                   feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Align spectral envelopes between 1D and 2D representations.
        Uses correlation between frequency response characteristics.
        """
        # Compute frequency responses via FFT
        if feat_1d.dim() == 3:  # [B, C, L]
            freq_resp_1d = torch.abs(torch.fft.fft(feat_1d, dim=-1))
            # Take magnitude and average over channels
            spectrum_1d = torch.mean(freq_resp_1d, dim=1)  # [B, L]
        else:
            spectrum_1d = torch.abs(torch.fft.fft(feat_1d, dim=-1))

        if feat_2d.dim() == 4:  # [B, C, H, W]
            # Compute 2D FFT and radial averaging
            freq_resp_2d = torch.abs(torch.fft.fft2(feat_2d, dim=(-2, -1)))
            spectrum_2d = torch.mean(freq_resp_2d, dim=1)  # [B, H, W]

            # Radial averaging to get 1D spectrum
            H, W = spectrum_2d.shape[-2:]
            center_h, center_w = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=spectrum_2d.device),
                torch.arange(W, device=spectrum_2d.device),
                indexing='ij'
            )
            r = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            max_r = int(torch.max(r).item())

            # Radial bins
            radial_spectrum = []
            for radius in range(max_r):
                mask = (r >= radius) & (r < radius + 1)
                if torch.any(mask):
                    radial_val = torch.mean(spectrum_2d[mask], dim=-1)
                    radial_spectrum.append(radial_val)
                else:
                    radial_spectrum.append(torch.zeros_like(spectrum_1d[..., 0]))

            spectrum_2d = torch.stack(radial_spectrum, dim=-1)  # [B, max_r]
        else:
            spectrum_2d = torch.abs(torch.fft.fft(feat_2d, dim=-1))

        # Interpolate to same length
        min_length = min(spectrum_1d.size(-1), spectrum_2d.size(-1))
        spectrum_1d_interp = F.interpolate(
            spectrum_1d.unsqueeze(1), size=min_length, mode='linear', align_corners=False
        ).squeeze(1)
        spectrum_2d_interp = F.interpolate(
            spectrum_2d.unsqueeze(1), size=min_length, mode='linear', align_corners=False
        ).squeeze(1)

        # Compute correlation loss
        correlation = F.cosine_similarity(spectrum_1d_interp, spectrum_2d_interp, dim=-1)
        correlation_loss = 1.0 - torch.mean(correlation)

        return correlation_loss

    def _parseval_consistency(self,
                            signal_1d: torch.Tensor,
                            signal_2d: torch.Tensor) -> torch.Tensor:
        """
        Enforce Parseval's theorem consistency between time and frequency domains.
        Total energy should be preserved between domains.
        """
        # Compute energy in time domain
        energy_time = torch.sum(signal_1d ** 2, dim=-1)  # [B]

        # Compute energy in frequency domain
        energy_freq = torch.sum(signal_2d ** 2, dim=(-2, -1))  # [B]

        # Normalize and compute difference
        energy_time_norm = energy_time / (torch.mean(energy_time) + self.eps)
        energy_freq_norm = energy_freq / (torch.mean(energy_freq) + self.eps)

        # Ensure both have same shape for MSE computation
        if energy_time_norm.dim() == 1 and energy_freq_norm.dim() == 1:
            parseval_loss = F.mse_loss(energy_time_norm, energy_freq_norm)
        elif energy_time_norm.dim() == 0 and energy_freq_norm.dim() == 0:
            parseval_loss = F.mse_loss(energy_time_norm.unsqueeze(0), energy_freq_norm.unsqueeze(0))
        else:
            # Handle dimension mismatch by taking mean where necessary
            parseval_loss = F.mse_loss(
                energy_time_norm.flatten(),
                energy_freq_norm.flatten()
            )

        return parseval_loss


class PhysicalConstraintLayer(nn.Module):
    """
    Layer that applies physical constraints to features during forward pass.
    """

    def __init__(self, constraint_type: str = 'energy_normalization'):
        super().__init__()
        self.constraint_type = constraint_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to features."""
        if self.constraint_type == 'energy_normalization':
            return self._energy_normalize(features)
        elif self.constraint_type == 'spectral_whitening':
            return self._spectral_whiten(features)
        else:
            return features

    def _energy_normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize energy of features."""
        energy = torch.sum(features ** 2, dim=-1, keepdim=True)
        return features / (torch.sqrt(energy) + 1e-8)

    def _spectral_whiten(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spectral whitening to features."""
        # Compute mean and covariance
        mean = torch.mean(features, dim=-1, keepdim=True)
        features_centered = features - mean

        # Simple whitening (diagonal covariance)
        std = torch.std(features_centered, dim=-1, keepdim=True)
        return features_centered / (std + 1e-8)


def compute_physical_alignment_metrics(feat_1d: torch.Tensor,
                                     feat_2d: torch.Tensor) -> Dict[str, float]:
    """
    Compute physical alignment metrics for evaluation.

    Args:
        feat_1d: 1D features
        feat_2d: 2D features

    Returns:
        Dictionary of physical alignment metrics
    """
    metrics = {}

    try:
        # Energy correlation
        energy_1d = torch.sum(feat_1d ** 2, dim=-1)
        if feat_2d.dim() >= 3:
            energy_2d = torch.sum(feat_2d ** 2, dim=(-2, -1))
        else:
            energy_2d = torch.sum(feat_2d ** 2, dim=-1)

        # Ensure both are 1D for correlation computation
        if energy_1d.dim() > 1:
            energy_1d = energy_1d.flatten()
        if energy_2d.dim() > 1:
            energy_2d = energy_2d.flatten()

        # Align dimensions for comparison
        min_size = min(energy_1d.numel(), energy_2d.numel())
        if min_size > 0:
            energy_1d_aligned = energy_1d.flatten()[:min_size]
            energy_2d_aligned = energy_2d.flatten()[:min_size]

            # Reshape for cosine similarity (need batch dimension)
            energy_1d_aligned = energy_1d_aligned.unsqueeze(0)
            energy_2d_aligned = energy_2d_aligned.unsqueeze(0)

            energy_correlation = F.cosine_similarity(energy_1d_aligned, energy_2d_aligned, dim=-1)
            metrics['energy_correlation'] = energy_correlation.item()
        else:
            metrics['energy_correlation'] = 0.0
    except Exception as e:
        print(f"Warning: Could not compute energy correlation: {e}")
        metrics['energy_correlation'] = 0.0

    # Spectral coherence (simplified)
    if feat_1d.dim() >= 2 and feat_2d.dim() >= 3:
        # Compute frequency responses
        freq_1d = torch.abs(torch.fft.fft(feat_1d, dim=-1))
        freq_2d = torch.abs(torch.fft.fft2(feat_2d, dim=(-2, -1)))

        # Compare dominant frequencies
        dominant_freq_1d = torch.argmax(freq_1d, dim=-1).float()
        dominant_freq_2d = torch.argmax(torch.mean(freq_2d, dim=(-2, -1)), dim=-1).float()

        freq_alignment = 1.0 - torch.mean(torch.abs(dominant_freq_1d - dominant_freq_2d) /
                                         feat_1d.size(-1))
        metrics['frequency_alignment'] = freq_alignment.item()
    else:
        metrics['frequency_alignment'] = 0.0

    return metrics


if __name__ == "__main__":
    # Test the physical alignment loss
    batch_size, seq_len = 4, 1024
    channels_1d, channels_2d = 64, 32
    height, width = 32, 32

    # Create dummy features
    feat_1d = torch.randn(batch_size, channels_1d, seq_len)
    feat_2d = torch.randn(batch_size, channels_2d, height, width)
    signal_1d = torch.randn(batch_size, seq_len)
    signal_2d = torch.randn(batch_size, height, width)

    # Initialize loss
    loss_fn = PhysicalAlignmentLoss()

    # Compute loss
    losses = loss_fn(feat_1d, feat_2d, signal_1d, signal_2d)

    print("Physical Alignment Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")

    # Compute metrics
    metrics = compute_physical_alignment_metrics(feat_1d, feat_2d)
    print("\nPhysical Alignment Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")