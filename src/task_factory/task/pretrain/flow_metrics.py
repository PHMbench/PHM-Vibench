"""
Flow Metrics Module for Generation Quality Assessment

This module provides comprehensive metrics for evaluating Flow model generation 
quality, training performance, and signal fidelity in industrial vibration 
signal analysis tasks.

Author: PHM-Vibench Team
Date: 2025-09-02
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import psutil
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft, fftfreq
# Optional imports with fallbacks
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("⚠️  seaborn not available, some visualizations will be simplified")

try:
    from sklearn.metrics import pairwise_distances
except ImportError:
    pairwise_distances = None
    print("⚠️  sklearn not available, some metrics will be simplified")
import warnings

# Suppress matplotlib warnings in non-GUI environments
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class FlowMetrics(nn.Module):
    """
    Comprehensive metrics suite for Flow model evaluation.
    
    Provides generation quality assessment, training performance monitoring,
    and signal analysis capabilities specifically designed for industrial
    vibration signal analysis with Flow generative models.
    
    Key Features:
    - Loss tracking and convergence monitoring
    - Statistical similarity assessment (KS test, distribution comparison)
    - Spectral analysis and frequency domain comparison
    - Signal quality metrics (SNR, distortion measures)
    - Sample diversity and coverage assessment
    - Real-time performance monitoring (speed, memory usage)
    - Visualization utilities for generated vs real samples
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        enable_visualization: bool = True,
        save_plots: bool = False,
        plot_dir: str = "./flow_metrics_plots",
        track_memory: bool = True,
        track_gradients: bool = True
    ):
        """
        Initialize FlowMetrics.
        
        Parameters
        ----------
        device : Optional[torch.device]
            Computation device. If None, uses CUDA if available
        enable_visualization : bool
            Whether to enable plot generation
        save_plots : bool
            Whether to save plots to disk
        plot_dir : str
            Directory for saving plots
        track_memory : bool
            Whether to track GPU memory usage
        track_gradients : bool
            Whether to track gradient statistics
        """
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_visualization = enable_visualization
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.track_memory = track_memory
        self.track_gradients = track_gradients
        
        # Initialize metric storage
        self.reset_metrics()
        
        # Performance monitoring
        self.training_start_time = None
        self.last_batch_time = None
        self.batch_times = []
        self.memory_usage = []
        
        print(f"🔍 初始化FlowMetrics:")
        print(f"   - 设备: {self.device}")
        print(f"   - 可视化: {self.enable_visualization}")
        print(f"   - 内存监控: {self.track_memory}")
        print(f"   - 梯度监控: {self.track_gradients}")
    
    def reset_metrics(self):
        """Reset all metric storage."""
        self.loss_history = {
            'flow_loss': [],
            'contrastive_loss': [],
            'total_loss': []
        }
        
        self.quality_metrics = {
            'ks_statistics': [],
            'spectral_similarity': [],
            'snr_scores': [],
            'diversity_scores': []
        }
        
        self.performance_metrics = {
            'iterations_per_second': [],
            'gpu_memory_usage': [],
            'gradient_norms': []
        }
        
    def update_loss_tracking(
        self,
        flow_loss: torch.Tensor,
        contrastive_loss: Optional[torch.Tensor] = None,
        total_loss: Optional[torch.Tensor] = None
    ):
        """
        Update loss tracking with current batch losses.
        
        Parameters
        ----------
        flow_loss : torch.Tensor
            Flow reconstruction loss
        contrastive_loss : Optional[torch.Tensor]
            Contrastive learning loss
        total_loss : Optional[torch.Tensor]
            Combined total loss
        """
        self.loss_history['flow_loss'].append(float(flow_loss.item()))
        
        if contrastive_loss is not None:
            self.loss_history['contrastive_loss'].append(float(contrastive_loss.item()))
        
        if total_loss is not None:
            self.loss_history['total_loss'].append(float(total_loss.item()))
        elif contrastive_loss is not None:
            # Compute total if not provided
            total = flow_loss + contrastive_loss
            self.loss_history['total_loss'].append(float(total.item()))
    
    def compute_convergence_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """
        Compute convergence metrics from loss history.
        
        Parameters
        ----------
        window_size : int
            Window size for computing moving statistics
            
        Returns
        -------
        Dict[str, float]
            Convergence metrics including trends and stability
        """
        metrics = {}
        
        for loss_name, loss_values in self.loss_history.items():
            if len(loss_values) < window_size:
                continue
                
            recent_values = np.array(loss_values[-window_size:])
            
            # Compute trend (slope of linear fit)
            x = np.arange(len(recent_values))
            slope, _, _, _, _ = stats.linregress(x, recent_values)
            
            # Compute stability (coefficient of variation)
            cv = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
            
            # Compute relative improvement
            if len(loss_values) >= 2 * window_size:
                old_mean = np.mean(loss_values[-2*window_size:-window_size])
                new_mean = np.mean(recent_values)
                improvement = (old_mean - new_mean) / (old_mean + 1e-8)
            else:
                improvement = 0.0
            
            metrics[f'{loss_name}_trend'] = float(slope)
            metrics[f'{loss_name}_stability'] = float(cv)
            metrics[f'{loss_name}_improvement'] = float(improvement)
        
        return metrics
    
    def compute_ks_test(
        self,
        real_samples: torch.Tensor,
        generated_samples: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov test between real and generated samples.
        
        Parameters
        ----------
        real_samples : torch.Tensor
            Real signal samples (B, L, C) or (B, L*C)
        generated_samples : torch.Tensor
            Generated signal samples (B, L, C) or (B, L*C)
            
        Returns
        -------
        Tuple[float, float]
            KS statistic and p-value
        """
        # Flatten samples if needed
        if len(real_samples.shape) > 2:
            real_flat = real_samples.reshape(real_samples.shape[0], -1)
        else:
            real_flat = real_samples
            
        if len(generated_samples.shape) > 2:
            gen_flat = generated_samples.reshape(generated_samples.shape[0], -1)
        else:
            gen_flat = generated_samples
        
        # Convert to numpy and compute KS test on flattened distributions
        real_np = real_flat.detach().cpu().numpy().flatten()
        gen_np = gen_flat.detach().cpu().numpy().flatten()
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(real_np, gen_np)
        
        # Store result
        self.quality_metrics['ks_statistics'].append(float(ks_stat))
        
        return float(ks_stat), float(p_value)
    
    def compute_spectral_similarity(
        self,
        real_samples: torch.Tensor,
        generated_samples: torch.Tensor,
        sample_rate: float = 1.0
    ) -> float:
        """
        Compute spectral similarity between real and generated samples.
        
        Parameters
        ----------
        real_samples : torch.Tensor
            Real signal samples (B, L, C)
        generated_samples : torch.Tensor
            Generated signal samples (B, L, C)
        sample_rate : float
            Signal sampling rate
            
        Returns
        -------
        float
            Spectral similarity score (0-1, higher is better)
        """
        # Convert to numpy
        real_np = real_samples.detach().cpu().numpy()
        gen_np = generated_samples.detach().cpu().numpy()
        
        similarities = []
        
        # Compute spectral similarity for each sample
        for i in range(min(real_np.shape[0], gen_np.shape[0])):
            real_signal = real_np[i].mean(axis=-1) if len(real_np.shape) == 3 else real_np[i]
            gen_signal = gen_np[i].mean(axis=-1) if len(gen_np.shape) == 3 else gen_np[i]
            
            # Compute FFT
            real_fft = np.abs(fft(real_signal))
            gen_fft = np.abs(fft(gen_signal))
            
            # Normalize
            real_fft = real_fft / (np.sum(real_fft) + 1e-8)
            gen_fft = gen_fft / (np.sum(gen_fft) + 1e-8)
            
            # Compute similarity (1 - Jensen-Shannon divergence)
            # JS divergence between two probability distributions
            m = 0.5 * (real_fft + gen_fft)
            js_div = 0.5 * stats.entropy(real_fft + 1e-10, m + 1e-10) + \
                     0.5 * stats.entropy(gen_fft + 1e-10, m + 1e-10)
            
            similarity = 1.0 - np.sqrt(js_div)
            similarities.append(similarity)
        
        avg_similarity = float(np.mean(similarities))
        self.quality_metrics['spectral_similarity'].append(avg_similarity)
        
        return avg_similarity
    
    def compute_snr_score(
        self,
        real_samples: torch.Tensor,
        generated_samples: torch.Tensor
    ) -> float:
        """
        Compute Signal-to-Noise Ratio comparison between real and generated samples.
        
        Parameters
        ----------
        real_samples : torch.Tensor
            Real signal samples (B, L, C)
        generated_samples : torch.Tensor
            Generated signal samples (B, L, C)
            
        Returns
        -------
        float
            Average SNR score
        """
        def compute_snr(signal: np.ndarray) -> float:
            """Compute SNR for a single signal."""
            signal_power = np.mean(signal ** 2)
            noise_power = np.var(signal - np.mean(signal))
            return 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        real_np = real_samples.detach().cpu().numpy()
        gen_np = generated_samples.detach().cpu().numpy()
        
        real_snrs = []
        gen_snrs = []
        
        for i in range(min(real_np.shape[0], gen_np.shape[0])):
            real_signal = real_np[i].mean(axis=-1) if len(real_np.shape) == 3 else real_np[i]
            gen_signal = gen_np[i].mean(axis=-1) if len(gen_np.shape) == 3 else gen_np[i]
            
            real_snrs.append(compute_snr(real_signal))
            gen_snrs.append(compute_snr(gen_signal))
        
        # Compute SNR similarity score
        snr_diff = np.abs(np.array(real_snrs) - np.array(gen_snrs))
        snr_score = float(np.exp(-np.mean(snr_diff) / 10.0))  # Normalize to 0-1
        
        self.quality_metrics['snr_scores'].append(snr_score)
        
        return snr_score
    
    def compute_diversity_score(
        self,
        generated_samples: torch.Tensor,
        method: str = 'pairwise_distance'
    ) -> float:
        """
        Compute diversity score for generated samples.
        
        Parameters
        ----------
        generated_samples : torch.Tensor
            Generated signal samples (B, L, C)
        method : str
            Method for computing diversity ('pairwise_distance' or 'entropy')
            
        Returns
        -------
        float
            Diversity score (higher is better)
        """
        gen_np = generated_samples.detach().cpu().numpy()
        
        # Flatten samples for distance computation
        if len(gen_np.shape) == 3:
            gen_flat = gen_np.reshape(gen_np.shape[0], -1)
        else:
            gen_flat = gen_np
        
        if method == 'pairwise_distance':
            # Compute pairwise distances
            if pairwise_distances is not None:
                distances = pairwise_distances(gen_flat, metric='euclidean')
                # Remove diagonal (self-distances)
                mask = np.eye(distances.shape[0], dtype=bool)
                non_diagonal_distances = distances[~mask]
                
                # Diversity is average pairwise distance
                diversity_score = float(np.mean(non_diagonal_distances))
            else:
                # Fallback: compute manual pairwise distances
                distances = np.linalg.norm(gen_flat[:, None, :] - gen_flat[None, :, :], axis=2)
                mask = np.eye(distances.shape[0], dtype=bool)
                non_diagonal_distances = distances[~mask]
                diversity_score = float(np.mean(non_diagonal_distances))
            
        elif method == 'entropy':
            # Compute entropy-based diversity
            # Discretize signals and compute entropy
            gen_flat_norm = (gen_flat - gen_flat.min()) / (gen_flat.max() - gen_flat.min() + 1e-8)
            gen_discrete = (gen_flat_norm * 100).astype(int)
            
            # Compute histogram entropy
            unique_vals, counts = np.unique(gen_discrete.flatten(), return_counts=True)
            probabilities = counts / np.sum(counts)
            diversity_score = float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
            
        else:
            raise ValueError(f"Unknown diversity method: {method}")
        
        self.quality_metrics['diversity_scores'].append(diversity_score)
        
        return diversity_score
    
    def create_comparison_plots(
        self,
        real_samples: torch.Tensor,
        generated_samples: torch.Tensor,
        num_samples: int = 5,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Optional[plt.Figure]:
        """
        Create visualization comparing real and generated samples.
        
        Parameters
        ----------
        real_samples : torch.Tensor
            Real signal samples (B, L, C)
        generated_samples : torch.Tensor
            Generated signal samples (B, L, C)
        num_samples : int
            Number of samples to visualize
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        Optional[plt.Figure]
            Matplotlib figure or None if visualization disabled
        """
        if not self.enable_visualization:
            return None
        
        try:
            real_np = real_samples.detach().cpu().numpy()
            gen_np = generated_samples.detach().cpu().numpy()
            
            # Select random samples
            n_real = min(num_samples, real_np.shape[0])
            n_gen = min(num_samples, gen_np.shape[0])
            
            real_indices = np.random.choice(real_np.shape[0], n_real, replace=False)
            gen_indices = np.random.choice(gen_np.shape[0], n_gen, replace=False)
            
            fig, axes = plt.subplots(2, num_samples, figsize=figsize)
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(num_samples):
                # Real samples
                if i < n_real:
                    real_signal = real_np[real_indices[i]]
                    if len(real_signal.shape) == 2:  # Multi-channel
                        real_signal = real_signal.mean(axis=1)
                    axes[0, i].plot(real_signal, 'b-', alpha=0.7)
                    axes[0, i].set_title(f'Real Sample {i+1}')
                    axes[0, i].grid(True, alpha=0.3)
                
                # Generated samples
                if i < n_gen:
                    gen_signal = gen_np[gen_indices[i]]
                    if len(gen_signal.shape) == 2:  # Multi-channel
                        gen_signal = gen_signal.mean(axis=1)
                    axes[1, i].plot(gen_signal, 'r-', alpha=0.7)
                    axes[1, i].set_title(f'Generated Sample {i+1}')
                    axes[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.save_plots:
                import os
                os.makedirs(self.plot_dir, exist_ok=True)
                plt.savefig(f"{self.plot_dir}/flow_comparison_{len(self.quality_metrics['ks_statistics'])}.png")
            
            return fig
            
        except Exception as e:
            print(f"⚠️  可视化创建失败: {e}")
            return None
    
    def track_training_speed(self):
        """Track training speed (iterations per second)."""
        current_time = time.time()
        
        if self.training_start_time is None:
            self.training_start_time = current_time
            self.last_batch_time = current_time
            return
        
        if self.last_batch_time is not None:
            batch_duration = current_time - self.last_batch_time
            self.batch_times.append(batch_duration)
            
            # Keep only recent batch times (last 100)
            if len(self.batch_times) > 100:
                self.batch_times = self.batch_times[-100:]
            
            # Compute iterations per second
            if len(self.batch_times) > 0:
                avg_batch_time = np.mean(self.batch_times)
                iter_per_sec = 1.0 / (avg_batch_time + 1e-8)
                self.performance_metrics['iterations_per_second'].append(iter_per_sec)
        
        self.last_batch_time = current_time
    
    def track_gpu_memory(self):
        """Track GPU memory usage."""
        if not self.track_memory:
            return
        
        try:
            if torch.cuda.is_available():
                # GPU memory
                gpu_memory = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
                self.memory_usage.append(gpu_memory)
                self.performance_metrics['gpu_memory_usage'].append(gpu_memory)
            else:
                # CPU memory fallback
                process = psutil.Process()
                cpu_memory = process.memory_info().rss / (1024**3)  # GB
                self.performance_metrics['gpu_memory_usage'].append(cpu_memory)
        except Exception as e:
            print(f"⚠️  内存监控失败: {e}")
    
    def track_gradient_norms(self, model: nn.Module):
        """
        Track gradient norms for training stability monitoring.
        
        Parameters
        ----------
        model : nn.Module
            Model to track gradients from
        """
        if not self.track_gradients:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** 0.5
            self.performance_metrics['gradient_norms'].append(total_norm)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns
        -------
        Dict[str, Any]
            Performance metrics summary
        """
        summary = {}
        
        # Training speed metrics
        if self.performance_metrics['iterations_per_second']:
            iter_per_sec = self.performance_metrics['iterations_per_second']
            summary['avg_iter_per_sec'] = float(np.mean(iter_per_sec[-50:]))  # Recent average
            summary['max_iter_per_sec'] = float(np.max(iter_per_sec))
        
        # Memory usage metrics
        if self.performance_metrics['gpu_memory_usage']:
            memory_usage = self.performance_metrics['gpu_memory_usage']
            summary['avg_memory_gb'] = float(np.mean(memory_usage))
            summary['max_memory_gb'] = float(np.max(memory_usage))
            summary['current_memory_gb'] = float(memory_usage[-1]) if memory_usage else 0.0
        
        # Gradient stability metrics
        if self.performance_metrics['gradient_norms']:
            grad_norms = self.performance_metrics['gradient_norms']
            summary['avg_gradient_norm'] = float(np.mean(grad_norms[-50:]))
            summary['max_gradient_norm'] = float(np.max(grad_norms))
            summary['gradient_stability'] = float(np.std(grad_norms[-50:]))  # Lower is more stable
        
        # Loss convergence metrics
        convergence_metrics = self.compute_convergence_metrics()
        summary.update(convergence_metrics)
        
        # Quality metrics summary
        if self.quality_metrics['ks_statistics']:
            summary['avg_ks_statistic'] = float(np.mean(self.quality_metrics['ks_statistics']))
        
        if self.quality_metrics['spectral_similarity']:
            summary['avg_spectral_similarity'] = float(np.mean(self.quality_metrics['spectral_similarity']))
        
        if self.quality_metrics['snr_scores']:
            summary['avg_snr_score'] = float(np.mean(self.quality_metrics['snr_scores']))
        
        if self.quality_metrics['diversity_scores']:
            summary['avg_diversity_score'] = float(np.mean(self.quality_metrics['diversity_scores']))
        
        return summary
    
    def save_metrics(self, filepath: str):
        """
        Save all metrics to file.
        
        Parameters
        ----------
        filepath : str
            Path to save metrics
        """
        metrics_data = {
            'loss_history': self.loss_history,
            'quality_metrics': self.quality_metrics,
            'performance_metrics': self.performance_metrics
        }
        
        torch.save(metrics_data, filepath)
        print(f"📊 指标已保存至: {filepath}")
    
    def load_metrics(self, filepath: str):
        """
        Load metrics from file.
        
        Parameters
        ----------
        filepath : str
            Path to load metrics from
        """
        metrics_data = torch.load(filepath, map_location='cpu')
        
        self.loss_history = metrics_data.get('loss_history', {})
        self.quality_metrics = metrics_data.get('quality_metrics', {})
        self.performance_metrics = metrics_data.get('performance_metrics', {})
        
        print(f"📊 指标已加载自: {filepath}")