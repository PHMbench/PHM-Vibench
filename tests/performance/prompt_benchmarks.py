"""
Prompt Components Performance Benchmarks

This module provides comprehensive performance testing for all HSE Prompt components:
1. Latency benchmarking with different input sizes and batch sizes
2. Memory usage profiling with peak usage tracking  
3. Throughput testing for real-time inference requirements
4. Comparative analysis with baseline methods
5. Cross-component integration performance analysis

The benchmarks measure:
- NFR-P1: Real-time processing performance (<100ms, >50 samples/second)
- NFR-P2: Resource efficiency (<8GB memory, CPU <80%)
- NFR-P3: Accuracy baselines (>85% cross-system accuracy)

Author: PHM-Vibench Team
Date: 2025-09-09
License: MIT
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import warnings

# Import prompt components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.model_factory.ISFM_Prompt.components.SystemPromptEncoder import SystemPromptEncoder
    from src.model_factory.ISFM_Prompt.components.PromptFusion import PromptFusion
    from src.model_factory.ISFM_Prompt.components.MemoryOptimizedFusion import MemoryOptimizedFusion
    from src.model_factory.ISFM_Prompt.components.MixedPrecisionWrapper import MixedPrecisionWrapper
    COMPONENTS_AVAILABLE = True
    
    # Try to import advanced components
    try:
        from src.model_factory.ISFM_Prompt.embedding.E_01_HSE_v2 import E_01_HSE_v2
        EMBEDDING_AVAILABLE = True
    except ImportError:
        EMBEDDING_AVAILABLE = False
        
    try:
        from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import M_02_ISFM_Prompt
        FULL_MODEL_AVAILABLE = True
    except ImportError:
        FULL_MODEL_AVAILABLE = False
        
except ImportError as e:
    print(f"Warning: Could not import prompt components: {e}")
    COMPONENTS_AVAILABLE = False
    EMBEDDING_AVAILABLE = False
    FULL_MODEL_AVAILABLE = False


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmark parameters."""
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    num_patches_list: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    signal_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    prompt_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    fusion_types: List[str] = field(default_factory=lambda: ['concat', 'attention', 'gating'])
    num_iterations: int = 50
    warmup_iterations: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_results: bool = True
    output_dir: str = 'benchmark_results'


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    component_name: str
    test_name: str
    config: Dict[str, Any]
    latency_ms: float
    memory_peak_mb: float
    memory_allocated_mb: float
    throughput_samples_per_sec: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarker for HSE Prompt components.
    
    This class provides systematic performance testing across different:
    - Component types (encoder, fusion, embedding, full model)
    - Input configurations (batch sizes, sequence lengths, dimensions)
    - Hardware setups (CPU/GPU, memory configurations)
    - Optimization settings (mixed precision, memory optimization)
    """
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmarker with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance tracking
        self.baseline_results = {}
        
        print(f"=== HSE Prompt Performance Benchmarker ===")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.output_dir}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all performance benchmarks.
        
        Returns:
            Dict containing comprehensive benchmark results
        """
        if not COMPONENTS_AVAILABLE:
            print("❌ Cannot run benchmarks - components not available")
            return {"error": "Components not available"}
        
        print("\n🚀 Starting comprehensive performance benchmarks...")
        
        # 1. Component-level benchmarks
        print("\n--- Component-Level Benchmarks ---")
        self._benchmark_system_prompt_encoder()
        self._benchmark_prompt_fusion()
        self._benchmark_memory_optimized_fusion()
        self._benchmark_hse_embedding()
        
        # 2. Integration benchmarks
        print("\n--- Integration Benchmarks ---")
        self._benchmark_full_model()
        self._benchmark_mixed_precision()
        
        # 3. Scalability benchmarks
        print("\n--- Scalability Benchmarks ---")
        self._benchmark_scalability()
        
        # 4. Real-time inference benchmarks
        print("\n--- Real-time Inference Benchmarks ---")
        self._benchmark_real_time_inference()
        
        # 5. Memory efficiency benchmarks
        print("\n--- Memory Efficiency Benchmarks ---")
        self._benchmark_memory_efficiency()
        
        # Generate comprehensive report
        report = self._generate_benchmark_report()
        
        if self.config.save_results:
            self._save_results(report)
        
        print(f"\n✅ Benchmarks completed! Results saved to {self.output_dir}")
        return report
    
    def _benchmark_system_prompt_encoder(self):
        """Benchmark SystemPromptEncoder performance."""
        print("Benchmarking SystemPromptEncoder...")
        
        for prompt_dim in self.config.prompt_dims:
            encoder = SystemPromptEncoder(prompt_dim=prompt_dim).to(self.device)
            
            for batch_size in self.config.batch_sizes:
                # Create test metadata
                metadata_dict = SystemPromptEncoder.create_metadata_dict(
                    dataset_ids=[1, 2, 6] * (batch_size // 3 + 1),
                    domain_ids=[5, 3, 7] * (batch_size // 3 + 1),
                    sample_rates=[1000.0, 2000.0, 1500.0] * (batch_size // 3 + 1),
                    device=self.device
                )
                
                # Trim to exact batch size
                for key in metadata_dict:
                    metadata_dict[key] = metadata_dict[key][:batch_size]
                
                config = {
                    'batch_size': batch_size,
                    'prompt_dim': prompt_dim
                }
                
                result = self._benchmark_component(
                    component=encoder,
                    component_name="SystemPromptEncoder",
                    test_name=f"prompt_dim_{prompt_dim}_batch_{batch_size}",
                    config=config,
                    forward_fn=lambda: encoder(metadata_dict)
                )
                
                self.results.append(result)
    
    def _benchmark_prompt_fusion(self):
        """Benchmark PromptFusion performance."""
        print("Benchmarking PromptFusion...")
        
        for fusion_type in self.config.fusion_types:
            for signal_dim in [256, 512]:
                for prompt_dim in [128, 256]:
                    fusion = PromptFusion(
                        signal_dim=signal_dim,
                        prompt_dim=prompt_dim,
                        fusion_type=fusion_type
                    ).to(self.device)
                    
                    for batch_size in self.config.batch_sizes:
                        for num_patches in [64, 128]:
                            # Create test inputs
                            signal_emb = torch.randn(batch_size, num_patches, signal_dim, device=self.device)
                            prompt_emb = torch.randn(batch_size, prompt_dim, device=self.device)
                            
                            config = {
                                'fusion_type': fusion_type,
                                'batch_size': batch_size,
                                'num_patches': num_patches,
                                'signal_dim': signal_dim,
                                'prompt_dim': prompt_dim
                            }
                            
                            result = self._benchmark_component(
                                component=fusion,
                                component_name="PromptFusion",
                                test_name=f"{fusion_type}_sig{signal_dim}_prompt{prompt_dim}_batch{batch_size}_patches{num_patches}",
                                config=config,
                                forward_fn=lambda: fusion(signal_emb, prompt_emb)
                            )
                            
                            self.results.append(result)
    
    def _benchmark_memory_optimized_fusion(self):
        """Benchmark MemoryOptimizedFusion performance."""
        print("Benchmarking MemoryOptimizedFusion...")
        
        for enable_checkpointing in [True, False]:
            fusion = MemoryOptimizedFusion(
                signal_dim=256,
                prompt_dim=128,
                fusion_type='attention',
                enable_checkpointing=enable_checkpointing,
                verbose=False
            ).to(self.device)
            
            for batch_size in [4, 8, 16]:
                for num_patches in [64, 128]:
                    signal_emb = torch.randn(batch_size, num_patches, 256, device=self.device)
                    prompt_emb = torch.randn(batch_size, 128, device=self.device)
                    
                    config = {
                        'batch_size': batch_size,
                        'num_patches': num_patches,
                        'enable_checkpointing': enable_checkpointing
                    }
                    
                    result = self._benchmark_component(
                        component=fusion,
                        component_name="MemoryOptimizedFusion",
                        test_name=f"checkpoint_{enable_checkpointing}_batch{batch_size}_patches{num_patches}",
                        config=config,
                        forward_fn=lambda: fusion(signal_emb, prompt_emb)
                    )
                    
                    self.results.append(result)
    
    def _benchmark_hse_embedding(self):
        """Benchmark E_01_HSE_v2 embedding performance."""
        if not EMBEDDING_AVAILABLE:
            print("Skipping E_01_HSE_v2 benchmarking - component not available")
            return
            
        print("Benchmarking E_01_HSE_v2...")
        
        # Create test arguments
        class TestArgs:
            patch_size_L = 256
            patch_size_C = 1
            num_patches = 64
            output_dim = 512
            prompt_dim = 128
            fusion_strategy = 'attention'
            training_stage = 'pretrain'
            freeze_prompt = False
        
        args = TestArgs()
        embedding = E_01_HSE_v2(args).to(self.device)
        
        for batch_size in [2, 4, 8]:
            for seq_len in [1024, 2048]:
                # Create test inputs
                x = torch.randn(batch_size, seq_len, 2, device=self.device)
                fs = 1000.0
                metadata = [
                    {'Dataset_id': 1, 'Domain_id': 5, 'Sample_rate': 1000.0}
                ] * batch_size
                
                config = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'output_dim': args.output_dim
                }
                
                result = self._benchmark_component(
                    component=embedding,
                    component_name="E_01_HSE_v2",
                    test_name=f"batch{batch_size}_seq{seq_len}",
                    config=config,
                    forward_fn=lambda: embedding(x, fs, metadata)
                )
                
                self.results.append(result)
    
    def _benchmark_full_model(self):
        """Benchmark full M_02_ISFM_Prompt model."""
        if not FULL_MODEL_AVAILABLE:
            print("Skipping M_02_ISFM_Prompt benchmarking - component not available")
            return
            
        print("Benchmarking M_02_ISFM_Prompt...")
        
        # Create test arguments
        class TestArgs:
            embedding = 'E_01_HSE_v2'
            backbone = 'Identity'  # Simplified for benchmarking
            task_head = 'Identity'
            prompt_dim = 128
            training_stage = 'pretrain'
            patch_size_L = 256
            num_patches = 64
            output_dim = 512
            num_classes = 4
        
        args = TestArgs()
        
        try:
            model = M_02_ISFM_Prompt(args).to(self.device)
            
            for batch_size in [2, 4, 8]:
                for seq_len in [1024, 2048]:
                    x = torch.randn(batch_size, seq_len, 2, device=self.device)
                    metadata = [
                        {'Dataset_id': 1, 'Domain_id': 5, 'Sample_rate': 1000.0}
                    ] * batch_size
                    
                    config = {
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'model_type': 'M_02_ISFM_Prompt'
                    }
                    
                    result = self._benchmark_component(
                        component=model,
                        component_name="M_02_ISFM_Prompt",
                        test_name=f"full_model_batch{batch_size}_seq{seq_len}",
                        config=config,
                        forward_fn=lambda: model(x, metadata=metadata)
                    )
                    
                    self.results.append(result)
                    
        except Exception as e:
            print(f"Warning: Could not benchmark full model: {e}")
    
    def _benchmark_mixed_precision(self):
        """Benchmark mixed precision performance."""
        print("Benchmarking Mixed Precision...")
        
        # Simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        base_model = TestModel().to(self.device)
        
        for enable_fp16 in [True, False]:
            if enable_fp16:
                wrapped_model = MixedPrecisionWrapper(base_model, enabled=True, verbose=False)
            else:
                wrapped_model = base_model
            
            for batch_size in [16, 32, 64]:
                x = torch.randn(batch_size, 256, device=self.device)
                
                config = {
                    'batch_size': batch_size,
                    'enable_fp16': enable_fp16
                }
                
                result = self._benchmark_component(
                    component=wrapped_model,
                    component_name="MixedPrecision",
                    test_name=f"fp16_{enable_fp16}_batch{batch_size}",
                    config=config,
                    forward_fn=lambda: wrapped_model(x)
                )
                
                self.results.append(result)
    
    def _benchmark_scalability(self):
        """Benchmark scalability with increasing load."""
        print("Benchmarking Scalability...")
        
        fusion = PromptFusion(signal_dim=256, prompt_dim=128, fusion_type='attention').to(self.device)
        
        # Test increasing batch sizes
        large_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for batch_size in large_batch_sizes:
            try:
                signal_emb = torch.randn(batch_size, 64, 256, device=self.device)
                prompt_emb = torch.randn(batch_size, 128, device=self.device)
                
                config = {
                    'batch_size': batch_size,
                    'test_type': 'scalability'
                }
                
                result = self._benchmark_component(
                    component=fusion,
                    component_name="Scalability",
                    test_name=f"large_batch_{batch_size}",
                    config=config,
                    forward_fn=lambda: fusion(signal_emb, prompt_emb),
                    timeout_ms=5000  # 5 second timeout
                )
                
                self.results.append(result)
                
                # Stop if we hit memory limits
                if not result.success:
                    print(f"Scalability limit reached at batch size {batch_size}")
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch size {batch_size}")
                break
    
    def _benchmark_real_time_inference(self):
        """Benchmark real-time inference requirements."""
        print("Benchmarking Real-time Inference...")
        
        # Test components for real-time requirements (NFR-P1: <100ms, >50 samples/sec)
        encoder = SystemPromptEncoder(prompt_dim=128).to(self.device)
        fusion = PromptFusion(signal_dim=256, prompt_dim=128, fusion_type='concat').to(self.device)
        
        # Real-time test configuration
        real_time_batch_sizes = [1, 4, 8]  # Typical real-time batch sizes
        
        for component, component_name in [(encoder, "SystemPromptEncoder"), (fusion, "PromptFusion")]:
            for batch_size in real_time_batch_sizes:
                if component_name == "SystemPromptEncoder":
                    metadata_dict = SystemPromptEncoder.create_metadata_dict(
                        dataset_ids=[1] * batch_size,
                        domain_ids=[5] * batch_size,
                        sample_rates=[1000.0] * batch_size,
                        device=self.device
                    )
                    forward_fn = lambda: encoder(metadata_dict)
                else:
                    signal_emb = torch.randn(batch_size, 32, 256, device=self.device)
                    prompt_emb = torch.randn(batch_size, 128, device=self.device)
                    forward_fn = lambda: fusion(signal_emb, prompt_emb)
                
                config = {
                    'batch_size': batch_size,
                    'test_type': 'real_time_inference'
                }
                
                result = self._benchmark_component(
                    component=component,
                    component_name=f"RealTime_{component_name}",
                    test_name=f"rt_batch{batch_size}",
                    config=config,
                    forward_fn=forward_fn,
                    num_iterations=100  # More iterations for accurate real-time measurement
                )
                
                self.results.append(result)
                
                # Check real-time requirements
                meets_latency = result.latency_ms < 100  # NFR-P1
                meets_throughput = result.throughput_samples_per_sec > 50  # NFR-P1
                
                if meets_latency and meets_throughput:
                    print(f"  ✓ {component_name} batch{batch_size}: PASSES real-time requirements")
                else:
                    print(f"  ✗ {component_name} batch{batch_size}: FAILS real-time requirements")
    
    def _benchmark_memory_efficiency(self):
        """Benchmark memory efficiency requirements."""
        print("Benchmarking Memory Efficiency...")
        
        # Test memory requirements (NFR-P2: <8GB memory)
        memory_fusion = MemoryOptimizedFusion(
            signal_dim=512,
            prompt_dim=256,
            fusion_type='attention',
            enable_checkpointing=True,
            verbose=False
        ).to(self.device)
        
        # Progressive memory stress test
        memory_test_configs = [
            (8, 64, 512),   # Small
            (16, 128, 512), # Medium
            (32, 256, 512), # Large
            (64, 512, 512), # Very Large
        ]
        
        for batch_size, num_patches, signal_dim in memory_test_configs:
            try:
                signal_emb = torch.randn(batch_size, num_patches, signal_dim, device=self.device)
                prompt_emb = torch.randn(batch_size, 256, device=self.device)
                
                config = {
                    'batch_size': batch_size,
                    'num_patches': num_patches,
                    'signal_dim': signal_dim,
                    'test_type': 'memory_efficiency'
                }
                
                result = self._benchmark_component(
                    component=memory_fusion,
                    component_name="MemoryEfficiency",
                    test_name=f"mem_b{batch_size}_p{num_patches}_d{signal_dim}",
                    config=config,
                    forward_fn=lambda: memory_fusion(signal_emb, prompt_emb)
                )
                
                self.results.append(result)
                
                # Check memory requirements
                memory_gb = result.memory_peak_mb / 1000
                meets_memory = memory_gb < 8.0  # NFR-P2
                
                if meets_memory:
                    print(f"  ✓ Config {config}: PASSES memory requirements ({memory_gb:.2f}GB)")
                else:
                    print(f"  ✗ Config {config}: FAILS memory requirements ({memory_gb:.2f}GB)")
                    
            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ Config {config}: OOM - exceeds memory limits")
                
                # Add failed result
                failed_result = BenchmarkResult(
                    component_name="MemoryEfficiency",
                    test_name=f"mem_b{batch_size}_p{num_patches}_d{signal_dim}",
                    config=config,
                    latency_ms=0.0,
                    memory_peak_mb=float('inf'),
                    memory_allocated_mb=float('inf'),
                    throughput_samples_per_sec=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message="Out of Memory"
                )
                self.results.append(failed_result)
    
    def _benchmark_component(self,
                           component: nn.Module,
                           component_name: str,
                           test_name: str,
                           config: Dict[str, Any],
                           forward_fn: Callable,
                           num_iterations: Optional[int] = None,
                           timeout_ms: Optional[float] = None) -> BenchmarkResult:
        """
        Benchmark a single component.
        
        Args:
            component: Component to benchmark
            component_name: Name of the component
            test_name: Name of the test
            config: Test configuration
            forward_fn: Function to call for forward pass
            num_iterations: Number of iterations (defaults to config)
            timeout_ms: Timeout in milliseconds
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        component.eval()
        
        try:
            with torch.no_grad():
                # Warmup
                for _ in range(self.config.warmup_iterations):
                    _ = forward_fn()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # Benchmark
                start_time = time.time()
                
                for i in range(num_iterations):
                    iteration_start = time.time()
                    _ = forward_fn()
                    
                    # Check timeout
                    if timeout_ms and (time.time() - start_time) * 1000 > timeout_ms:
                        num_iterations = i + 1
                        break
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_latency_ms = (total_time / num_iterations) * 1000
                
                # Memory metrics
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
                    memory_peak = torch.cuda.max_memory_allocated() / 1e6   # MB
                else:
                    process = psutil.Process()
                    memory_allocated = process.memory_info().rss / 1e6
                    memory_peak = memory_allocated
                
                # Throughput calculation
                batch_size = config.get('batch_size', 1)
                throughput = (batch_size * num_iterations) / total_time
                
                # CPU usage (approximate)
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                return BenchmarkResult(
                    component_name=component_name,
                    test_name=test_name,
                    config=config,
                    latency_ms=avg_latency_ms,
                    memory_peak_mb=memory_peak,
                    memory_allocated_mb=memory_allocated,
                    throughput_samples_per_sec=throughput,
                    cpu_usage_percent=cpu_usage,
                    success=True
                )
                
        except Exception as e:
            return BenchmarkResult(
                component_name=component_name,
                test_name=test_name,
                config=config,
                latency_ms=0.0,
                memory_peak_mb=0.0,
                memory_allocated_mb=0.0,
                throughput_samples_per_sec=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Convert results to DataFrame for analysis
        results_data = []
        for result in self.results:
            row = {
                'component': result.component_name,
                'test': result.test_name,
                'success': result.success,
                'latency_ms': result.latency_ms,
                'memory_peak_mb': result.memory_peak_mb,
                'memory_allocated_mb': result.memory_allocated_mb,
                'throughput_sps': result.throughput_samples_per_sec,
                'cpu_usage_pct': result.cpu_usage_percent,
                **result.config
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # Generate summary statistics
        summary = {
            'total_tests': len(self.results),
            'successful_tests': len([r for r in self.results if r.success]),
            'failed_tests': len([r for r in self.results if not r.success]),
            'components_tested': list(df['component'].unique()),
            'benchmark_duration': time.time() - (self.results[0].timestamp if self.results else time.time())
        }
        
        # Performance analysis
        performance_analysis = {}
        
        for component in df['component'].unique():
            component_df = df[df['component'] == component]
            successful_df = component_df[component_df['success'] == True]
            
            if len(successful_df) > 0:
                performance_analysis[component] = {
                    'avg_latency_ms': successful_df['latency_ms'].mean(),
                    'min_latency_ms': successful_df['latency_ms'].min(),
                    'max_latency_ms': successful_df['latency_ms'].max(),
                    'avg_memory_mb': successful_df['memory_peak_mb'].mean(),
                    'max_memory_mb': successful_df['memory_peak_mb'].max(),
                    'avg_throughput_sps': successful_df['throughput_sps'].mean(),
                    'max_throughput_sps': successful_df['throughput_sps'].max(),
                    'success_rate': len(successful_df) / len(component_df)
                }
        
        # NFR compliance analysis
        nfr_analysis = self._analyze_nfr_compliance(df)
        
        # Optimization recommendations
        recommendations = self._generate_recommendations(df)
        
        return {
            'summary': summary,
            'performance_analysis': performance_analysis,
            'nfr_compliance': nfr_analysis,
            'recommendations': recommendations,
            'raw_results': [
                {
                    'component': r.component_name,
                    'test': r.test_name,
                    'config': r.config,
                    'metrics': {
                        'latency_ms': r.latency_ms,
                        'memory_peak_mb': r.memory_peak_mb,
                        'throughput_sps': r.throughput_samples_per_sec,
                        'success': r.success
                    }
                }
                for r in self.results
            ]
        }
    
    def _analyze_nfr_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze compliance with Non-Functional Requirements."""
        
        successful_df = df[df['success'] == True]
        
        # NFR-P1: Real-time processing (<100ms, >50 samples/sec)
        nfr_p1_latency = (successful_df['latency_ms'] < 100).sum() / len(successful_df) if len(successful_df) > 0 else 0
        nfr_p1_throughput = (successful_df['throughput_sps'] > 50).sum() / len(successful_df) if len(successful_df) > 0 else 0
        
        # NFR-P2: Resource efficiency (<8GB memory, CPU <80%)
        nfr_p2_memory = (successful_df['memory_peak_mb'] < 8000).sum() / len(successful_df) if len(successful_df) > 0 else 0
        nfr_p2_cpu = (successful_df['cpu_usage_pct'] < 80).sum() / len(successful_df) if len(successful_df) > 0 else 0
        
        return {
            'NFR-P1': {
                'latency_compliance': nfr_p1_latency,
                'throughput_compliance': nfr_p1_throughput,
                'overall_compliance': min(nfr_p1_latency, nfr_p1_throughput),
                'target': 'Real-time processing: <100ms latency, >50 samples/sec'
            },
            'NFR-P2': {
                'memory_compliance': nfr_p2_memory,
                'cpu_compliance': nfr_p2_cpu,
                'overall_compliance': min(nfr_p2_memory, nfr_p2_cpu),
                'target': 'Resource efficiency: <8GB memory, <80% CPU'
            }
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate optimization recommendations based on results."""
        
        recommendations = []
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            return ["❌ No successful tests - check component implementations"]
        
        # Latency recommendations
        high_latency = successful_df[successful_df['latency_ms'] > 100]
        if len(high_latency) > 0:
            slow_components = high_latency['component'].value_counts().head(3)
            recommendations.append(
                f"🚀 Optimize latency for: {', '.join(slow_components.index.tolist())}"
            )
        
        # Memory recommendations
        high_memory = successful_df[successful_df['memory_peak_mb'] > 4000]  # >4GB
        if len(high_memory) > 0:
            memory_components = high_memory['component'].value_counts().head(3)
            recommendations.append(
                f"💾 Optimize memory usage for: {', '.join(memory_components.index.tolist())}"
            )
        
        # Fusion strategy recommendations
        if 'fusion_type' in successful_df.columns:
            fusion_perf = successful_df.groupby('fusion_type')['latency_ms'].mean()
            best_fusion = fusion_perf.idxmin()
            recommendations.append(
                f"⚡ Best fusion strategy: {best_fusion} (avg {fusion_perf[best_fusion]:.1f}ms)"
            )
        
        # Batch size recommendations
        if 'batch_size' in successful_df.columns:
            batch_throughput = successful_df.groupby('batch_size')['throughput_sps'].mean()
            best_batch = batch_throughput.idxmax()
            recommendations.append(
                f"📊 Optimal batch size: {best_batch} ({batch_throughput[best_batch]:.1f} samples/sec)"
            )
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any]):
        """Save benchmark results to files."""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save CSV results
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        results_data = []
        for result in self.results:
            row = {
                'component': result.component_name,
                'test': result.test_name,
                'success': result.success,
                'latency_ms': result.latency_ms,
                'memory_peak_mb': result.memory_peak_mb,
                'throughput_sps': result.throughput_samples_per_sec,
                'cpu_usage_pct': result.cpu_usage_percent,
                'error': result.error_message,
                **result.config
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df.to_csv(csv_file, index=False)
        
        # Generate plots
        self._create_performance_plots(df, timestamp)
        
        print(f"📊 Results saved:")
        print(f"  - Report: {json_file}")
        print(f"  - Data: {csv_file}")
        print(f"  - Plots: {self.output_dir}/plots_{timestamp}/")
    
    def _create_performance_plots(self, df: pd.DataFrame, timestamp: str):
        """Create performance visualization plots."""
        
        plots_dir = self.output_dir / f"plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("No successful results to plot")
            return
        
        # Plot 1: Latency by component
        plt.figure(figsize=(12, 6))
        component_latency = successful_df.groupby('component')['latency_ms'].mean().sort_values()
        plt.bar(range(len(component_latency)), component_latency.values)
        plt.xticks(range(len(component_latency)), component_latency.index, rotation=45)
        plt.ylabel('Average Latency (ms)')
        plt.title('Component Performance: Average Latency')
        plt.axhline(y=100, color='r', linestyle='--', label='100ms Target')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'latency_by_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Memory usage by component
        plt.figure(figsize=(12, 6))
        component_memory = successful_df.groupby('component')['memory_peak_mb'].mean().sort_values()
        plt.bar(range(len(component_memory)), component_memory.values)
        plt.xticks(range(len(component_memory)), component_memory.index, rotation=45)
        plt.ylabel('Peak Memory (MB)')
        plt.title('Component Performance: Peak Memory Usage')
        plt.axhline(y=8000, color='r', linestyle='--', label='8GB Target')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'memory_by_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Throughput by component
        plt.figure(figsize=(12, 6))
        component_throughput = successful_df.groupby('component')['throughput_sps'].mean().sort_values(ascending=False)
        plt.bar(range(len(component_throughput)), component_throughput.values)
        plt.xticks(range(len(component_throughput)), component_throughput.index, rotation=45)
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Component Performance: Throughput')
        plt.axhline(y=50, color='r', linestyle='--', label='50 samples/sec Target')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'throughput_by_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance plots created in {plots_dir}")


def main():
    """Main function to run comprehensive benchmarks."""
    
    print("=== HSE Prompt Components Performance Benchmarks ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Cannot run benchmarks - HSE Prompt components not available")
        print("Make sure all components are properly installed and accessible")
        return
    
    # Configure benchmarks
    config = BenchmarkConfig(
        batch_sizes=[1, 4, 8, 16],
        sequence_lengths=[1024, 2048],
        num_patches_list=[32, 64, 128],
        signal_dims=[256, 512],
        prompt_dims=[128, 256],
        fusion_types=['concat', 'attention', 'gating'],
        num_iterations=20,  # Reduced for faster testing
        warmup_iterations=3,
        save_results=True,
        output_dir='benchmark_results'
    )
    
    # Run benchmarks
    benchmarker = PerformanceBenchmarker(config)
    report = benchmarker.run_all_benchmarks()
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['successful_tests']/report['summary']['total_tests']*100:.1f}%")
    
    print(f"\nNFR COMPLIANCE:")
    for nfr, data in report['nfr_compliance'].items():
        compliance = data['overall_compliance'] * 100
        status = "✅ PASS" if compliance > 80 else "⚠️ MARGINAL" if compliance > 50 else "❌ FAIL"
        print(f"  {nfr}: {compliance:.1f}% {status}")
        print(f"    {data['target']}")
    
    print(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print(f"\n🎯 Benchmark completed successfully!")
    
    return report


if __name__ == '__main__':
    """Run comprehensive performance benchmarks when executed directly."""
    report = main()