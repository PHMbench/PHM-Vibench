# ContrastiveIDTask Performance Benchmark Suite

This document describes the comprehensive performance benchmark suite for the ContrastiveIDTask implementation (Task-011).

## Overview

The benchmark suite provides detailed performance analysis across five key areas:

1. **Training Performance** - Speed, memory usage, throughput
2. **Data Processing Performance** - Loading, windowing, batch preparation
3. **Model Performance** - Forward/backward pass timing, InfoNCE computation
4. **Scalability Testing** - Batch size and window size limits
5. **Hardware Optimization** - CPU vs GPU, mixed precision benefits

## Quick Start

### Basic Usage

```bash
# Run complete benchmark suite
python scripts/run_performance_benchmark.py

# Run specific benchmark category
python scripts/run_performance_benchmark.py --test training
python scripts/run_performance_benchmark.py --test scalability

# Quick mode for faster testing
python scripts/run_performance_benchmark.py --quick

# Force CPU testing
python scripts/run_performance_benchmark.py --device cpu
```

### Direct Python Usage

```python
from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark

# Create benchmark instance
benchmark = AdvancedPerformanceBenchmark(save_dir="./my_results")

# Run complete suite
success = benchmark.run_comprehensive_benchmark()

# Run individual tests
benchmark.benchmark_training_performance()
benchmark.benchmark_scalability_testing()

# Generate reports
benchmark.generate_comprehensive_report()
```

## Benchmark Details

### 1. Training Performance Benchmark

**Tests:**
- Training throughput (samples/second)
- Batch processing time
- Forward pass timing
- Backward pass timing
- Memory usage during training

**Configurations Tested:**
- Small: batch_size=16, window_size=512
- Medium: batch_size=32, window_size=1024
- Large: batch_size=64, window_size=2048

**Performance Targets:**
- Throughput: ≥50 samples/second
- Batch time: ≤200ms
- Forward pass: ≤50ms
- Backward pass: ≤100ms

### 2. Data Processing Performance Benchmark

**Tests:**
- Data generation speed
- Window generation timing
- Batch preparation efficiency
- Memory usage during processing

**Data Complexities:**
- Simple: Random noise signals
- Medium: Sinusoidal patterns with noise
- Complex: Multi-component signals with trends

**Performance Targets:**
- Batch preparation: ≤50ms
- Window generation: ≥1000 windows/second

### 3. Model Performance Benchmark

**Tests:**
- Forward pass timing across model sizes
- Backward pass timing
- InfoNCE loss computation speed
- Model memory usage
- Parameter count analysis

**Model Complexities:**
- Simple: 2-layer MLP
- Medium: 4-layer MLP with dropout
- Complex: 6-layer MLP with regularization

**Performance Targets:**
- Forward pass: ≤50ms
- Backward pass: ≤100ms
- InfoNCE computation: ≤10ms

### 4. Scalability Testing

**Batch Size Scalability:**
- Tests: 4, 8, 16, 32, 64, 128, 256
- Measures maximum working batch size
- Analyzes memory scaling patterns

**Window Size Scalability:**
- Tests: 256, 512, 1024, 2048, 4096, 8192
- Measures maximum working window size
- Analyzes computational scaling

**Performance Targets:**
- Max batch size: ≥128
- Max window size: ≥4096

### 5. Hardware Optimization Testing

**CPU vs GPU Comparison:**
- Processing time comparison
- Memory usage analysis
- GPU acceleration factor

**Mixed Precision Testing:**
- FP32 vs FP16 performance
- Memory savings analysis
- Accuracy preservation check

**Performance Expectations:**
- GPU speedup: ≥5x over CPU
- Mixed precision: ≥1.2x speedup with memory savings

## Output Reports

The benchmark generates comprehensive reports in multiple formats:

### 1. HTML Report (`reports/performance_report.html`)
- Interactive web-based report
- Performance summary table
- Detailed test results
- Visual charts and graphs

### 2. Markdown Summary (`reports/performance_summary.md`)
- Executive summary with pass/fail rates
- Key performance indicators table
- Optimization recommendations
- Detailed test breakdowns

### 3. Performance Plots (`plots/`)
- `training_performance.png` - Training metrics comparison
- `scalability_analysis.png` - Batch and window size scaling
- `hardware_optimization.png` - CPU vs GPU performance

### 4. Raw Data (`comprehensive_benchmark_results.json`)
- Complete benchmark results in JSON format
- Detailed timing and memory measurements
- Configuration parameters used
- Error information if tests failed

## Performance Targets

The benchmark uses the following performance targets for validation:

```python
targets = {
    'memory_efficiency': {
        'gpu_memory_per_sample_mb': 0.5,
        'cpu_memory_per_sample_mb': 2.0,
        'memory_growth_rate': 0.1,
    },
    'training_performance': {
        'samples_per_second': 50,
        'batch_time_ms': 200,
        'epoch_time_minutes': 5,
        'gpu_utilization': 0.8,
    },
    'model_performance': {
        'forward_pass_ms': 50,
        'backward_pass_ms': 100,
        'infonce_computation_ms': 10,
    },
    'scalability': {
        'max_batch_size': 128,
        'max_window_size': 4096,
        'memory_scaling_factor': 1.2,
    },
    'data_processing': {
        'h5_loading_mb_per_sec': 100,
        'window_generation_per_sec': 1000,
        'batch_preparation_ms': 50,
    }
}
```

## Optimization Recommendations

Based on benchmark results, the system provides automated recommendations:

### Common Optimizations
- **Batch Size**: Increase if memory allows for better throughput
- **Mixed Precision**: Enable FP16 training on GPUs for speed and memory savings
- **Data Loading**: Use faster storage (NVMe SSD) for better I/O performance
- **Memory Management**: Enable gradient checkpointing for large models

### Hardware-Specific
- **CPU Optimization**: Use optimized BLAS libraries (MKL, OpenBLAS)
- **GPU Optimization**: Ensure CUDA toolkit and cuDNN are properly installed
- **Multi-GPU**: Consider data parallel training for very large datasets

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Performance on GPU**
   - Check CUDA installation
   - Verify GPU utilization with `nvidia-smi`
   - Consider data transfer bottlenecks

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify PyTorch GPU support

### Dependencies

Required packages:
```bash
torch>=2.0.0
numpy
matplotlib
seaborn
pandas
psutil
```

Optional (for enhanced profiling):
```bash
memory_profiler
line_profiler
```

## Continuous Integration

The benchmark can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Performance Benchmark
  run: |
    python scripts/run_performance_benchmark.py --quick --device cpu
    # Upload benchmark results as artifacts
```

## Performance Regression Testing

To detect performance regressions:

1. Run benchmarks on baseline implementation
2. Save results as reference
3. Compare new results against baseline
4. Alert if performance degrades beyond threshold

```python
# Example regression detection
def check_performance_regression(current_results, baseline_results, threshold=0.1):
    for metric in ['throughput_samples_per_sec', 'avg_batch_time_ms']:
        current_val = current_results.get(metric, 0)
        baseline_val = baseline_results.get(metric, 0)
        
        if baseline_val > 0:
            regression = (current_val - baseline_val) / baseline_val
            if abs(regression) > threshold:
                print(f"Performance regression detected in {metric}: {regression:.1%}")
```

## Contributing

When adding new performance tests:

1. Add test method to `AdvancedPerformanceBenchmark`
2. Update performance targets if needed
3. Add corresponding report generation
4. Include test in main benchmark suite
5. Update documentation

For questions or issues, please refer to the project's main documentation or create an issue in the repository.