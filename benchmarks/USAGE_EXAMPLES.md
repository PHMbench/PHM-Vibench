# Performance Benchmark Usage Examples

This document provides practical examples of how to use the ContrastiveIDTask performance benchmark suite.

## Command Line Examples

### 1. Run Complete Benchmark Suite
```bash
# Full comprehensive benchmark (takes ~10-15 minutes)
python scripts/run_performance_benchmark.py

# Quick mode for faster testing (takes ~2-3 minutes)
python scripts/run_performance_benchmark.py --quick

# Verbose output with detailed logging
python scripts/run_performance_benchmark.py --verbose
```

### 2. Run Specific Benchmark Categories
```bash
# Training performance only
python scripts/run_performance_benchmark.py --test training

# Data processing performance only
python scripts/run_performance_benchmark.py --test data

# Model performance profiling
python scripts/run_performance_benchmark.py --test model

# Scalability testing
python scripts/run_performance_benchmark.py --test scalability

# Hardware optimization analysis
python scripts/run_performance_benchmark.py --test hardware
```

### 3. Device-Specific Testing
```bash
# Force CPU testing (useful for CI/CD)
python scripts/run_performance_benchmark.py --device cpu --quick

# Force CUDA GPU testing
python scripts/run_performance_benchmark.py --device cuda

# Auto-detect best available device (default)
python scripts/run_performance_benchmark.py --device auto
```

### 4. Custom Output Directory
```bash
# Save results to custom directory
python scripts/run_performance_benchmark.py --save-dir ./my_benchmark_results

# Create timestamped results directory
python scripts/run_performance_benchmark.py --save-dir "./results_$(date +%Y%m%d_%H%M%S)"
```

## Python API Examples

### 1. Basic Usage
```python
from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark

# Create benchmark instance
benchmark = AdvancedPerformanceBenchmark(save_dir="./benchmark_results")

# Run complete suite
success = benchmark.run_comprehensive_benchmark()

if success:
    print("Benchmark completed successfully!")
    # Access results
    overall_score = benchmark.results['overall_performance']['score']
    print(f"Overall Performance Score: {overall_score:.1f}/100")
```

### 2. Custom Configuration
```python
from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark
import torch

# Create benchmark with custom settings
benchmark = AdvancedPerformanceBenchmark(save_dir="./custom_results")

# Override performance targets
benchmark.targets['training_performance']['samples_per_second'] = 100  # Higher target
benchmark.targets['scalability']['max_batch_size'] = 256  # Higher batch size

# Set specific device
benchmark.device = torch.device('cuda:1')  # Use specific GPU

# Run individual benchmarks
benchmark.benchmark_training_performance()
benchmark.benchmark_scalability_testing()

# Generate custom report
benchmark.generate_comprehensive_report()
```

### 3. Automated Testing Integration
```python
import json
from pathlib import Path
from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark

def run_performance_regression_test(baseline_file="baseline_results.json"):
    """Run performance benchmark and compare with baseline"""
    
    # Run current benchmark
    benchmark = AdvancedPerformanceBenchmark(save_dir="./current_results")
    success = benchmark.run_comprehensive_benchmark()
    
    if not success:
        raise RuntimeError("Benchmark failed to complete")
    
    # Load baseline results
    baseline_path = Path(baseline_file)
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        # Compare key metrics
        current_score = benchmark.results['overall_performance']['score']
        baseline_score = baseline.get('overall_performance', {}).get('score', 0)
        
        regression_threshold = 10.0  # 10% degradation threshold
        if current_score < baseline_score - regression_threshold:
            raise RuntimeError(f"Performance regression detected: {current_score:.1f} vs {baseline_score:.1f}")
        
        print(f"Performance check passed: {current_score:.1f}/100 (baseline: {baseline_score:.1f}/100)")
    
    else:
        # Save current results as new baseline
        with open(baseline_path, 'w') as f:
            json.dump(benchmark.results, f, indent=2, default=str)
        print("Baseline results saved for future comparisons")
    
    return benchmark.results

# Usage
try:
    results = run_performance_regression_test()
    print("Performance regression test passed!")
except RuntimeError as e:
    print(f"Performance test failed: {e}")
```

### 4. Custom Benchmark Extension
```python
from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark
import time
import torch

class CustomPerformanceBenchmark(AdvancedPerformanceBenchmark):
    """Extended benchmark with custom tests"""
    
    def benchmark_custom_optimization(self):
        """Custom optimization benchmark"""
        self.logger.info("Running custom optimization benchmark...")
        
        with self.performance_monitor("custom_optimization"):
            # Custom test logic here
            config = self.create_mock_config(batch_size=32, window_size=1024)
            
            # Test custom optimization techniques
            start_time = time.time()
            
            # Example: Test different data layouts
            data_formats = ['channels_first', 'channels_last']
            format_results = {}
            
            for fmt in data_formats:
                # Simulate different data format performance
                test_time = time.time()
                
                # Your custom optimization test logic here
                torch.manual_seed(42)  # For reproducibility
                test_data = torch.randn(32, 1024, 1)
                
                if fmt == 'channels_first':
                    test_data = test_data.permute(0, 2, 1)
                
                # Simulate processing
                _ = torch.nn.functional.relu(test_data).sum()
                
                format_results[fmt] = time.time() - test_time
            
            # Store results
            self.results['custom_optimization'] = {
                'format_results': format_results,
                'best_format': min(format_results.items(), key=lambda x: x[1])[0],
                'total_time': time.time() - start_time
            }
            
            self.logger.info(f"Custom optimization completed: {format_results}")
    
    def run_extended_benchmark(self):
        """Run extended benchmark suite with custom tests"""
        # Run standard benchmarks
        success = self.run_comprehensive_benchmark()
        
        if success:
            # Add custom benchmarks
            self.benchmark_custom_optimization()
            
            # Regenerate report with custom results
            self.generate_comprehensive_report()
        
        return success

# Usage
extended_benchmark = CustomPerformanceBenchmark(save_dir="./extended_results")
success = extended_benchmark.run_extended_benchmark()
```

## Continuous Integration Examples

### 1. GitHub Actions Workflow
```yaml
name: Performance Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install torch numpy matplotlib seaborn pandas psutil
    
    - name: Run performance benchmark
      run: |
        python scripts/run_performance_benchmark.py --quick --device cpu --save-dir ./benchmark_results
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: ./benchmark_results/
        
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const path = './benchmark_results/reports/performance_summary.md';
          if (fs.existsSync(path)) {
            const summary = fs.readFileSync(path, 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Performance Benchmark Results\n\n${summary}`
            });
          }
```

### 2. Docker Container Testing
```dockerfile
# Dockerfile for benchmark testing
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy benchmark code
COPY benchmarks/ ./benchmarks/
COPY scripts/ ./scripts/
COPY src/ ./src/

# Run benchmark
CMD ["python", "scripts/run_performance_benchmark.py", "--quick", "--device", "cpu"]
```

```bash
# Build and run benchmark container
docker build -t contrastive-benchmark .
docker run --rm -v $(pwd)/results:/app/benchmark_results contrastive-benchmark
```

## Performance Analysis Examples

### 1. Trend Analysis
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_performance_trends(results_dir="./historical_results"):
    """Analyze performance trends over time"""
    
    results_path = Path(results_dir)
    historical_data = []
    
    # Load historical results
    for result_file in results_path.glob("**/comprehensive_benchmark_results.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            if 'overall_performance' in data:
                historical_data.append({
                    'date': result_file.parent.name,
                    'score': data['overall_performance']['score'],
                    'training_throughput': data.get('training_perf_medium_config', {}).get('throughput_samples_per_sec', 0),
                })
    
    # Plot trends
    if historical_data:
        dates = [d['date'] for d in historical_data]
        scores = [d['score'] for d in historical_data]
        throughput = [d['training_throughput'] for d in historical_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(dates, scores, 'o-', label='Overall Score')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Trend Analysis')
        ax1.grid(True)
        
        ax2.plot(dates, throughput, 's-', color='orange', label='Training Throughput')
        ax2.set_ylabel('Samples/Second')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig('performance_trends.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Analyzed {len(historical_data)} historical benchmark results")

# Usage
analyze_performance_trends()
```

### 2. Comparative Analysis
```python
def compare_configurations():
    """Compare performance across different configurations"""
    
    configurations = [
        {'batch_size': 16, 'window_size': 512, 'name': 'small'},
        {'batch_size': 32, 'window_size': 1024, 'name': 'medium'},
        {'batch_size': 64, 'window_size': 2048, 'name': 'large'},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"Testing configuration: {config['name']}")
        
        benchmark = AdvancedPerformanceBenchmark(save_dir=f"./results_{config['name']}")
        
        # Override default config
        benchmark.create_mock_config = lambda **kwargs: benchmark.create_mock_config(
            batch_size=config['batch_size'],
            window_size=config['window_size'],
            **kwargs
        )
        
        # Run specific tests
        benchmark.benchmark_training_performance()
        
        results[config['name']] = {
            'config': config,
            'results': benchmark.results
        }
    
    # Generate comparison report
    print("\nConfiguration Comparison:")
    print("-" * 50)
    
    for name, data in results.items():
        training_result = data['results'].get('training_perf_medium_config', {})
        throughput = training_result.get('throughput_samples_per_sec', 0)
        batch_time = training_result.get('avg_batch_time_ms', 0)
        
        print(f"{name:10s}: {throughput:6.1f} samples/s, {batch_time:6.1f}ms/batch")

# Usage
compare_configurations()
```

These examples demonstrate the flexibility and power of the ContrastiveIDTask performance benchmark suite. You can adapt them to your specific needs and integrate them into your development workflow.