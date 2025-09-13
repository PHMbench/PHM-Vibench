# Bug Analysis

## Root Cause Analysis

### Investigation Summary

Conducted comprehensive analysis of foundation model training failures in script/Vibench_paper/foundation_model/ directory. Investigation revealed three interconnected issues:

1. **Memory Exhaustion**: DLinear model crashed with OOM error at Epoch 176/200
2. **Configuration Bottlenecks**: Resource allocation insufficient for multi-task learning overhead
3. **Training Inefficiency**: Complex models (PatchTST, TimesNet, FNO) running >24 hours due to suboptimal hyperparameters

Analysis involved examining SLURM logs, configuration files, multi-task training pipeline, and model architectures to identify failure patterns.

### Root Cause

**Primary Cause: Memory Configuration Mismatch for Multi-Task Learning**

The fundamental issue is that the SLURM resource allocation and model configurations were not designed to handle the memory overhead of **simultaneous 4-task learning** (classification, anomaly_detection, signal_prediction, rul_prediction).

**Technical Details:**

- **Multi-Task Memory Multiplication**: Each task requires separate loss computation, metrics tracking, and gradient computation
- **Batch Size Amplification**: Effective memory usage = batch_size × num_tasks × model_size × sequence_length
- **Memory Fragmentation**: PyTorch's memory allocator struggles with concurrent tensor operations across tasks

### Contributing Factors

#### 1. SLURM Resource Under-Allocation

```bash
# Current allocation (insufficient)
#SBATCH --mem=48G              # Too small for multi-task
#SBATCH --gpus=v100:1          # V100 has only 16GB VRAM
#SBATCH --cpus-per-gpu=8       # Insufficient for num_workers=32
```

#### 2. Aggressive Batch Size Configurations

```yaml
# DLinear configuration (problematic)
data:
  batch_size: 384      # Too large for multi-task
  num_workers: 32      # Excessive worker processes
  
# PatchTST configuration (even worse)  
data:
  batch_size: 512      # Extremely large
  num_window: 128      # High window count
```

#### 3. Multi-Task Training Pipeline Overhead

From `/src/task_factory/task/In_distribution/multi_task_phm.py`:

- Simultaneous forward pass for 4 tasks
- Individual loss computation per task
- Separate metric tracking (16 metrics × 4 tasks = 64 metrics)
- Gradient accumulation across all tasks

#### 4. Model Architecture Memory Requirements

**DLinear** (paradoxically memory-heavy despite being "simple"):

- Creates duplicate tensors for seasonal/trend decomposition
- Instantiates separate linear layers per channel when individual=True
- Memory allocation for moving average operations

**PatchTST/TimesNet/FNO** (complex architectures):

- Multi-head attention mechanisms
- Large intermediate representations
- Complex patch embedding strategies
- large parameters config

## Technical Details

### Affected Code Locations

- **File**: `/script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`

  - **Lines**: 23, 24, 28 (batch_size, num_workers, num_window)
  - **Issue**: Batch size (384) too large for V100 GPU with 48GB RAM
- **File**: `/script/Vibench_paper/foundation_model/run_dlinear.sbatch`

  - **Lines**: 4, 6, 7 (GPU type, memory, time allocation)
  - **Issue**: V100 GPU insufficient, 48GB RAM too small, 24h time limit inadequate
- **File**: `/src/task_factory/task/In_distribution/multi_task_phm.py`

  - **Function**: `_shared_step()` (lines ~550-600)
  - **Issue**: No gradient accumulation support for large effective batch sizes
- **File**: `/src/model_factory/ISFM/backbone/B_04_Dlinear.py`

  - **Lines**: 84-92 (tensor allocation in forward pass)
  - **Issue**: Creates large intermediate tensors without memory optimization

### Data Flow Analysis

```
1. DataLoader creates batches: batch_size × sequence_length × channels
                                      ↓
2. Multi-task pipeline processes: 4 × (batch × sequence × features)
                                      ↓
3. Model forward pass: embedding → backbone → 4 task heads
                                      ↓
4. Loss computation: 4 separate loss tensors + gradients
                                      ↓
5. Metrics computation: 64 metric tensors stored in GPU memory
                                      ↓
6. Memory fragmentation leads to OOM at peak usage
```

### Dependencies

- **PyTorch Lightning**: Memory management and gradient handling
- **TorchMetrics**: Large number of metric objects consuming GPU memory
- **H5py**: Dataset loading creates memory pressure
- **SLURM**: Resource constraints limit available memory

## Impact Analysis

### Direct Impact

1. **Experimental Failure**: Foundation model experiments cannot complete successfully
2. **Resource Waste**: 24+ hours of GPU time consumed without results
3. **Research Delay**: Paper reproduction timeline significantly impacted
4. **Dataset Processing Incomplete**: Only 3/N datasets processed by DLinear

### Indirect Impact

1. **Reproducibility Crisis**: Unable to reproduce paper results
2. **Confidence Loss**: Uncertainty about framework's production readiness
3. **Resource Competition**: Other researchers blocked from GPU access
4. **Method Validation**: Cannot validate multi-task learning approach

### Risk Assessment

- **High Risk**: Continued failures will block entire foundation model research track
- **Medium Risk**: Configuration issues may propagate to other experiments
- **Low Risk**: Single-task experiments likely unaffected

## Solution Approach

### Fix Strategy

**Multi-Pronged Memory Optimization with Gradient Accumulation**

1. **Hardware Resource Scaling**

   - Upgrade from V100 (16GB VRAM) to A100 (80GB VRAM)
   - Increase system RAM from 48GB to 96GB
   - Use A100 nodes with high memory bandwidth
2. **Smart Batch Size Management**

   - Reduce physical batch sizes to fit in memory
   - Implement gradient accumulation for large effective batches
   - Dynamic batch sizing based on available memory
3. **Multi-Task Training Optimization**

   - Task-wise gradient accumulation
   - Memory-efficient metric computation
   - Selective task execution during training phases
4. **Model-Specific Memory Optimization**

   - DLinear: Optimize tensor allocation patterns
   - PatchTST: Reduce attention dimension and patch count
   - Enable gradient checkpointing for complex models

### Alternative Solutions

#### Option 1: Sequential Task Training

Train one task at a time instead of simultaneous multi-task learning.
**Pros**: Lower memory requirements
**Cons**: Loses multi-task synergy benefits

#### Option 2: Model Distillation

Train smaller models that mimic larger ones.
**Pros**: Faster training, lower resource usage
**Cons**: May sacrifice performance quality

#### Option 3: Mixed Precision Training

Use FP16 instead of FP32 for reduced memory usage.
**Pros**: ~50% memory reduction
**Cons**: Potential numerical instability

### Risks and Trade-offs

- **Performance vs Memory**: Smaller batch sizes may impact convergence
- **Training Time vs Resources**: More resources reduce training time but increase cost
- **Model Complexity vs Stability**: Simplified models may be more stable but less expressive

## Implementation Plan

### Changes Required

1. **SLURM Configuration Updates**

   - File: `run_*.sbatch` (all 4 scripts)
   - Modification: Upgrade GPU to A100, increase memory to 96GB, adjust CPU allocation
2. **Model Configuration Optimization**

   - File: `multitask_*.yaml` (all 4 configs)
   - Modification: Reduce batch sizes, implement gradient accumulation, optimize model dimensions
3. **Training Pipeline Enhancement**

   - File: `/src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Add gradient accumulation support, memory-efficient metric computation
4. **Memory Monitoring Infrastructure**

   - File: New `/script/Vibench_paper/foundation_model/utils/memory_monitor.py`
   - Modification: Create GPU/RAM monitoring tools for proactive memory management

### Testing Strategy

#### Unit Testing

- Test gradient accumulation with small synthetic datasets
- Verify memory optimization doesn't affect model outputs
- Validate metric computation accuracy with reduced memory footprint

#### Integration Testing

- Run single-epoch tests on each model with optimized configurations
- Monitor memory usage throughout training pipeline
- Test checkpoint recovery from OOM scenarios

#### Performance Testing

- Compare training speed before/after optimizations
- Measure memory usage reduction
- Validate final model performance on test sets

### Rollback Plan

1. **Configuration Rollback**: Keep original configurations as `.backup` files
2. **Resource Restoration**: Return to V100 nodes if A100 unavailable
3. **Single-Task Fallback**: Disable multi-task learning if memory issues persist
4. **Model Simplification**: Use smallest model dimensions that fit in available memory

## Notes

Investigation revealed that "simple" models like DLinear can actually be memory-intensive in multi-task scenarios due to:

- Decomposition operations creating duplicate tensors
- Individual linear layers per channel
- Lack of memory optimization in forward pass

The solution requires both hardware upgrades and software optimizations to handle the 4× memory multiplication factor of simultaneous multi-task learning.

**Next Steps**: Present this analysis for approval, then proceed to implement the memory optimization fixes in order of priority: SLURM configs → model configs → training pipeline → monitoring tools.
