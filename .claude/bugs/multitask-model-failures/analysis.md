# Bug Analysis: Multi-Task Foundation Model Failures

## Root Cause Analysis

### Issue 1: Memory Overflow (TimesNet B_06, FNO B_09)

**Root Cause**: Model parameter explosion in multi-task scenarios
- **TimesNet**: 5.6B parameters (22.5GB) exceeds 24GB GPU capacity
- **FNO**: 3.2B parameters (12.9GB) + gradient computation causes OOM during backprop
- **Contributing Factor**: Multi-task heads add significant parameter overhead

**Memory Breakdown**:
```
Model Weights: 13-22GB
Activations: 2-8GB  
Gradients: Same as weights
Optimizer States: 2x weights (AdamW)
Total: 40-80GB estimated vs 24GB available
```

### Issue 2: Configuration Parameter Mismatch (PatchTST B_08)

**Root Cause**: Missing `e_layers` parameter in configuration mapping
- **Location**: `B_08_PatchTST.py:61` expects `cfg.e_layers`
- **Missing From**: YAML configurations don't define `e_layers`
- **ConfigWrapper Issue**: Parameter not accessible through wrapper

**Expected vs Actual**:
```python
# Expected in PatchTST __init__
for _ in range(cfg.e_layers)  # cfg.e_layers undefined

# Available in config
num_layers: 3  # But mapped incorrectly
```

### Issue 3: Tensor Shape Mismatch (FNO B_09)

**Root Cause**: Multi-task loss calculation dimension mismatch
- **Warning**: Target size `torch.Size([])` vs input size `torch.Size([128])`  
- **Impact**: MSE loss expects matching dimensions
- **Location**: Multi-task RUL prediction task

**Shape Analysis**:
```python
# Expected: both [batch_size] or [batch_size, 1]
input: torch.Size([128])    # Prediction tensor
target: torch.Size([])      # Scalar target - WRONG
```

## Technical Deep Dive

### Memory Optimization Strategies

1. **Gradient Accumulation**:
   - Current: `accumulate_grad_batches: 2` insufficient
   - Needed: 8-16 accumulation steps for large models

2. **Model Sharding**:
   - Consider DeepSpeed ZeRO for parameter distribution
   - Offload optimizer states to CPU

3. **Mixed Precision**:
   - Already enabled: `precision: 16`
   - Could add gradient scaling optimization

### Configuration System Issues

1. **Parameter Mapping**:
   ```yaml
   # Missing in YAML configs
   e_layers: 6        # Required by PatchTST
   d_layers: 1        # Decoder layers
   factor: 1          # Attention factor
   ```

2. **ConfigWrapper Limitations**:
   - Some parameters not properly wrapped
   - Need attribute mapping validation

### Multi-Task Architecture Problems

1. **Task Head Overhead**:
   - 4 task heads add significant parameters
   - Each head has hidden_dim=512 parameters
   - Multiplicative effect on memory usage

2. **Loss Shape Handling**:
   ```python
   # Current problematic code
   target: float  # Scalar value
   pred: torch.Tensor([batch_size])  # Batch predictions
   
   # Fixed approach needed
   target: torch.Tensor([batch_size, 1])  # Proper shape
   ```

## Impact Analysis

### Immediate Impact
- **Development Blocked**: Cannot evaluate 3/4 backbone models
- **Resource Waste**: 41+ minutes of failed GPU compute
- **False Reporting**: Success messages mask critical failures

### Long-term Impact  
- **Research Validity**: Cannot compare backbone architectures
- **Production Risk**: Memory issues will persist in larger deployments
- **Scalability**: Models won't work on similar GPU configurations

### Affected Components
1. **Model Factory**: Backbone initialization failures
2. **Task Factory**: Multi-task loss calculation errors
3. **Configuration System**: Parameter mapping incomplete
4. **Experiment Scripts**: Success reporting misleading

## Solution Strategy

### Phase 1: Memory Optimization (High Priority)
1. **Reduce Model Dimensions**:
   - Cut `output_dim: 1024 → 256`
   - Reduce `num_heads: 8 → 4`
   - Decrease `d_ff: 2048 → 512`

2. **Batch Size Reduction**:
   - Standard: `batch_size: 128 → 32`
   - Increase gradient accumulation accordingly

### Phase 2: Configuration Fixes (High Priority)
1. **Add Missing Parameters**:
   ```yaml
   # Add to all PatchTST configs
   e_layers: 3
   d_layers: 1  
   factor: 1
   ```

2. **Validate ConfigWrapper**: Ensure all parameters accessible

### Phase 3: Shape Corrections (Medium Priority)
1. **Fix Loss Tensor Shapes**:
   - Ensure target tensors have batch dimension
   - Add shape validation before loss computation

2. **Multi-task Loss Refactor**:
   - Standardize tensor dimensions across tasks
   - Add shape debugging information

### Phase 4: Reporting Fixes (Low Priority)  
1. **Improve Error Detection**: Catch exceptions and report failures correctly
2. **Add Memory Monitoring**: Track GPU usage during training

## Risk Assessment

### Implementation Risks
- **Performance Degradation**: Dimension reductions may hurt model quality
- **Compatibility**: Configuration changes might break other experiments
- **Validation**: Need comprehensive testing across all models

### Mitigation Strategies
- **Gradual Rollout**: Fix one model type at a time
- **Regression Testing**: Verify B_04_Dlinear continues working
- **Memory Profiling**: Monitor actual vs estimated memory usage

---

**Status**: Analysis Complete  
**Next Phase**: Fix Implementation  
**Priority**: Critical - Blocking Research Progress