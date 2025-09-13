# Bug Verification Plan: Multi-Task Foundation Model Failures

## Verification Strategy

This document outlines the testing approach to verify fixes for the multi-task foundation model failures affecting TimesNet, PatchTST, and FNO backbones.

## Test Scenarios

### 1. Memory Usage Verification
**Objective**: Ensure models fit within GPU memory limits

**Test Cases**:
- [ ] B_06_TimesNet: Load model without OOM
- [ ] B_09_FNO: Complete forward pass without OOM  
- [ ] B_09_FNO: Complete backward pass without OOM
- [ ] All models: Memory usage < 20GB (safety margin)

**Success Criteria**:
- Models initialize successfully
- Training progresses past sanity check
- Memory monitoring shows usage below threshold
- No CUDA OOM errors

### 2. Configuration Parameter Validation  
**Objective**: Verify all required parameters are accessible

**Test Cases**:
- [ ] B_08_PatchTST: `cfg.e_layers` accessible during init
- [ ] B_08_PatchTST: Model initialization completes
- [ ] B_08_PatchTST: All config parameters properly mapped
- [ ] ConfigWrapper: Parameter access validation

**Success Criteria**:
- No AttributeError for missing parameters
- Model builds successfully  
- All backbone-specific parameters available
- Configuration validation passes

### 3. Tensor Shape Consistency
**Objective**: Ensure loss calculations have matching tensor dimensions

**Test Cases**:
- [ ] B_09_FNO: No shape mismatch warnings
- [ ] Multi-task loss: All task outputs have correct shapes
- [ ] RUL prediction: Target tensor has batch dimension
- [ ] Loss computation: No broadcasting warnings

**Success Criteria**:
- No UserWarning about tensor size differences
- MSE loss computation proceeds without issues
- All task losses have consistent shapes
- Training loss decreases normally

### 4. End-to-End Training Verification
**Objective**: Complete multi-task training pipeline

**Test Cases**:
- [ ] B_06_TimesNet: Complete 1 epoch successfully  
- [ ] B_08_PatchTST: Complete 1 epoch successfully
- [ ] B_09_FNO: Complete 1 epoch successfully
- [ ] All models: Multi-task loss convergence
- [ ] All models: Validation step completion

**Success Criteria**:
- Training progresses through full epoch
- Validation runs without errors  
- Multi-task metrics computed correctly
- Model checkpointing works

## Memory Profiling Tests

### GPU Memory Monitoring
```bash
# Run with memory profiling
nvidia-smi dmon -s pucvmet -d 5 -c 20 > memory_profile.log &
python main_LQ.py --config_path multitask_B_06_TimesNet.yaml
```

**Expected Metrics**:
- Peak memory usage < 20GB
- Memory growth stabilizes after initial allocation
- No memory fragmentation issues

### CPU Memory Tracking
```bash
# Monitor system memory during data loading
free -h && python main_LQ.py --config_path multitask_B_09_FNO.yaml
```

## Regression Testing

### Existing Functionality
- [ ] B_04_Dlinear: Still works after fixes
- [ ] Single-task experiments: Unaffected by changes
- [ ] Data factory: No performance degradation
- [ ] Configuration loading: Backwards compatible

### Performance Benchmarks
- [ ] B_04_Dlinear: Training time within 10% of baseline
- [ ] Memory usage: Fixed models use reasonable resources
- [ ] Batch processing: Throughput maintained

## Test Execution Plan

### Phase 1: Quick Validation (30 minutes)
1. **Configuration Test**: Verify parameter access
2. **Memory Test**: Check model loading only  
3. **Shape Test**: Validate tensor dimensions in forward pass

### Phase 2: Integration Testing (2 hours)
1. **Training Test**: Run 1 epoch for each model
2. **Multi-task Test**: Verify all task heads function
3. **Checkpoint Test**: Save/load model states

### Phase 3: Stress Testing (4 hours)  
1. **Long Training**: Multiple epochs to check stability
2. **Memory Stress**: Large batch sizes within limits
3. **Performance Test**: Compare against working baseline

## Acceptance Criteria

### Must-Pass Requirements
- [ ] **Zero OOM Errors**: All models load and train
- [ ] **Configuration Complete**: No missing parameter errors
- [ ] **Shape Consistency**: No tensor dimension warnings
- [ ] **Training Progress**: All models complete epochs successfully

### Performance Requirements  
- [ ] **Memory Efficiency**: <20GB GPU usage per model
- [ ] **Speed Maintained**: Training time comparable to B_04_Dlinear
- [ ] **Quality Preserved**: Model accuracy not significantly degraded

### Stability Requirements
- [ ] **Reproducible**: Multiple runs produce consistent results  
- [ ] **Robust**: No intermittent failures or crashes
- [ ] **Scalable**: Works with different batch sizes

## Test Environment

### Hardware Requirements
- NVIDIA RTX 3090 24GB (minimum)
- 32GB+ system RAM
- SSD storage for data loading

### Software Dependencies
- PyTorch with CUDA support
- PyTorch Lightning
- WandB for logging (optional)
- Memory profiling tools

## Failure Response Plan

### If Memory Tests Fail
1. Further reduce model dimensions
2. Implement gradient checkpointing
3. Consider model parallelism

### If Configuration Tests Fail  
1. Add missing parameters to YAML
2. Update ConfigWrapper mappings
3. Validate parameter inheritance

### If Shape Tests Fail
1. Debug tensor creation points
2. Add explicit shape transformations
3. Implement shape validation helpers

---

**Test Lead**: Development Team  
**Timeline**: 1-2 days depending on fix complexity  
**Success Threshold**: 100% of must-pass requirements  
**Documentation**: Results logged in verification-results.md