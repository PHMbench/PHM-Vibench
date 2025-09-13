# Bug Analysis

## Root Cause Analysis

### Investigation Summary
After thorough code examination, I've identified the primary root causes for the multi-task training failures:

1. **Test Script Parameter Inconsistency**: `test_model_params.py` uses `max_len=512` while YAML files correctly use `max_len=4096`
2. **Missing Metrics Infrastructure**: Multi-task implementation lacks task-specific metrics computation and logging
3. **RUL Label Validation Issues**: No validation for missing RUL labels causing NaN propagation

### Root Cause

**Primary Cause**: **Parameter Test Script Inconsistency**
- The YAML configurations correctly specify `max_len: 4096` to match input data dimensions
- But `test_model_params.py` incorrectly uses `max_len=512` for "memory optimization" 
- This creates a mismatch: model outputs (B,512,2) in test but should output (B,4096,2)
- The tensor size error causes signal prediction to fail during actual training

**Secondary Cause**: **Incomplete Multi-task Metrics Implementation**
- `multi_task_phm.py` only implements loss computation, no metrics
- Existing `get_metrics()` system supports classification metrics but not regression metrics (MSE, MAE, R2)
- No task-specific metric computation or logging in training steps

### Contributing Factors
1. **Parameter Test Mismatch**: Memory optimization test used wrong dimensions vs actual configuration
2. **Missing RUL Validation**: No validation that RUL labels exist in metadata, causing NaN when missing
3. **Missing Error Handling**: No graceful handling of tensor dimension mismatches or missing data

## Technical Details

### Affected Code Locations

- **File**: `test_model_params.py`
  - **Lines**: 171, 185, 211, 237
  - **Issue**: Uses `max_len=512` but should use `max_len=4096` to match YAML configs

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Function**: `_build_task_labels()` (lines 156-160)
  - **Issue**: RUL labels may be missing, creating invalid tensors causing NaN

- **File**: `src/task_factory/Components/metrics.py`
  - **Lines**: 18-24
  - **Issue**: Only classification metrics defined, missing MSE/MAE/R2 for regression tasks

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Missing**: No `_compute_task_metrics()` method for task-specific performance metrics

### Data Flow Analysis
1. **Input Processing**: Data loader provides (B,4096,2) tensors (correct)
2. **Model Forward**: H_03_Linear_pred should output (B,4096,2) with correct max_len
3. **Loss Computation**: With matching dimensions, MSE loss should work correctly
4. **RUL Processing**: Missing RUL labels cause NaN in RUL prediction loss
5. **Metrics**: No task-specific metrics computed, only losses logged

### Dependencies
- **PyTorch**: MSE loss requires matching tensor dimensions
- **torchmetrics**: Available metrics system supports classification, needs extension for regression
- **PyTorch Lightning**: Logging system ready for additional metrics

## Impact Analysis

### Direct Impact
- **Inconsistent Testing**: Parameter analysis doesn't reflect actual model behavior
- **RUL prediction produces NaN**: Missing labels create invalid loss calculations  
- **Total loss corrupted**: NaN propagation makes training unstable
- **No performance visibility**: Researchers can't evaluate model quality beyond losses

### Indirect Impact  
- **Research productivity loss**: No accuracy/F1/AUC/MSE/R2 metrics available for analysis
- **Model comparison impossible**: Only loss values available, no standardized metrics
- **Debugging difficulty**: No insight into per-task performance patterns
- **Memory analysis incorrect**: Test script parameters don't match actual usage

### Risk Assessment
- **High**: RUL NaN propagation makes training unstable
- **Medium**: Missing metrics severely limit model evaluation
- **Low**: Signal prediction should work with correct dimensions

## Solution Approach

### Fix Strategy
**Three-Phase Fix Approach:**

**Phase 1: Fix Parameter Consistency**
- Correct `test_model_params.py` to use `max_len=4096` matching YAML files
- Verify memory usage with correct parameters
- Add validation for RUL label availability in multi_task_phm.py

**Phase 2: Implement Missing Metrics**
- Extend `metrics.py` to support regression metrics (MSE, MAE, R2, MAPE)  
- Add task-specific metric computation methods in `multi_task_phm.py`
- Integrate metric logging in training/validation/test steps for all tasks

**Phase 3: Robust Error Handling**
- Add graceful handling for missing RUL labels (use default value)
- Add comprehensive validation and logging for debugging
- Add fallback mechanisms for missing metadata

### Alternative Solutions
**Alternative 1**: Reduce input data to 512 timesteps
- **Pros**: Matches optimized test parameters
- **Cons**: Data loss, requires data pipeline changes, defeats purpose of using full signals

**Alternative 2**: Keep inconsistent parameters
- **Pros**: No changes needed
- **Cons**: Incorrect memory analysis, potential dimension mismatches

**Chosen Approach**: **Correct parameter consistency with max_len=4096** - preserves full signal information and matches actual configuration

### Risks and Trade-offs
- **Risk**: Slightly higher memory usage than test script suggested
- **Mitigation**: Memory optimization already achieved through hidden_dim=64 reduction (99.9% parameter reduction)
- **Trade-off**: Accurate testing vs optimized memory (accuracy more important)

## Implementation Plan

### Changes Required

1. **Fix Parameter Test Script**
   - File: `test_model_params.py`
   - Modification: Change all `max_len=512` to `max_len=4096` to match YAML configurations

2. **Add RUL Label Validation**
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`  
   - Modification: Add validation and default handling in `_build_task_labels()` for missing RUL

3. **Extend Metrics System**
   - File: `src/task_factory/Components/metrics.py`
   - Modification: Add MSE, MAE, R2, MAPE metrics to metric_classes dictionary

4. **Implement Multi-Task Metrics**
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Add `_compute_task_metrics()` method and integrate in `_shared_step()`

5. **Add Task-Specific Metric Logging**
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Log metrics for each task in training/validation/test phases

### Memory Impact Assessment
With corrected parameters:
- **Signal prediction head**: hidden²(64²=4096) × max_len(4096) × max_out(2) = **33.5M parameters**
- **Total model**: ~34-184M parameters (depending on backbone)
- **Memory usage**: 2-6GB (much better than original 48GB, still fits in 24GB GPU)

### Testing Strategy
1. **Parameter Verification**: Re-run corrected test script to verify memory usage
2. **Dimension Tests**: Verify all tensor operations work with 4096 timesteps
3. **Integration Tests**: Run all four backbone models with corrected parameters
4. **Metrics Tests**: Verify all task-specific metrics are computed and logged correctly

### Rollback Plan
- **Immediate**: Revert test script changes if memory issues arise
- **Configuration**: YAML files remain unchanged (already correct)
- **Code**: Use git to revert specific commits if needed
- **Monitoring**: Track actual GPU memory usage during training