# Bug Analysis

## Root Cause Analysis

### Investigation Summary
The signal prediction metrics computation fails because the recently implemented channel truncation fix for loss computation was not applied to the metrics computation pathway. While `_compute_task_loss()` properly truncates 3-channel targets to match 2-channel predictions, `_compute_task_metrics()` still receives the original full-channel targets, causing shape mismatch in torchmetrics validation.

### Root Cause
**Primary Issue**: Inconsistent channel handling between loss and metrics computation in multi-task training.

The loss computation in `_compute_task_loss()` (lines 682-695 in multi_task_phm.py) includes dimension mismatch handling:
```python
if task_output.shape[-1] != targets.shape[-1]:
    target_channels = targets.shape[-1]
    output_channels = task_output.shape[-1]
    
    if output_channels < target_channels:
        print(f"Info: Truncating target channels from {target_channels} to {output_channels}")
        targets = targets[..., :output_channels]
```

However, the metrics computation in `_compute_task_metrics()` lacks this same handling, causing torchmetrics to receive mismatched tensor shapes.

### Contributing Factors
1. **Memory Optimization Constraints**: H_03_Linear_pred configured with max_out=2 to prevent GPU OOM
2. **Multi-channel Input Data**: Industrial vibration datasets commonly have 3 spatial components (XYZ)
3. **Separate Code Paths**: Loss and metrics computation use different methods, requiring duplicate fixes
4. **Recent Implementation**: System-specific metrics tracking was recently added, increasing metrics computation usage

## Technical Details

### Affected Code Locations

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Method**: `_compute_task_metrics()` (lines ~250-300)
  - **Issue**: Missing channel dimension validation and truncation
  - **Current**: Receives full 3-channel targets, compares with 2-channel predictions
  - **Fix**: Apply same channel truncation logic as in loss computation

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`  
  - **Method**: `_shared_step()` (lines 612-616)
  - **Issue**: Passes original targets to metrics without channel adjustment
  - **Current**: `task_metrics = self._compute_task_metrics(task_name, task_output, metric_targets, mode)`
  - **Fix**: Apply channel truncation to metric_targets before metrics computation

### Data Flow Analysis
1. **Input**: Multi-channel signal `[B, L, 3]` enters multi-task training
2. **Model Output**: H_03_Linear_pred produces `[B, L, 2]` due to max_out=2 constraint
3. **Loss Path**: `_compute_task_loss()` truncates targets to `[B, L, 2]` → Loss computed successfully
4. **Metrics Path**: `_compute_task_metrics()` receives original `[B, L, 3]` targets → Shape mismatch error
5. **Result**: Training continues but metrics are not computed/logged

### Dependencies
- **torchmetrics**: Strict tensor shape validation (MeanSquaredError, MeanAbsoluteError, R2Score)
- **Memory Management**: max_out=2 constraint must be preserved
- **Multi-task Framework**: Both loss and metrics need consistent handling

## Impact Analysis

### Direct Impact
- Signal prediction metrics unavailable during training
- Incomplete training monitoring and debugging information
- System-specific metrics tracking partially broken for signal_prediction
- User confusion due to repeated warning messages

### Indirect Impact  
- Reduced ability to tune signal prediction performance
- Incomplete experimental logs for research reproducibility
- Potential masking of other signal prediction issues
- Degraded user experience with training feedback

### Risk Assessment
- **Medium Priority**: Training still proceeds, but monitoring is impaired
- **Wide Scope**: Affects all multi-task configurations with signal prediction
- **Memory Safety**: Fix must preserve existing memory optimizations

## Solution Approach

### Fix Strategy
**Unified Channel Handling**: Implement the same dimension mismatch detection and target truncation logic in metrics computation that already exists in loss computation.

**Key Principles**:
1. **Consistency**: Metrics and loss should handle channel mismatches identically
2. **Memory Preservation**: Maintain max_out constraints and memory optimizations
3. **Clean Implementation**: Avoid code duplication through shared utility function
4. **Backward Compatibility**: Ensure fix works with various channel configurations

### Alternative Solutions

#### Option 1: Shared Utility Function (Selected)
- **Approach**: Create `_handle_channel_mismatch()` utility used by both loss and metrics
- **Pros**: DRY principle, consistent behavior, easier maintenance
- **Cons**: Minor refactoring required
- **Decision**: Selected for long-term maintainability

#### Option 2: Direct Duplication (Rejected)
- **Approach**: Copy-paste channel handling code to metrics method
- **Pros**: Simple, minimal changes
- **Cons**: Code duplication, maintenance burden, inconsistency risk
- **Decision**: Rejected due to maintainability concerns

#### Option 3: Pre-process in _shared_step (Rejected)
- **Approach**: Adjust metric_targets in _shared_step before calling _compute_task_metrics
- **Pros**: Single adjustment point
- **Cons**: Less clear, affects metrics computation interface
- **Decision**: Rejected for clarity and modularity

### Risks and Trade-offs

#### Risks
- **Information Loss**: Only first 2 of 3 channels used for metrics (same as loss)
- **Performance Impact**: Minimal additional processing for metrics computation
- **Code Complexity**: Slightly more complex metrics computation logic

#### Mitigations  
- **Consistent Behavior**: Same information available in both loss and metrics
- **Clear Logging**: Inform users when channel truncation occurs
- **Documentation**: Update code comments explaining channel handling

## Implementation Plan

### Changes Required

1. **Create Shared Utility Method**
   - **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Method**: `_handle_channel_mismatch(task_output, targets, task_name)`
   - **Purpose**: Centralized logic for handling dimension mismatches

2. **Update Loss Computation**
   - **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Method**: `_compute_task_loss()` (lines 682-695)
   - **Change**: Use shared utility instead of inline logic

3. **Update Metrics Computation**
   - **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Method**: `_compute_task_metrics()`
   - **Change**: Add channel mismatch handling using shared utility

### Testing Strategy
- **Unit Tests**: Verify channel truncation works correctly for various input shapes
- **Integration Tests**: Ensure both loss and metrics compute successfully with mismatched channels
- **Regression Tests**: Confirm existing functionality remains intact
- **Multi-channel Tests**: Test with 1, 2, 3, and 4+ channel inputs

### Rollback Plan
If issues arise, changes can be reverted by:
1. Removing the shared utility function
2. Restoring original loss computation code
3. Leaving metrics computation unchanged (current broken state)
4. Configuration rollback not required