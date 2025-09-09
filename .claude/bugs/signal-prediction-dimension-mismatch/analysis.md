# Bug Analysis

## Root Cause Analysis

### Investigation Summary
The signal prediction task failed consistently during multi-task PHM training due to tensor dimension mismatch between model output and target signals. The investigation revealed a configuration-level constraint that limited output channels for memory efficiency, creating incompatibility with multi-channel input data.

### Root Cause
**Primary Issue**: Memory optimization configurations set `max_out: 2` in H_03_Linear_pred to prevent memory overflow, but datasets contain 3-channel vibration signals (XYZ components), causing tensor dimension mismatch during MSE loss computation.

**Configuration Evidence**:
```yaml
# Found in foundation model configs
max_out: 2         # MEMORY FIX: Reduced from 3 to 2
```

**Error Pattern**:
```
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2
Task output shape: torch.Size([16, 4096, 2])  # Model limited to 2 channels
Target shape: torch.Size([16, 4096, 3])       # Input has 3 channels
```

### Contributing Factors
1. **Memory Constraints**: GPU memory limitations necessitated `max_out=2` reduction
2. **Multi-channel Data**: Industrial vibration datasets commonly have 3 spatial components
3. **Signal Reconstruction Logic**: Signal prediction uses input as target for reconstruction
4. **Rigid Validation**: No graceful handling of dimension mismatches in loss computation

## Technical Details

### Affected Code Locations

- **File**: `src/model_factory/ISFM/task_head/H_03_Linear_pred.py`
  - **Lines**: 54-59
  - **Issue**: Rigid constraint checking without graceful degradation
  - **Fix**: Added warning message and graceful channel limiting

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Lines**: 509-532
  - **Issue**: No dimension compatibility checking before loss computation
  - **Fix**: Added automatic target channel truncation

- **File**: Multiple config files (foundation model configs)
  - **Parameter**: `max_out: 2`
  - **Issue**: Memory constraint creates channel limitation
  - **Status**: Preserved - fix adapts to constraint rather than removing it

### Data Flow Analysis
1. **Input**: Multi-channel signal `[B, L, 3]` enters model
2. **Shape Parameter**: M_01_ISFM correctly extracts `(4096, 3)` from input
3. **Constraint**: H_03_Linear_pred limits output to `max_out=2` channels
4. **Output**: Model produces `[B, 4096, 2]` prediction
5. **Loss Computation**: MSE attempted between `[B, 4096, 2]` and `[B, 4096, 3]`
6. **Error**: PyTorch tensor dimension mismatch at channel dimension

### Dependencies
- **PyTorch MSE Loss**: Requires identical tensor dimensions
- **Memory Management**: GPU memory constraints drive max_out limitation
- **Multi-task Training**: Signal prediction is one of multiple concurrent tasks

## Impact Analysis

### Direct Impact
- Signal prediction task completely non-functional
- Multi-task training interrupted with RuntimeError
- Loss computation impossible for signal reconstruction
- Training metrics unavailable for signal prediction

### Indirect Impact  
- Reduced model capabilities for signal reconstruction tasks
- Poor user experience with cryptic error messages
- Development workflow disrupted
- Multi-task learning benefits diminished

### Risk Assessment
- **High Priority**: Core functionality broken
- **Wide Impact**: Affects all multi-task configurations with signal prediction
- **Memory Sensitivity**: Fix must preserve memory optimizations

## Solution Approach

### Fix Strategy
**Adaptive Channel Handling**: Implement graceful degradation that respects memory constraints while enabling signal prediction functionality.

**Key Principles**:
1. **Preserve Memory Efficiency**: Maintain max_out constraints
2. **Enable Functionality**: Allow signal prediction to proceed
3. **Transparent Operation**: Inform users about channel adaptations
4. **Robust Handling**: Handle various channel configurations

### Alternative Solutions Considered

#### Option 1: Increase max_out (Rejected)
- **Approach**: Change config to `max_out: 3`
- **Pros**: Perfect channel matching
- **Cons**: Defeats memory optimization, may cause OOM errors
- **Decision**: Rejected due to memory constraints

#### Option 2: Skip Signal Prediction (Rejected)
- **Approach**: Disable signal prediction in multi-channel scenarios
- **Pros**: Avoids dimension mismatch
- **Cons**: Loses important reconstruction capability
- **Decision**: Rejected as reduces model functionality

#### Option 3: Channel Interpolation (Rejected)
- **Approach**: Interpolate 2 output channels to 3 channels
- **Pros**: Maintains input dimensions
- **Cons**: Questionable signal processing validity, adds complexity
- **Decision**: Rejected as scientifically dubious

#### Option 4: Adaptive Truncation (Selected)
- **Approach**: Truncate target channels to match output limitations
- **Pros**: Simple, preserves memory efficiency, enables functionality
- **Cons**: Information loss in unused channels
- **Decision**: Selected as optimal balance

### Risks and Trade-offs

#### Risks
- **Information Loss**: Only first 2 of 3 channels used for reconstruction
- **Performance Impact**: Reconstruction quality may be reduced
- **User Confusion**: Channel truncation may be unexpected

#### Mitigations
- **Clear Warnings**: Inform users about channel truncation
- **Consistent Behavior**: Always use first N channels (predictable)
- **Performance Monitoring**: Log signal prediction metrics for quality assessment

## Implementation Plan

### Changes Required

1. **H_03_Linear_pred Enhancement**
   - **File**: `src/model_factory/ISFM/task_head/H_03_Linear_pred.py`
   - **Modification**: Replace rigid error with graceful warning when channels exceed max_out

2. **Multi-task Loss Adaptation**
   - **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Modification**: Add dimension compatibility checking and automatic target truncation

3. **Test Verification**
   - **File**: `test_signal_prediction_fix.py`
   - **Modification**: Create comprehensive test cases for multiple channel scenarios

### Testing Strategy
- **Unit Tests**: Verify H_03_Linear_pred handles channel constraints
- **Integration Tests**: Verify multi-task training proceeds without errors
- **Edge Case Tests**: Test 1, 2, 3+ channel configurations
- **Performance Tests**: Monitor signal prediction metrics quality

### Rollback Plan
If issues arise, changes can be reverted by:
1. Removing warning logic from H_03_Linear_pred (restore original error)
2. Removing dimension handling from multi_task_phm (restore original loss computation)
3. Configuration rollback: No config changes required for rollback