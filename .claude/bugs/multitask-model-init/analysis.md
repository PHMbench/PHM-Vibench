# Bug Analysis

## Root Cause Analysis

### Investigation Summary
Conducted comprehensive analysis of multi-task model initialization failure by examining the full call stack, analyzing affected code components, and reviewing existing patterns in the codebase. Investigation confirmed that XJTU dataset uses -1 labels for samples that don't participate in classification training, and identified established filtering patterns already implemented in other components.

### Root Cause
The `get_num_classes()` method in `M_01_ISFM.py` (lines 66-70) fails to handle datasets with NaN values and -1 labels. The method directly applies `max()` to pandas Series containing these invalid values:

```python
num_classes[key] = max(self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']) + 1
```

When `max()` encounters NaN values, it returns NaN, which propagates through the calculation and ultimately causes PyTorch's `nn.Linear` layer to fail with TypeError.

### Contributing Factors
1. **Industrial dataset conventions**: Different datasets use -1 for non-training samples (confirmed: XJTU uses -1 for samples not participating in classification training)
2. **Mixed label formats**: Some datasets contain NaN values for unlabeled samples  
3. **No input validation**: Missing preprocessing to filter invalid label values
4. **Inconsistent with existing patterns**: Other components already implement proper -1 filtering (e.g., `src/data_factory/ID/Get_id.py:19`)

## Technical Details

### Affected Code Locations

- **File**: `src/model_factory/ISFM/M_01_ISFM.py`
  - **Function/Method**: `get_num_classes()`
  - **Lines**: `66-70`
  - **Issue**: Direct max() operation without filtering invalid values

- **File**: `src/model_factory/ISFM/task_head/multi_task_head.py`
  - **Function/Method**: `_build_classification_heads()`
  - **Lines**: `131`
  - **Issue**: `nn.Linear(self.hidden_dim // 2, n_classes)` receives None when n_classes is NaN

### Data Flow Analysis
1. **Metadata Loading**: DataFrame contains Label column with mixed values (0,1,2,3,-1,NaN)
2. **Model Initialization**: `M_01_ISFM.__init__()` calls `get_num_classes()` (line 62)
3. **Class Calculation**: For each Dataset_id, `max()` applied to all Label values including invalid ones
4. **NaN Propagation**: NaN values cause `max()` to return NaN, then NaN + 1 = NaN
5. **Dictionary Storage**: num_classes dictionary stores NaN values
6. **Head Creation**: MultiTaskHead receives NaN, converts to None for PyTorch
7. **PyTorch Error**: `nn.Linear` validates arguments and raises TypeError

### Dependencies Analysis
- **pandas**: Series.max() returns NaN when any values are NaN
- **numpy**: NaN arithmetic propagation (NaN + 1 = NaN)
- **PyTorch**: Strict type checking in nn.Linear constructor

### Existing Pattern Analysis
Investigation revealed the codebase already has established patterns for handling -1 labels:

```python
# From src/data_factory/ID/Get_id.py:19
filtered_df = df[df[label_column] != -1].copy()

# Function: remove_invalid_labels() (lines 3-24)
def remove_invalid_labels(df, label_column='Label'):
    filtered_df = df[df[label_column] != -1].copy()
```

This confirms that filtering -1 labels is the correct approach and aligns with existing codebase conventions.

## Impact Analysis

### Direct Impact
- **Critical system failure**: All multi-task model initialization fails
- **Pipeline blockage**: Cannot run foundation model experiments  
- **Training impossibility**: 4 planned models (B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO) all affected

### Indirect Impact  
- **Research disruption**: Multi-task learning experiments completely blocked
- **Code quality concerns**: Suggests similar vulnerabilities in other label-processing code
- **Data integrity questions**: Highlights need for robust invalid data handling

### Risk Assessment
- **High severity**: Core functionality completely non-functional
- **Medium scope**: Affects all multi-task workflows but not single-task
- **Low complexity**: Fix is straightforward and low-risk

## Solution Approach

### Fix Strategy
Implement robust label filtering following existing codebase patterns:
1. **Filter invalid values**: Remove NaN and -1 labels before calculation
2. **Use established pattern**: Follow `Get_id.py` filtering approach
3. **Provide defaults**: Use binary classification (2 classes) when no valid labels exist
4. **Maintain compatibility**: Preserve existing behavior for valid datasets

### Alternative Solutions Considered
1. **Metadata preprocessing**: Clean labels during data loading (too invasive)
2. **Configuration override**: Allow manual class specification (adds complexity)
3. **Validation-only approach**: Fail fast with detailed errors (doesn't solve issue)
4. **Per-dataset handling**: Custom logic per dataset (not scalable)

### Risks and Trade-offs
- **Chosen approach**: Low risk, follows existing patterns, backward compatible
- **Main risk**: Default value of 2 might be suboptimal for some datasets
- **Mitigation**: Could add logging/warnings when defaults are applied

## Implementation Plan

### Changes Required

1. **Primary Change**: Update `get_num_classes()` method in `M_01_ISFM.py`
   - **Location**: Lines 66-70
   - **Approach**: Replace with robust filtering implementation following existing patterns

**Proposed Implementation:**
```python
def get_num_classes(self):
    num_classes = {}
    for key in np.unique(self.metadata.df['Dataset_id']):
        # Filter out NaN and -1 values (following existing pattern from Get_id.py)
        # -1 typically indicates samples that don't participate in classification training
        labels = self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']
        valid_labels = labels[labels.notna() & (labels >= 0)]
        
        if len(valid_labels) > 0:
            # Use valid labels to calculate class count
            num_classes[key] = int(valid_labels.max()) + 1
        else:
            # Default to binary classification if no valid labels exist
            # This handles edge cases where entire datasets have only -1/NaN labels
            num_classes[key] = 2
            
    return num_classes
```

### Testing Strategy
1. **Reproduction test**: Verify original bug can be reproduced
2. **Fix validation**: Confirm TypeError no longer occurs after fix
3. **Regression test**: Ensure existing valid datasets continue to work
4. **Edge case testing**: Test datasets with all -1, all NaN, mixed scenarios
5. **Integration test**: Run complete multi-task training pipeline

### Rollback Plan
- **Simple revert**: Original code is only 5 lines, easy to restore
- **No side effects**: Change is isolated to single method
- **Backward compatibility**: Fix preserves existing behavior for valid data