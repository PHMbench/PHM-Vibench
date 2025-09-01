# Bug Analysis

## Root Cause Analysis

### Investigation Summary
Conducted detailed analysis of multi-task model initialization failure. Investigation focused on the TypeError occurring during MultiTaskHead creation, traced back through the call stack to identify the source of None values in class count calculations.

### Root Cause
The `get_num_classes()` method in `M_01_ISFM.py` fails to properly handle datasets with NaN values and -1 labels. When `max()` is applied to pandas Series containing NaN values, it returns NaN, which propagates through the calculation and results in None being passed to PyTorch's `nn.Linear` layer.

### Contributing Factors
1. **Inconsistent label formats**: Industrial datasets use different conventions (-1 for anomalies, NaN for unlabeled)
2. **Lack of data validation**: No preprocessing to handle invalid label values
3. **Brittle calculation**: Direct max() without filtering invalid values

## Technical Details

### Affected Code Locations

- **File**: `src/model_factory/ISFM/M_01_ISFM.py`
  - **Function/Method**: `get_num_classes()`
  - **Lines**: `66-70`
  - **Issue**: max() returns NaN when Label column contains NaN values

- **File**: `src/model_factory/ISFM/task_head/multi_task_head.py`
  - **Function/Method**: `__init__()`
  - **Lines**: `131`
  - **Issue**: nn.Linear() receives None instead of integer for output dimension

### Data Flow Analysis
1. Metadata loaded with Label column containing mixed data (valid labels, NaN, -1)
2. `get_num_classes()` called during model initialization
3. For each Dataset_id, `max()` applied to Label values
4. NaN values cause max() to return NaN
5. NaN + 1 = NaN, stored in num_classes dictionary
6. MultiTaskHead tries to create nn.Linear with NaN dimension
7. PyTorch raises TypeError for invalid arguments

### Dependencies
- **pandas**: Series.max() behavior with NaN values
- **numpy**: Data type handling and NaN propagation
- **PyTorch**: nn.Linear dimension validation

## Impact Analysis

### Direct Impact
- Complete failure of multi-task model initialization
- Unable to run any multi-task experiments
- Blocks foundation model training pipeline

### Indirect Impact  
- Research workflow disrupted
- Potential similar issues in other components using label statistics
- Reduced confidence in data preprocessing robustness

### Risk Assessment
- **High risk** if not fixed: Core functionality completely broken
- **Medium risk** of similar issues: Other components may have same vulnerability
- **Low risk** of data corruption: Issue is in computation, not storage

## Solution Approach

### Fix Strategy
Implement robust label handling in `get_num_classes()` method:
1. Filter out NaN and invalid values (-1) before calculation
2. Use only valid labels (>= 0) for class count
3. Provide sensible default (2 for binary classification) when no valid labels exist

### Alternative Solutions
1. **Preprocessing approach**: Clean metadata during loading
2. **Validation approach**: Add explicit validation with detailed error messages
3. **Configuration approach**: Allow manual specification of class counts

### Risks and Trade-offs
- **Chosen solution**: Minimal risk, preserves existing behavior for valid data
- **Risk**: Default value of 2 might be incorrect for some datasets
- **Mitigation**: Log warnings when defaults are used

## Implementation Plan

### Changes Required

1. **Change 1**: Modify get_num_classes() method
   - File: `src/model_factory/ISFM/M_01_ISFM.py`
   - Modification: Replace lines 66-70 with robust implementation

```python
def get_num_classes(self):
    num_classes = {}
    for key in np.unique(self.metadata.df['Dataset_id']):
        # Filter out NaN and -1 values (-1 typically indicates unlabeled/anomalous samples)
        labels = self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']
        valid_labels = labels[labels.notna() & (labels >= 0)]
        
        if len(valid_labels) > 0:
            # Use valid labels to calculate class count
            num_classes[key] = int(valid_labels.max()) + 1
        else:
            # Default to binary classification if no valid labels
            num_classes[key] = 2
            
    return num_classes
```

### Testing Strategy
1. **Unit test**: Verify method handles NaN, -1, and valid labels correctly
2. **Integration test**: Confirm model initialization succeeds
3. **Regression test**: Ensure existing valid datasets still work
4. **End-to-end test**: Run complete multi-task experiment

### Rollback Plan
- Simple git revert if issues arise
- Original implementation is 5 lines, easy to restore
- No database or configuration changes required