# Bug Analysis

## Root Cause Analysis

### Investigation Summary
Investigated the complete flow from YAML configuration to metric computation. Found that `evaluation.compute_metrics` configuration is never passed to or used by the multi-task training system.

### Root Cause
**Missing configuration integration**: The task factory system does not pass `args_evaluation` parameters to task classes, and the multi-task implementation uses hardcoded metric mappings instead of reading from configuration.

### Contributing Factors
1. **Architecture gap**: Task factory only passes 7 parameters, missing evaluation config
2. **Legacy design**: Multi-task module was designed before configurable metrics
3. **No validation**: No checks or warnings when config is ignored

## Technical Details

### Affected Code Locations

- **File**: `src/task_factory/task_factory.py`
  - **Function/Method**: `task_factory()`
  - **Lines**: `57-65`
  - **Issue**: Missing `args_evaluation` parameter in task instantiation

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Function/Method**: `_initialize_task_metrics()`
  - **Lines**: `241-249`
  - **Issue**: Hardcoded metric mappings, no config reading

- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **Function/Method**: `__init__()`
  - **Lines**: `36-90`
  - **Issue**: Constructor doesn't accept or store evaluation arguments

### Data Flow Analysis
1. **Config Loading**: YAML correctly parsed by config system ✅
2. **Task Factory**: args_evaluation never passed to task ❌
3. **Task Init**: Multi-task class never receives evaluation config ❌
4. **Metrics Setup**: Hardcoded mappings used instead ❌
5. **Validation**: All hardcoded metrics computed ❌

### Dependencies
- PyTorch Lightning metric computation system
- TorchMetrics library for metric implementations
- YAML configuration parsing (works correctly)

## Impact Analysis

### Direct Impact
- **Performance**: 4x slower validation than configured (extra metric computations)
- **Resource usage**: Unnecessary GPU/CPU cycles for unwanted metrics
- **User experience**: Configuration appears broken/ignored

### Indirect Impact
- **Training time**: Significantly longer for large experiments
- **Development workflow**: Developers can't optimize validation for fast iteration
- **Research efficiency**: Longer experiment cycles

### Risk Assessment
**Medium risk**: Current validation optimizations are only 25% effective (40x vs potential 160x speedup)

## Solution Approach

### Fix Strategy
1. **Add evaluation arguments to task factory**
2. **Modify multi-task init to accept evaluation config**  
3. **Replace hardcoded metrics with config-driven initialization**
4. **Maintain backward compatibility with default metrics**

### Alternative Solutions
1. **Global config approach**: Read evaluation config from global state
2. **Monkey patching**: Override metric mapping at runtime
3. **New task class**: Create configurable version of multi-task

### Risks and Trade-offs
**Chosen approach (1) risks**:
- Breaking changes to task factory interface
- Need to update all task classes
- Backward compatibility challenges

**Benefits**: Clean, maintainable, follows existing patterns

## Implementation Plan

### Changes Required

1. **Task Factory Interface**: Add args_evaluation parameter
   - File: `src/task_factory/task_factory.py`
   - Modification: Add evaluation args to all task instantiations

2. **Multi-task Constructor**: Accept evaluation arguments
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Update `__init__` signature and store evaluation config

3. **Dynamic Metrics**: Use config to build metrics
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Replace hardcoded mapping with config-driven logic

4. **Backward Compatibility**: Default metrics when config missing
   - File: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - Modification: Fallback to current hardcoded behavior if no config

### Testing Strategy
1. **Unit tests**: Verify metric initialization with various configs
2. **Integration tests**: End-to-end config → metrics flow
3. **Performance tests**: Validate expected speedup achieved
4. **Regression tests**: Ensure existing behavior preserved when no config

### Rollback Plan
- Keep original hardcoded mapping as fallback
- Feature flag to enable/disable config-driven metrics
- Easy revert by removing args_evaluation parameter