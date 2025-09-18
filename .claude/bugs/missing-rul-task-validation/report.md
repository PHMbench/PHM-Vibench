# Bug Report: Missing RUL Task Validation

## Summary
**Bug ID**: missing-rul-task-validation  
**Severity**: Medium  
**Type**: Logic Error / Data Validation  
**Component**: Multi-task PHM Training (`multi_task_phm.py`)  
**Reporter**: User  
**Date**: 2025-01-09  

## Description
The multi-task training system attempts to train all enabled tasks (classification, anomaly detection, signal prediction, RUL prediction) for all datasets, but not all datasets support all tasks. The metadata contains fields `Fault_Diagnosis`, `Anomaly_Detection`, and `Remaining_Life` that indicate which tasks are supported for each dataset, but these are not being checked before enabling tasks.

## Current Behavior
1. System tries to run RUL prediction task for datasets that don't support it (e.g., Dataset 1, 5 have `Remaining_Life=0`)
2. This causes:
   - NaN values in RUL labels
   - Training on invalid/default values  
   - Degraded model performance
   - Unnecessary warnings in logs

## Expected Behavior
1. System should check metadata fields before enabling tasks:
   - `Fault_Diagnosis=1/TRUE` → Enable classification task
   - `Anomaly_Detection=1/TRUE` → Enable anomaly detection task  
   - `Remaining_Life=1/TRUE` → Enable RUL prediction task
2. Tasks should be dynamically enabled/disabled per dataset
3. No attempts to train unsupported tasks

## Evidence
From metadata analysis:
```
Dataset 1: FD=1, AD=1, RL=0  (CWRU - no RUL support)
Dataset 2: FD=TRUE, AD=TRUE, RL=TRUE  (has RUL support)
Dataset 5: FD=1, AD=1, RL=0  (no RUL support)
```

## Steps to Reproduce
1. Run multi-task training with CWRU dataset (Dataset_id=1)  
2. Enable RUL prediction in config: `enabled_tasks: ["classification", "rul_prediction"]`
3. Observe warnings about missing RUL labels and NaN values

## Root Cause
In `multi_task_phm.py`:
- `_get_enabled_tasks()` uses static configuration without checking metadata capabilities
- `_build_task_labels_batch()` tries to extract RUL labels even when dataset doesn't support them
- No validation against metadata's task capability fields

## Impact
- **Performance**: Training on invalid data degrades model quality
- **Robustness**: NaN-related errors in RUL tasks
- **Usability**: Excessive warning messages

## Status
- [x] Reported
- [x] Analyzed  
- [x] Fixed
- [x] Verified

## Fix Summary
**Fix Date**: 2025-01-09  
**Fix Location**: `src/task_factory/task/In_distribution/multi_task_phm.py`

### Implemented Solution
1. **Added task capability validation**: `_get_dataset_supported_tasks()` method checks metadata fields
2. **Dynamic task filtering**: `_get_enabled_tasks()` now validates tasks against dataset capabilities  
3. **Configuration options**: Added `enable_task_validation`, `validation_mode`, `force_enable_tasks`
4. **Comprehensive testing**: Created test suite with 8 scenarios covering all validation modes

### Configuration Options
```yaml
task:
  enable_task_validation: true      # Enable/disable validation
  validation_mode: "warn"           # "warn", "error", "ignore"  
  force_enable_tasks: []            # Skip validation for specific tasks
```

### Validation Results
- ✅ Correctly identifies dataset task capabilities
- ✅ Automatically disables unsupported tasks (e.g., RUL for CWRU)
- ✅ Prevents training on invalid/NaN RUL data
- ✅ Provides clear warnings and error messages
- ✅ All 8 test scenarios pass

### Impact
- **Eliminated invalid RUL training** on datasets without RUL support
- **Improved model performance** by preventing training on NaN/default values
- **Enhanced user experience** with clear validation messages
- **Flexible configuration** supports different validation strategies