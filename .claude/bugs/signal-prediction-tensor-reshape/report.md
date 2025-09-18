# Bug Report

## Bug Summary
Signal prediction metrics fail to compute due to tensor view incompatibility error in multi-task PHM module

## Bug Details

### Expected Behavior
Signal prediction metrics (MSE, MAE, R2) should compute successfully for all tensor shapes and configurations during multi-task training and testing

### Actual Behavior  
Warning messages appear during testing indicating tensor view size incompatibility:
- `Warning: Failed to compute test_mse for signal_prediction: view size is not compatible with input tensor's size and stride`
- `Warning: Failed to compute test_mae for signal_prediction: view size is not compatible with input tensor's size and stride`
- Only R2 metric computes successfully, MSE and MAE metrics fail

### Steps to Reproduce
1. Run multi-task PHM training with signal_prediction task enabled
2. Execute testing phase with multi-task model
3. Observe metric computation failures during test evaluation
4. Check logs for tensor view incompatibility warnings

### Environment
- **Version**: PHM-Vibench multi-task module (current)
- **Platform**: PyTorch Lightning training environment
- **Configuration**: Multi-task enabled with signal_prediction task
- **Models**: ISFM models (B_08_PatchTST, B_06_TimesNet, etc.)

## Impact Assessment

### Severity
- [X] Medium - Feature impaired but workaround exists
- [ ] Critical - System unusable
- [ ] High - Major functionality broken
- [ ] Low - Minor issue or cosmetic

### Affected Users
Data scientists and researchers using multi-task PHM models for signal prediction and reconstruction tasks

### Affected Features
- Signal prediction metric computation (MSE, MAE specifically)
- Multi-task training evaluation and monitoring
- Model performance assessment for signal reconstruction tasks

## Additional Context

### Error Messages
```
Warning: Failed to compute test_mse for signal_prediction: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
Warning: Failed to compute test_mae for signal_prediction: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

### Screenshots/Media
Test results showing successful training but failed metric computation:
```
   signal_prediction_test_r2          -0.4496903419494629
  test_signal_prediction_loss          1.4516125917434692
```
(Note: MSE and MAE metrics missing from results due to computation failure)

### Related Issues
- This issue emerged after recent fixes to device mismatch problems
- Related to tensor reshaping operations in multi-dimensional signal data
- Part of broader multi-task PHM module robustness improvements

## Initial Analysis

### Suspected Root Cause
Using `view()` method on non-contiguous tensors in the `_compute_regression_metrics` function. The error message explicitly suggests using `reshape()` instead of `view()`.

### Affected Components
- **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
- **Function**: `_compute_regression_metrics()`
- **Lines**: 273-274 (tensor reshaping operations)
- **Specific code**:
  ```python
  preds = preds.view(-1, preds.size(-1))
  targets = targets.view(-1, targets.size(-1))
  ```

### Technical Context
- Occurs when flattening 3D tensors for signal prediction metric computation
- `view()` requires tensors to be contiguous in memory
- `reshape()` handles both contiguous and non-contiguous tensors
- Issue manifests during test phase metric evaluation

---
*Bug created: 2025-09-08*
*Reporter: Multi-task PHM Testing*
*Priority: Medium*
*Component: Task Factory - Multi-task PHM Module*