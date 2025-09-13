# Bug Report

## Bug Summary
Multi-task PHM training has two critical issues: (1) NaN values appearing in total loss and RUL prediction loss, and (2) missing task-specific metrics logging for individual tasks (classification accuracy, anomaly detection AUC, signal prediction MSE, RUL prediction MAE).

## Bug Details

### Expected Behavior
- All loss values should be valid floating-point numbers
- Total loss should be the weighted sum of individual task losses
- Signal prediction should handle different input/output dimensions gracefully
- **Each task should log its specific metrics**:
  - Classification: accuracy, f1_score, precision, recall
  - Anomaly Detection: AUC-ROC, precision, recall, f1
  - Signal Prediction: MSE, MAE, R2
  - RUL Prediction: MAE, MSE, R2, relative error
- Training should proceed without NaN values or tensor mismatches

### Actual Behavior  
- `test_loss`, `val_loss`, and `train_loss_epoch` show NaN values
- `test_rul_prediction_loss` and `val_rul_prediction_loss` show NaN
- `train_rul_prediction_loss_epoch` shows NaN
- Signal prediction fails with: "The size of tensor a (512) must match the size of tensor b (4096)"
- **Only loss values are logged, no task-specific metrics are recorded**
- No accuracy, F1, AUC, or other performance metrics appear in logs

### Steps to Reproduce
1. Run multi-task training with optimized YAML configurations
2. Execute: `python main_LQ.py --config_path script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`
3. Observe training logs and WandB metrics
4. Note NaN values in loss metrics and signal prediction failures
5. Check WandB dashboard - only loss values visible, no performance metrics

### Environment
- **Version**: PHM-Vibench latest (commit 47d9027)
- **Platform**: Linux, NVIDIA RTX 3090 (24GB)
- **Configuration**: 
  - PyTorch 2.7.1+cu126
  - Python 3.10.18
  - Reduced model dimensions (hidden_dim=64, max_len=512→4096 issue)

## Impact Assessment

### Severity
- [X] High - Major functionality broken
- Signal prediction task completely non-functional
- RUL prediction produces invalid results
- Total loss calculation corrupted
- **No visibility into model performance beyond loss values**

### Affected Users
- All users attempting multi-task foundation model training
- Researchers using the PHM-Vibench benchmark for multi-task learning
- Users needing to evaluate model performance per task

### Affected Features
- Multi-task PHM training pipeline
- Signal prediction task
- RUL (Remaining Useful Life) prediction task
- Loss aggregation and reporting
- **Performance metrics logging and evaluation**

## Additional Context

### Error Messages
```
WARNING: signal_prediction train failed: The size of tensor a (512) must match the size of tensor b (4096) at non-singleton dimension 1

wandb:                          test_loss nan
wandb:           test_rul_prediction_loss nan
wandb:                   train_loss_epoch nan
wandb:    train_rul_prediction_loss_epoch nan
wandb:                           val_loss nan
wandb:            val_rul_prediction_loss nan
```

### Current WandB Output (Missing Metrics)
```
wandb:        test_anomaly_detection_loss 0.33327  # ✓ Loss only
wandb:           test_classification_loss 0.39367  # ✓ Loss only
# Missing: test_classification_acc, test_classification_f1
# Missing: test_anomaly_detection_auc, test_anomaly_detection_precision
# Missing: test_signal_prediction_mse, test_signal_prediction_mae
# Missing: test_rul_prediction_mae, test_rul_prediction_r2
```

### Screenshots/Media
Training completed but with degraded metrics:
- test_anomaly_detection_loss: 0.33327 ✓
- test_classification_loss: 0.39367 ✓
- test_loss: nan ✗
- test_rul_prediction_loss: nan ✗
- **No task-specific metrics available** ✗

### Related Issues
- Memory optimization changes reduced max_len from 4096 to 512
- YAML files show inconsistency: comment says "Reduced from 4096 to 512" but value is still 4096
- H_03_Linear_pred task head parameter mismatch
- multi_task_phm.py lacks metric computation implementation

## Initial Analysis

### Suspected Root Cause
1. **Signal Prediction**: Mismatch between model output size (512 timesteps from reduced max_len) and target size (4096 timesteps from input data)
2. **RUL Prediction**: Possible missing or invalid RUL labels in metadata causing NaN propagation
3. **Total Loss**: NaN from any task propagates to total loss calculation
4. **Missing Metrics**: The `multi_task_phm.py` implementation only computes losses in `_compute_task_loss()` but never calculates or logs task-specific metrics

### Affected Components
- `src/task_factory/task/In_distribution/multi_task_phm.py` - Loss computation logic, missing metric computation
- `src/model_factory/ISFM/task_head/H_03_Linear_pred.py` - Signal prediction head
- `script/Vibench_paper/foundation_model/multitask_*.yaml` - Configuration files with inconsistent max_len values
- Loss aggregation in `_shared_step` method (lines 192-214)
- **Missing**: Metric computation methods for each task
- **Missing**: Metric logging calls in training/validation/test steps