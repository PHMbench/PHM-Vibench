# OneEpochValidator Report
Generated: 2025-09-15 09:39:32
Device: cuda

## Overall Result: ❌ FAIL

## Validation Stages
### Data Loading: ✅ PASS
- **Processing Speed**: 1438.9 samples/sec
- **Samples Processed**: 80

### Forward Pass: ✅ PASS
- **Forward Time**: 0.1668s
- **Output Shape**: (16, 10)

### Loss Computation: ✅ PASS
- **Loss Value**: 2.3194079399108887
- **Loss Finite**: True

### Backward Pass: ✅ PASS
- **Backward Time**: 0.0358s
- **Gradients Computed**: True

### One Epoch Training: ❌ FAIL
- **Epoch Time**: 0.06s
- **Convergence Rate**: -0.96%
- **Steps Completed**: 20
- **Warnings**:
  - Loss did not decrease during 1-epoch training
  - Low convergence rate (-0.96%) during 1-epoch training

## Memory Usage Summary
- **Data Loading**: 0.00GB ✅ EFFICIENT
- **Forward Pass**: 0.03GB ✅ EFFICIENT
- **Loss Computation**: 0.02GB ✅ EFFICIENT
- **Backward Pass**: 0.01GB ✅ EFFICIENT
- **One Epoch Training**: 0.01GB ✅ EFFICIENT

## 95% Confidence Prediction
❌ **LOW CONFIDENCE** - Issues detected that may cause training failure

**Required actions before full training:**