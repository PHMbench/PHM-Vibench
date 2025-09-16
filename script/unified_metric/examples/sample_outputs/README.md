# 📊 Example Outputs Reference

This directory contains sample outputs from the unified metric learning pipeline to help researchers understand expected results.

## 📋 Contents

- `validation_report_sample.md` - Example validation test results
- `pipeline_summary_sample.json` - Sample experiment completion summary
- `performance_comparison_sample.tex` - Example LaTeX table
- `analysis_report_sample.md` - Sample statistical analysis
- `zero_shot_results_sample.csv` - Example zero-shot evaluation data

## 🎯 Expected Performance Ranges

### Zero-Shot Performance (Stage 1 Results)
- **CWRU**: 78-85% accuracy
- **XJTU**: 76-82% accuracy
- **THU**: 80-87% accuracy
- **Ottawa**: 79-84% accuracy
- **JNU**: 75-81% accuracy
- **Average**: 78-84% across all datasets

### Fine-Tuned Performance (Stage 2 Results)
- **CWRU**: 94-98% accuracy
- **XJTU**: 93-97% accuracy
- **THU**: 95-98% accuracy
- **Ottawa**: 94-97% accuracy
- **JNU**: 92-96% accuracy
- **Average**: 94-97% across all datasets

### Statistical Significance
- **p-values**: Typically < 0.001 vs single-dataset baselines
- **Effect sizes**: Cohen's d > 0.8 (large effect)
- **Confidence intervals**: 95% CI typically ±2-3% around mean

## ⏱️ Timeline Expectations

### Pretraining Stage (12 hours)
```
Epoch 1/50:  Loss: 3.24 → 2.89  Acc: 0.23 → 0.28
Epoch 10/50: Loss: 2.89 → 1.87  Acc: 0.28 → 0.45
Epoch 25/50: Loss: 1.87 → 1.23  Acc: 0.45 → 0.67
Epoch 50/50: Loss: 1.23 → 0.89  Acc: 0.67 → 0.82
```

### Fine-Tuning Stage (2 hours per dataset)
```
CWRU:    Epochs 1-20  →  0.82 → 0.96  (+14% improvement)
XJTU:    Epochs 1-20  →  0.79 → 0.95  (+16% improvement)
THU:     Epochs 1-20  →  0.85 → 0.97  (+12% improvement)
Ottawa:  Epochs 1-20  →  0.81 → 0.96  (+15% improvement)
JNU:     Epochs 1-20  →  0.78 → 0.94  (+16% improvement)
```

## 📈 Quality Indicators

### ✅ High-Quality Results
- Zero-shot accuracy >80% average
- Fine-tuned accuracy >95% average
- Training converges smoothly
- Statistical significance p < 0.01

### ⚠️ Warning Signs
- Zero-shot accuracy <75% average
- High variance across seeds (>5% std)
- Training instability or divergence
- No statistical significance p > 0.05

### ❌ Poor Results
- Zero-shot accuracy <70% average
- Fine-tuned accuracy <90% average
- Frequent training failures
- Random performance levels

## 🔧 Using These Examples

1. **Compare your validation**: Check if `quick_validate.py` results match `validation_report_sample.md`
2. **Verify progress**: Monitor training logs against expected timeline
3. **Quality check**: Ensure final results fall within expected ranges
4. **Troubleshoot**: Use warning signs to identify issues early

## 📞 When to Seek Help

Contact the development team if:
- Results consistently fall below warning thresholds
- Multiple experiments fail without clear error messages
- Training time exceeds expected ranges by >50%
- Statistical analysis shows no significant improvements