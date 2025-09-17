# CWRU Few-Shot Learning with Contrastive Pretraining: Unexpected Results Analysis

**Experiment Date**: September 16, 2025
**Dataset**: CWRU Bearing Fault Diagnosis
**Framework**: PHM-Vibench Flow Integration

## 🚨 Executive Summary

**Unexpected Finding**: Contrastive learning pretraining significantly **decreased** performance across all three tasks compared to direct few-shot learning, contrary to expected improvements from self-supervised pretraining.

### Key Results Summary

| Method | Fault Diagnosis (Acc) | Anomaly Detection (Acc) | Signal Prediction (MSE) |
|--------|------------------------|-------------------------|-------------------------|
| **Case 1 (Direct)** | **81.67%** ✅ | **96.67%** ✅ | **2.91** ✅ |
| Case 2 (Contrastive) | 68.33% ❌ (-16.3%) | 93.33% ❌ (-3.4%) | 4.86 ❌ (+67.3%) |
| Case 3 (Flow+Contrastive) | 63.33% ❌ (-22.4%) | 96.67% ⚖️ (0%) | 4.60 ❌ (+58.3%) |

**Critical Observation**: Direct learning without any pretraining achieved the best performance across all metrics.

---

## 🔬 Experimental Setup

### Dataset Configuration
- **Source**: CWRU Bearing Dataset via PHM-Vibench metadata system
- **Signals Processed**: 155 CWRU entries from `metadata_6_11.xlsx`
- **Total Windows**: 134,454 (after NaN filtering)
- **Window Size**: 1024 samples (85.3 ms @ 12kHz)
- **Overlap**: 75% (stride = 256 samples)
- **Memory Usage**: 2,100.84 MB

### Few-Shot Learning Setup
- **Support Set**: 5 samples per class (5-shot learning)
- **Query Set**: 15 samples per class
- **Train/Test Split**: 70/30 by signal ID (prevents data leakage)
- **Training Epochs**: 30 for fine-tuning, 20 for pretraining

### Three Experimental Cases

#### Case 1: Direct Few-Shot Learning (Baseline)
- **Architecture**: Simple CNN backbone + task-specific heads
- **Training**: Direct optimization on support set
- **Parameters**: 35,488 backbone + 516-35,362 head parameters

#### Case 2: Contrastive Pretraining
- **Pretraining**: SimCLR-style contrastive learning with data augmentation
- **Architecture**: Enhanced encoder with projection head
- **Fine-tuning**: Frozen encoder, trainable classification heads only
- **Augmentation**: Gaussian noise (σ=0.1)

#### Case 3: Flow + Contrastive Pretraining
- **Pretraining**: Combined Flow matching + Contrastive learning
- **Architecture**: Flow model + Contrastive encoder + fusion heads
- **Training**: Joint pretraining followed by frozen backbone fine-tuning

---

## 📊 Detailed Performance Results

### 🔧 Fault Diagnosis (4-class classification)

| Method | Epoch 1 | Epoch 10 | Epoch 20 | Final (Epoch 30) |
|--------|---------|----------|----------|------------------|
| **Case 1 (Direct)** | 25.00% | 25.00% | 68.33% | **81.67%** |
| Case 2 (Contrastive) | 21.67% | 40.00% | 45.00% | 68.33% |
| Case 3 (Flow+Contrastive) | 36.67% | 63.33% | 65.00% | 63.33% |

**Analysis**: Direct learning showed steady improvement and achieved highest accuracy. Pretraining methods plateaued early and never reached baseline performance.

### 🚨 Anomaly Detection (binary classification)

| Method | Epoch 1 | Epoch 10 | Epoch 20 | Final (Epoch 30) |
|--------|---------|----------|----------|------------------|
| **Case 1 (Direct)** | 50.00% | 100.00% | 96.67% | **96.67%** |
| Case 2 (Contrastive) | 50.00% | 50.00% | 90.00% | 93.33% |
| Case 3 (Flow+Contrastive) | 63.33% | 93.33% | 96.67% | 96.67% |

**Analysis**: Case 1 achieved perfect classification quickly. Case 3 matched final performance but with slower convergence. Case 2 underperformed throughout.

### 📈 Signal Prediction (next-window forecasting)

| Method | Epoch 1 | Epoch 10 | Epoch 20 | Final (Epoch 30) |
|--------|---------|----------|----------|------------------|
| **Case 1 (Direct)** | 3.12 | 2.90 | 2.90 | **2.91** |
| Case 2 (Contrastive) | 3.51 | 3.95 | 4.46 | 4.86 |
| Case 3 (Flow+Contrastive) | 5.78 | 4.03 | 4.41 | 4.60 |

**Analysis**: Direct learning achieved lowest MSE and maintained stable performance. Both pretraining methods showed significantly higher prediction errors.

---

## 🔍 Analysis of Unexpected Results

### Why Did Contrastive Pretraining Fail?

#### 1. **Feature Representation Mismatch**
- Contrastive learning optimizes for sample similarity/dissimilarity
- May not align with fault diagnosis feature requirements
- Industrial signals might need different inductive biases than natural images

#### 2. **Frozen Encoder Limitation**
- Frozen pretrained encoder cannot adapt to few-shot task specifics
- Fine-tuning only classification heads may be insufficient
- Domain gap between pretraining and downstream tasks

#### 3. **Limited Pretraining Data & Epochs**
- Only 20 pretraining epochs may be insufficient for meaningful representations
- Self-supervised learning typically requires extensive pretraining
- CWRU dataset size may be too small for effective contrastive learning

#### 4. **Task-Pretraining Objective Mismatch**
- Contrastive learning emphasizes instance discrimination
- Fault diagnosis requires class-specific feature learning
- Signal prediction needs temporal modeling, not instance contrast

#### 5. **Augmentation Strategy Issues**
- Simple Gaussian noise augmentation may not preserve fault characteristics
- Could introduce artificial patterns that harm downstream performance
- Domain-specific augmentations might be necessary

### Performance Degradation Patterns

1. **Diagnosis Task**: Severe degradation (-16.3% to -22.4%)
   - Suggests contrastive features are counterproductive for fault classification
   - Multi-class problem most affected by representation mismatch

2. **Anomaly Detection**: Minimal impact (-3.4% to 0%)
   - Binary classification more robust to representation changes
   - Simpler decision boundary less sensitive to feature quality

3. **Signal Prediction**: Significant degradation (+58% to +67% MSE)
   - Temporal modeling requires specific inductive biases
   - Contrastive learning may disrupt time-series patterns

---

## 🔧 Technical Implementation Details

### Contrastive Loss Fix Applied
```python
def contrastive_loss(embeddings, temperature=0.5):
    embeddings = F.normalize(embeddings, dim=1)
    similarity = torch.mm(embeddings, embeddings.t()) / temperature
    batch_size = embeddings.shape[0] // 2

    # FIXED: Correct positive pair mapping
    labels = torch.cat([
        torch.arange(batch_size, batch_size * 2),  # Original -> Augmented
        torch.arange(batch_size)                    # Augmented -> Original
    ]).to(device)

    mask = torch.eye(similarity.shape[0]).bool().to(device)
    similarity = similarity.masked_fill(mask, -float('inf'))

    loss = F.cross_entropy(similarity, labels)
    return loss
```

### Data Processing Pipeline
1. **NaN Filtering**: Removed 2,353 windows with invalid labels
2. **Normalization**: StandardScaler per-channel z-score normalization
3. **ID-based Splitting**: Prevented data leakage by splitting on signal IDs
4. **Few-shot Episode Creation**: Balanced sampling across available classes

---

## 📈 Training Curves Analysis

### Convergence Patterns

**Case 1 (Direct Learning)**:
- Rapid initial improvement (epochs 1-10)
- Steady convergence to optimal performance
- Stable final performance across all tasks

**Case 2 (Contrastive Pretraining)**:
- Slower convergence in all tasks
- Early plateauing in diagnosis accuracy
- Consistent underperformance vs baseline

**Case 3 (Flow + Contrastive)**:
- Variable convergence patterns
- Good anomaly detection, poor diagnosis
- Intermediate prediction performance

---

## 🎯 Recommendations for Future Work

### Immediate Improvements

1. **Unfreeze Encoder During Fine-tuning**
   ```python
   # Instead of freezing encoder completely
   for param in encoder.parameters():
       param.requires_grad = True  # Allow adaptation
   ```

2. **Increase Pretraining Duration**
   - Scale from 20 to 100+ epochs
   - Monitor contrastive loss convergence
   - Use learning rate scheduling

3. **Domain-Specific Augmentations**
   - Time-frequency domain transformations
   - Amplitude scaling within physical limits
   - Phase shift augmentations

### Advanced Strategies

4. **Supervised Contrastive Learning**
   ```python
   # Use class labels for better representation learning
   labels = y_batch.unsqueeze(1) == y_batch.unsqueeze(0)
   supervised_contrastive_loss(embeddings, labels)
   ```

5. **Progressive Fine-tuning**
   - Gradual unfreezing of encoder layers
   - Layer-wise learning rate adaptation
   - Task-specific head pretraining

6. **Alternative Self-Supervised Objectives**
   - Masked signal modeling (like BERT for time series)
   - Temporal order prediction
   - Signal reconstruction tasks

### Experimental Validation

7. **Ablation Studies**
   - Test different augmentation strategies
   - Vary pretraining/fine-tuning epoch ratios
   - Compare frozen vs unfrozen encoder performance

8. **Cross-Dataset Evaluation**
   - Validate on XJTU, FEMTO datasets
   - Test domain transfer capabilities
   - Measure pretraining robustness

---

## 📋 Appendix: Technical Specifications

### Hardware Configuration
- **Device**: CUDA-enabled GPU
- **Framework**: PyTorch 2.7.1+cu126
- **Memory**: 2.1GB for full dataset

### Hyperparameters
```yaml
batch_size: 32
learning_rate: 0.001
pretrain_epochs: 20
finetune_epochs: 30
temperature: 0.5  # contrastive loss
augmentation_noise: 0.1  # gaussian std
```

### Model Architectures
```python
# Case 1: Direct Backbone
Conv1d(2→32, k=7) → Conv1d(32→64, k=5) → Conv1d(64→128, k=3)
Parameters: 35,488 backbone + task heads

# Case 2: Contrastive Encoder
Conv1d + MaxPool layers → Projection(128→128)
Parameters: 68,512 encoder + task heads

# Case 3: Flow + Contrastive
Flow: Linear(2048→256→2048) + Contrastive Encoder
Combined feature fusion: 256-dim representations
```

---

## 🎓 Conclusion

This experiment revealed a **counterintuitive result**: contrastive pretraining significantly degraded few-shot learning performance on industrial vibration signals. This challenges the assumption that self-supervised pretraining universally improves downstream task performance.

**Key Takeaways**:
1. Domain-specific pretraining strategies are crucial for industrial signals
2. Simple direct learning can outperform complex pretraining approaches
3. Feature representation alignment with downstream tasks is critical
4. Frozen encoder fine-tuning may be too restrictive for adaptation

**Future Research Directions**:
- Develop industrial signal-specific self-supervised objectives
- Investigate optimal pretraining-finetuning trade-offs
- Design domain-aware augmentation strategies for vibration data

This finding emphasizes the importance of empirical validation and domain-specific adaptation in machine learning for industrial applications.

---

**Generated**: September 16, 2025
**Framework**: PHM-Vibench Flow Integration
**Notebook**: `cwru_multitask_fewshot_study.ipynb`
![result](e1.png)
