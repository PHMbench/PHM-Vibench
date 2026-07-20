# Experimental Reproducibility Checklist

## 1. Datasets

### 1.1 CWRU Bearing Dataset
- **Source**: Case Western Reserve University Bearing Data Center
- **Download URL**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Task**: Rolling element bearing fault diagnosis
- **Fault Types**: Normal (NOR), Inner race (IR), Ball (B), Outer race (OR)
- **Fault Sizes**: 0.007", 0.014", 0.021", 0.028"
- **Load Conditions**: 0, 1, 2, 3 hp
- **Sampling Frequency**: 48 kHz
- **Data Split**:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- **Preprocessing**:
  - Segment length: 4096 points
  - Overlap: 50%
  - Normalization: Z-score per segment

### 1.2 XJTU Bearing Dataset
- **Source**: Xi'an Jiaotong University
- **Download URL**: Available upon request
- **Task**: Bearing remaining useful life prediction and fault diagnosis
- **Operating Conditions**:
  - Speed: 2100 rpm
  - Load: 11 kN
  - Sampling: 25.6 kHz
- **Fault Types**:
  - Inner race, outer race, cage, ball
  - Multiple severity levels
- **Data Split**:
  - Train: 80%
  - Validation: 10%
  - Test: 10%

### 1.3 THU_018 Dataset
- **Status**: Out of scope for the paper-local 2026-03-19 truth-first autoresearch cycle.
- **Reason**: No parent-accepted THU_018 artifact exists in the current evidence pack, so the dataset is not claimed in manuscript-facing results.

## 2. Experimental Configuration

### 2.1 Hardware Setup
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: Intel Xeon Gold 6248R
- **RAM**: 128 GB
- **OS**: Ubuntu 20.04 LTS
- **CUDA**: 11.8
- **Python**: 3.9.16

### 2.2 Software Dependencies
```yaml
# requirements.txt
torch==2.1.2+cu121
torchvision==0.16.2+cu121
pytorch-lightning==2.1.3
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
pandas==1.5.3
ptwt==0.1.7
wandb==0.16.0
tqdm==4.64.1
```

### 2.3 Training Hyperparameters
```yaml
# Base configuration
model:
  name: Fusion1D2D
  fusion_type: progressive

training:
  max_epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: Adam
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 100
    eta_min: 0.0001

data:
  window_size: 4096
  overlap: 0.5
  normalization: zscore

regularization:
  dropout: 0.2
  l1_norm: 0.0001

early_stopping:
  monitor: val_loss
  patience: 15
  min_delta: 0.001
```

### 2.4 Signal Processing Configuration
```yaml
# Four-layer signal processing stack
signal_processing:
  layer1: ['FFT', 'WF', 'I']  # Feature extraction
  layer2: ['HT', 'WF', 'I']   # Non-linear transformation
  layer3: ['FFT', 'I', 'I']    # Frequency enhancement
  layer4: ['I', 'WF', 'I']     # Final processing

# Statistical features
statistical_features:
  - Mean
  - Standard Deviation
  - Variance
  - Entropy
  - Maximum
  - Minimum
  - Absolute Mean
  - Kurtosis
  - RMS
  - Crest Factor
  - Skewness
  - Clearance Factor
  - Shape Factor
```

### 2.5 Random Seeds
For reproducibility, experiments are conducted with three random seeds:
- Seed 20
- Seed 42
- Seed 2024

## 3. Evaluation Metrics

### 3.1 Classification Metrics
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{Total}}$
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Confusion Matrix**: For detailed error analysis

### 3.2 Explainability Metrics

#### 3.2.1 Faithfulness (Deletion Test)
- **Metric**: Del@k - Average accuracy when top-k important features are removed
- **Implementation**:
  1. Extract feature importance from model
  2. Sort features by importance score
  3. Iteratively remove top-k features
  4. Evaluate model performance
- **Expected**: Faithful models show significant performance drop when important features are removed

#### 3.2.2 Stability
- **Metric**: Stab@σ - Average cosine similarity between explanations under Gaussian noise
- **Implementation**:
  ```python
  def stability_test(model, x, noise_std=0.01):
      # Original explanation
      exp_orig = explain(model, x)

      # Add Gaussian noise
      x_noisy = x + torch.randn_like(x) * noise_std
      exp_noisy = explain(model, x_noisy)

      # Calculate similarity
      similarity = F.cosine_similarity(exp_orig, exp_noisy)
      return similarity.mean()
  ```

#### 3.2.3 Efficiency
- **Metric**: Time per sample (ms) for explanation generation
- **Measurement**: Average over 100 random samples
- **Hardware**: Same as training setup

## 4. Ablation Study Setup

### 4.1 Fusion Strategies
- **Early Fusion**: Concatenate raw 1D and 2D features
- **Mid Fusion**: Feature-level interaction
- **Late Fusion**: Decision-level combination
- **Progressive Fusion**: Our proposed multi-stage approach

### 4.2 Alignment Components
- **Physical Alignment Only**
- **Semantic Alignment Only**
- **Geometric Alignment Only**
- **Full Tri-Level Alignment** (proposed)

### 4.3 Feature Ablations
- **1D Only**: Time-series branch only
- **2D Only**: Spectrogram branch only
- **Statistical Features Only**
- **Full Model**: All features combined

## 5. Baseline Methods

### 5.1 Single-Modal Methods
- **TSPN**: Transparent Signal Processing Network
- **ResNet-1D**: 1D CNN for raw signals
- **LSTM**: Recurrent neural network for sequences
- **Transformer**: Self-attention based model

### 5.2 2D Methods
- **ResNet-2D**: CNN on spectrograms
- **VGG-16**: Pretrained on ImageNet
- **EfficientNet**: Efficient convolutional architecture

### 5.3 Fusion Methods
- **Concat Fusion**: Simple feature concatenation
- **Attention Fusion**: Attention-based feature weighting
- **MoE**: Mixture of Experts fusion

## 6. Implementation Details

### 6.1 Model Architecture
```python
class ProgressiveFusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D branch
        self.conv1d_1 = nn.Conv1d(2, 32, kernel_size=7, stride=2)
        self.conv1d_2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)

        # 2D branch
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=7, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        # Fusion modules
        self.early_fusion = EarlyFusionModule()
        self.mid_fusion = MidFusionModule()
        self.late_fusion = LateFusionModule()

        # Alignment
        self.physical_alignment = PhysicalAlignment()
        self.semantic_alignment = SemanticAlignment()
        self.geometric_alignment = GeometricAlignment()

        # Classifier
        self.classifier = nn.Linear(256, num_classes)
```

### 6.2 Training Procedure
```python
# Training loop
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        # Data: (x_1d, x_2d, stats), labels

        # Forward pass
        outputs = model(x_1d, x_2d, stats)
        loss = criterion(outputs, labels)

        # Add alignment losses
        loss += alignment_weight * alignment_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = validate(model, val_loader)

    # Early stopping check
    if early_stopping.should_stop(val_loss):
        break
```

## 7. Statistical Analysis

### 7.1 Multiple Runs
- **Number of runs**: 3 (different random seeds)
- **Reporting format**: Mean ± Standard Deviation
- **Confidence intervals**: 95% CI using t-distribution

### 7.2 Significance Testing
- **Test**: Paired t-test between proposed method and best baseline
- **Significance level**: α = 0.05
- **Effect size**: Cohen's d

## 8. Failure Case Collection

### 8.1 Criteria
- Misclassified samples with high confidence (>0.9)
- Samples with high feature importance but wrong prediction
- Boundary cases between fault types

### 8.2 Documentation Format
```markdown
### Failure Case #ID
- **Sample ID**: [dataset_sample_id]
- **True Label**: [fault_type]
- **Predicted**: [predicted_type]
- **Confidence**: [score]
- **Feature Analysis**: [important features]
- **Possible Reason**: [hypothesis]
- **Visualization**: [link to figure]
```

## 9. Code and Data Availability

### 9.1 Code Repository
- **Platform**: GitHub
- **URL**: https://github.com/[repository]/1D-2D-fusion-explainable
- **License**: MIT
- **Dependencies**: Provided in `requirements.txt`

### 9.2 Data Access
- **CWRU**: Publicly available
- **XJTU**: Available upon request
- **THU_018**: Not claimed in the accepted 2026-03-19 truth-first cycle

### 9.3 Model Checkpoints
- **Location**: `models/checkpoints/`
- **Format**: PyTorch .pth
- **Metadata**: Training configuration, performance metrics

## 10. Expected Computational Resources

### 10.1 Training Time
- **Single run**: ~2 hours (single GPU)
- **3 seeds**: ~6 hours (3 GPUs in parallel)
- **Full ablation**: ~24 hours

### 10.2 Memory Requirements
- **Training**: ~8GB GPU memory
- **Inference**: ~2GB GPU memory
- **Explanation generation**: ~4GB GPU memory

### 10.3 Storage
- **Raw data**: ~10GB
- **Processed features**: ~5GB
- **Models and logs**: ~2GB