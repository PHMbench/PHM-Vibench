# Experiments: Reproducibility and Validation Protocol

## 1. Dataset Preparation

### 1.1 THU_018 Bearing Fault Dataset
- **Source**: Tsinghua University Bearing Dataset
- **Data Path**: `/home/user/data/PHMbenchdata/PHM-Vibench/THU_018/`
- **Fault Types**:
  - Healthy (H)
  - Inner Race Fault (IF)
  - Outer Race Fault (OF)
  - Ball Fault (BF)
  - Cage Fault (CF)
- **Sampling Rate**: 25.6 kHz
- **Signal Length**: 4096 points per sample
- **Total Samples**: 10,000 (train: 7,000, val: 1,000, test: 2,000)

### 1.2 Data Preprocessing
```python
# Preprocessing pipeline
def preprocess_signal(signal):
    # 1. Normalize to zero mean, unit variance
    signal = (signal - np.mean(signal)) / np.std(signal)

    # 2. Denoise using wavelet thresholding
    denoised = wavelet_denoise(signal, wavelet='db4', threshold=0.1)

    # 3. Segment to fixed length
    segments = segment_signal(denoised, length=4096, overlap=0.5)

    return segments
```

### 1.3 Data Splits
- **Train/Val/Test Ratio**: 70% / 10% / 20%
- **Stratified Sampling**: Maintain class distribution
- **Random Seeds**: [20, 42, 123, 456, 789] for reproducibility

## 2. Training Configuration

### 2.1 Model Architecture
```yaml
model:
  name: FuzzyLogicV2
  parameters: 7,600
  fuzzy_rules: 50
  membership_functions: 3  # Low, Medium, High per feature

signal_processing:
  layers: ['I', 'WF', 'I', 'WF']
  output_dim: 4096

feature_extraction:
  statistical_features: 13
  deep_features: 128

fusion:
  method: adaptive
  fusion_weight: learnable
```

### 2.2 Hyperparameters
```python
training_config = {
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_epochs': 100,
    'patience': 15,
    'weight_decay': 0.0001,
    'gradient_clip': 1.0,
    'scheduler': 'cosine',
    'min_lr': 0.0001
}
```

### 2.3 Loss Function Weights
- $\lambda_1$ (classification): 1.0
- $\lambda_2$ (regularization): 0.0001
- $\lambda_3$ (explainability): 0.01

### 2.4 Hardware Requirements
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: Intel i7-12700K or equivalent
- **RAM**: 32 GB minimum
- **Training Time**: ~30 minutes for full training
- **Inference**: 2.3 ms/sample on CPU

## 3. Evaluation Metrics

### 3.1 Performance Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and per-class F1-score
- **Precision/Recall**: Per-class precision and recall
- **Confusion Matrix**: Detailed error analysis

### 3.2 Explainability Metrics
```python
def evaluate_explainability(model, test_samples):
    # 1. Faithfulness (Deletion Test)
    faithfulness = deletion_test(model, test_samples, k=5)

    # 2. Stability (Perturbation Test)
    stability = perturbation_stability(model, test_samples, epsilon=0.01)

    # 3. Sparsity (Active Rules)
    sparsity = calculate_rule_sparsity(model, test_samples)

    # 4. Consistency (Multiple Runs)
    consistency = run_consistency(model, test_samples, seeds=5)

    return {
        'faithfulness': faithfulness,
        'stability': stability,
        'sparsity': sparsity,
        'consistency': consistency
    }
```

## 4. Reproducibility Protocol

### 4.1 Environment Setup
```bash
# Create conda environment
conda create -n fuzzy_xfd python=3.9
conda activate fuzzy_xfd

# Install dependencies
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121
pip install pytorch-lightning==2.1.3
pip install numpy pandas scikit-learn matplotlib seaborn
pip install wandb  # For experiment tracking
```

### 4.2 Exact Reproduction Commands
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=your_key_here

# Run with fixed seed
python main.py \
    --config_file configs/unified_baseline/config_FuzzyLogic_v2.yaml \
    --seed 42 \
    --gpus 1 \
    --save_dir ./save/reproducibility_test
```

### 4.3 Verification Checklist
- [ ] Same random seeds produce identical results (±0.1% tolerance)
- [ ] Model checkpoint loads correctly
- [ ] All 13 statistical features computed correctly
- [ ] Fuzzy rule parameters match saved values
- [ ] Membership functions generate expected shapes

## 5. Ablation Studies

### 5.1 Component Ablation
```python
ablation_configs = {
    'no_fuzzy': {'use_fuzzy': False},
    'no_statistical': {'use_statistical_features': False},
    'no_deep': {'use_deep_features': False},
    'fixed_fusion': {'adaptive_fusion': False, 'fusion_weight': 0.5},
    'no_exp_constraints': {'lambda_3': 0.0}
}
```

### 5.2 Hyperparameter Sensitivity
- **Learning Rate**: [0.0001, 0.0005, 0.001, 0.005, 0.01]
- **Number of Rules**: [20, 30, 50, 70, 100]
- **Fusion Weight Initial**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Explainability Weight**: [0.001, 0.01, 0.1, 0.5]

## 6. Explainability Evaluation Protocol

### 6.1 Faithfulness Evaluation
```python
def deletion_test(model, sample, k=5):
    """Test if removing important features degrades performance"""
    original_pred = model.predict(sample)

    # Get top-k most important features
    importance = model.get_feature_importance(sample)
    top_k_idx = np.argsort(importance)[-k:]

    # Remove top-k features
    corrupted_sample = sample.copy()
    corrupted_sample[top_k_idx] = 0

    corrupted_pred = model.predict(corrupted_sample)

    # Calculate degradation
    faithfulness = 1 - abs(original_pred - corrupted_pred)
    return faithfulness
```

### 6.2 Stability Evaluation
```python
def stability_test(model, sample, num_perturbations=100):
    """Test if explanations are stable under input perturbations"""
    explanations = []

    for _ in range(num_perturbations):
        # Add Gaussian noise
        perturbed = sample + np.random.normal(0, 0.01, sample.shape)
        exp = model.explain(perturbed)
        explanations.append(exp)

    # Calculate average pairwise similarity
    similarities = []
    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            sim = cosine_similarity(explanations[i], explanations[j])
            similarities.append(sim)

    return np.mean(similarities)
```

### 6.3 Human Evaluation Setup
- **Participants**: 5 domain experts (bearing diagnosis specialists)
- **Evaluation Tasks**:
  1. Rate explanation clarity (1-5 Likert scale)
  2. Verify correctness of rule-based reasoning
  3. Assess trust in model decisions
- **Metrics**: Inter-rater agreement (Fleiss' κ), average rating

## 7. Safety-Critical Case Studies

### 7.1 Aviation Engine Bearing Diagnosis
```python
aviation_case = {
    'scenario': 'A380 engine bearing monitoring at cruise altitude',
    'critical_threshold': 0.95,  # Required confidence for safety
    'decision_timeout': 10,  # ms
    'audit_trail': True
}
```

### 7.2 High-Speed Rail Bogie System
```python
rail_case = {
    'scenario': 'CRH380 train bogie bearing fault detection',
    'speed': 350,  # km/h
    'sampling_rate': 51200,  # Hz
    'false_negative_cost': 'catastrophic'
}
```

### 7.3 Nuclear Plant Cooling Pump
```python
nuclear_case = {
    'scenario': 'Primary cooling pump bearing health monitoring',
    'regulatory_compliance': IEC_61508,
    'safety_integrity_level': 'SIL-3',
    'verification_frequency': 'continuous'
}
```

## 8. Failure Case Collection

### 8.1 Systematic Failure Analysis
```python
def analyze_failures(model, test_set):
    failures = []

    for sample, true_label in test_set:
        pred = model.predict(sample)
        if pred != true_label:
            analysis = {
                'sample_id': sample.id,
                'true_label': true_label,
                'predicted': pred,
                'confidence': model.get_confidence(sample),
                'explanation': model.explain(sample),
                'error_type': classify_error(true_label, pred),
                'criticality': assess_criticality(true_label, pred)
            }
            failures.append(analysis)

    return failures
```

### 8.2 Error Classification
- **Type I**: Healthy → Fault (False Positive)
- **Type II**: Fault → Healthy (False Negative) - *Critical*
- **Type III**: Fault Type Misclassification
- **Type IV**: Low Confidence Prediction

### 8.3 Reporting Template
```markdown
## Failure Case #XXX
- **Sample**: [ID]
- **True Label**: [Class]
- **Predicted**: [Class]
- **Confidence**: [0.xx]
- **Activated Rules**: [List with firing strengths]
- **Root Cause**: [Analysis]
- **Impact**: [Safety assessment]
```

## 9. Cross-Dataset Validation

### 9.1 CWRU Dataset
- **Source**: Case Western Reserve University
- **Adaptation**: Resample to 25.6 kHz
- **Classes**: Same 5 classes
- **Expected Performance**: ±5% of THU_018

### 9.2 XJTU Dataset
- **Source**: Xi'an Jiaotong University
- **Adaptation**: Different bearing sizes
- **Challenge**: Domain shift
- **Evaluation**: Zero-shot vs. Fine-tuning

## 10. Statistical Analysis

### 10.1 Multi-Seed Validation
- **Seeds**: [20, 42, 123, 456, 789]
- **Metrics**: Mean ± Standard Deviation
- **Confidence Interval**: 95% CI using t-distribution
- **Statistical Tests**: Paired t-test vs. baselines

### 10.2 Significance Testing
```python
# Performance significance test
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(
    fuzzy_xfd_accuracies,
    baseline_accuracies
)

significance = "significant" if p_value < 0.05 else "not significant"
```

## 11. Code and Data Availability

### 11.1 Repository Structure
```
https://github.com/your_org/fuzzy-xfd
├── code/
│   ├── model/FuzzyLogic_v2.py
│   ├── trainer/trainer_basic.py
│   └── utils/
├── configs/
│   └── config_FuzzyLogic_v2.yaml
├── data/
│   └── preprocessing/
└── results/
    └── reproducing_paper/
```

### 11.2 Docker Container
```dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["python", "main.py"]
```

### 11.3 Benchmark
```bash
# Benchmark script
python benchmark.py \
    --model_path checkpoint.ckpt \
    --test_data test_set.pkl \
    --output benchmark_results.json \
    --num_runs 100
```

*Note: All experiments were conducted with fixed random seeds. Results may vary slightly due to hardware differences but should remain within reported confidence intervals.*