# Experimental Protocol and Reproducibility Checklist

## 1. Dataset Details

### 1.1 Synthetic Dataset
- **Generation Code**: `experiments/proposition2_simple.py` (generate_synthetic_data function)
- **Signal Parameters**:
  - Sampling rate: 10 kHz
  - Duration: 1 second (10,000 samples → downscaled to 50 for efficiency)
  - Base frequency range: 10-50 Hz
  - Fault types: 4 classes (Normal, Inner Race, Outer Race, Rolling Element)
- **Noise Injection**: Gaussian noise with σ = noise_level × signal_std
- **Data Split**: 800 training samples, 200 test samples per experiment

### 1.2 Real-World Datasets
[TODO - Add details for THU-018 and CWRU datasets]

## 2. Model Architecture

### 2.1 Baseline Model
```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim=50, num_classes=4, use_physics=False):
        self.physics_constraint = SimplePhysicsConstraint(input_dim, use_physics)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
```

### 2.2 Physics-Informed Constraints
- **Energy Conservation**: $\|\|x\|_2^2 - \|f(x)\|_2^2\|_1$
- **Frequency Filtering**: Learnable filter on FFT coefficients
- **Smoothness Constraint**: $\sum_t \|x_{t+1} - x_t\|_2^2$

## 3. Training Details

### 3.1 Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 30 (P0 experiment), 100-200 (recommended for P1)
- **Seeds**: 20, 42, 100 (for reproducibility)

### 3.2 Loss Function
```python
if use_physics:
    loss = CE_loss + 0.1 * physics_loss
else:
    loss = CE_loss
```

## 4. Evaluation Metrics

### 4.1 Performance Metrics
- **Classification Accuracy**: $\frac{\text{correct predictions}}{\text{total predictions}}$
- **F1-Score**: Harmonic mean of precision and recall
- **Robustness Score**: $\frac{\text{accuracy}_{noisy}}{\text{accuracy}_{clean}}$

### 4.2 Explainability Metrics
- **Faithfulness**: Correlation between explanation importance and actual feature importance
- **Stability**: Jensen-Shannon divergence between explanations of similar inputs
- **Simplicity**: Number of concepts/rules in explanation (normalized)

## 5. Experiment Protocols

### 5.1 Proposition 2 Validation (Physical Homomorphism)
1. **Objective**: Verify that physical constraints enhance robustness
2. **Procedure**:
   - Train models with/without physics constraints
   - Test on 5 noise levels: [0.0, 0.05, 0.1, 0.15, 0.2]
   - Repeat for 3 random seeds
3. **Expected Outcome**: Physics-informed model maintains higher accuracy

### 5.2 Proposition 1 Validation (Symbolic Constraints)
[TODO - Design protocol for symbolic constraint experiments]

### 5.3 Proposition 3 Validation (Pareto Boundary)
[TODO - Design protocol for interpretability-performance tradeoff]

## 6. Ablation Studies

### 6.1 Constraint Types
- No constraints (baseline)
- L1 regularization only
- L2 regularization only
- Energy conservation only
- Full physics constraints
- Hybrid (physics + L1)

### 6.2 Layer Analysis
- Signal processing only
- Feature extraction only
- Symbolic reasoning only
- Full four-layer architecture

## 7. Computational Requirements

### 7.1 Hardware
- **GPU**: Optional (CPU training takes ~2x longer)
- **Memory**: Minimum 4GB RAM
- **Storage**: < 1GB for all experiments

### 7.2 Runtime
- **Single experiment**: ~5 minutes (30 epochs)
- **Full validation**: ~2 hours (all seeds and conditions)

## 8. Failure Case Collection

### 8.1 Criteria for Failure
- Accuracy drop > 20% compared to baseline
- Explanation contradicts domain knowledge
- Training instability (loss oscillations)

### 8.2 Documentation Template
```
Failure Case ID: FC_001
Condition: High noise (0.3) + Inner race fault
Observation: Model misclassifies as normal
Explanation: Physics constraint oversmoothes signal
Proposed Fix: Adaptive constraint strength
```

## 9. Reproducibility Checklist

- [ ] Random seeds set for Python, NumPy, PyTorch
- [ ] All dependencies versions fixed (requirements.txt)
- [ ] Data generation code deterministic
- [ ] Model initializations logged
- [ ] Full hyperparameters documented
- [ ] Raw results saved (not just aggregated)
- [ ] Visualization generation scripts included

## 10. Code Structure

```
experiments/
├── proposition2_simple.py      # Main experiment code
├── proposition2_redesigned.py  # Full version (with bugs)
├── physics_informed_model.py   # Physics constraint implementation
└── results/
    └── proposition2_12_14/
        ├── simple_results.json
        └── simple_validation.png
```

## 11. Minimum Additional Experiments (Week-scale)

1. **Real Data Validation** (2 days):
   - Load THU-018 dataset
   - Apply same preprocessing as synthetic
   - Run physics-informed vs baseline comparison

2. **Constraint Strength Analysis** (2 days):
   - Vary λ from 0.01 to 1.0
   - Plot performance vs constraint strength
   - Identify optimal range

3. **Explanation Quality Evaluation** (3 days):
   - Generate explanations for 100 test cases
   - Have 2 domain experts rate quality (1-5)
   - Calculate inter-rater reliability