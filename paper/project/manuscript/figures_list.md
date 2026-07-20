# Fuzzy-XFD Paper: Figure and Table List

## Figures

### Figure 1: Four-Layer Neuro-Symbolic Architecture
- **Type**: Architecture diagram
- **Content**: Visual representation of the four layers (Signal Processing → Feature Extraction → Symbolic Reasoning → Linguistic Explanation)
- **Caption**: Fig. 1. Fuzzy-XFD four-layer architecture integrating neural networks with fuzzy logic for explainable fault diagnosis
- **Notes**: Show data flow from raw vibration signal through processing layers to final explanation
- **Format**: 2-column width, color-coded layers

### Figure 2: Fuzzy Logic System Detailed View
- **Type**: System diagram
- **Content**: Detailed view of fuzzy membership functions, rule inference, and defuzzification process
- **Caption**: Fig. 2. Detailed fuzzy logic system showing membership functions, rule activation, and decision-making process
- **Notes**: Include example of Gaussian membership functions and rule firing visualization
- **Format**: 1.5-column width

### Figure 3: Confusion Matrix
- **Type**: Heatmap matrix
- **Content**: 5×5 confusion matrix showing per-class performance (H, IF, OF, BF, CF)
- **Caption**: Fig. 3. Confusion matrix of Fuzzy-XFD on THU_018 test set
- **Data**:
  - H→H: 98.2%
  - IF→IF: 76.5%
  - OF→OF: 68.7%
  - BF→BF: 62.1%
  - CF→CF: 58.3%
- **Format**: Square matrix with color intensity

### Figure 4: Rule Activation Patterns
- **Type**: Bar chart/Heatmap combination
- **Content**: Typical rule activation patterns for each fault type
- **Caption**: Fig. 4. Rule activation patterns for different fault types showing distinct rule combinations
- **Notes**: Show top 5 most activated rules per class with firing strengths
- **Format**: 2-column width

### Figure 5: Performance vs Parameter Count
- **Type**: Scatter plot
- **Content**: Performance comparison across different methods showing accuracy vs parameter count
- **Caption**: Fig. 5. Accuracy vs parameter count for different fault diagnosis methods
- **Data**: Fuzzy-XFD (7.6K, 70.7%), TSPN (76K, 65.2%), CNN (125K, 62.8%), etc.
- **Format**: Log scale for x-axis (parameters)

### Figure 6: Membership Function Visualization
- **Type**: Line plots
- **Content**: Example learned membership functions for key features (RMS, kurtosis, entropy)
- **Caption**: Fig. 6. Learned Gaussian membership functions for key statistical features
- **Notes**: Show Low, Medium, High membership for 3 representative features
- **Format**: 3 subplots side by side

### Figure 7: Safety-Critical Case Study Timeline
- **Type**: Timeline visualization
- **Content**: Aviation engine bearing fault detection timeline with rule activations
- **Caption**: Fig. 7. Timeline of fault detection in aviation engine with rule-based evidence chain
- **Notes**: Show progressive rule activation and confidence increase
- **Format**: Horizontal timeline with annotation

### Figure 8: Explainability Comparison
- **Type**: Bar chart
- **Content**: Comparison of Fuzzy-XFD with LIME and SHAP across faithfulness, conciseness, generation time
- **Caption**: Fig. 8. Explainability metrics comparison between intrinsic and post-hoc explanation methods
- **Data**: From Table 2 in paper
- **Format**: Grouped bar chart

### Figure 9: Ablation Study Results
- **Type**: Waterfall chart
- **Content**: Impact of removing each component on overall performance
- **Caption**: Fig. 9. Ablation study showing performance degradation when removing key components
- **Notes**: Start with 70.7% and show drops for each removed component
- **Format**: 2-column width

### Figure 10: Cross-Dataset Performance
- **Type**: Bar chart with error bars
- **Content**: Performance on THU_018 and CWRU datasets
- **Caption**: Fig. 10. Cross-dataset generalization performance 【TODO-EXP】
- **Notes**: Show 95% confidence intervals
- **Format**: Grouped bar chart

## Tables

### Table 1: Performance Comparison
- **Type**: Results table
- **Content**: Comparison of Fuzzy-XFD with baselines on accuracy, F1-score, parameters, inference time
- **Caption**: Table 1. Performance comparison on THU_018 dataset
- **Data**: Already in Section 4.2.1
- **Format**: 5-column table with best values in bold

### Table 2: Explainability Comparison
- **Type**: Results table
- **Content**: Comparison of explanation methods on faithfulness, conciseness, generation time
- **Caption**: Table 2. Explainability comparison between different approaches
- **Data**: Already in Section 4.3.4
- **Format**: 4-column table

### Table 3: Ablation Study Results
- **Type**: Results table
- **Content**: Impact of component removal on performance and explainability
- **Caption**: Table 3. Ablation study results
- **Data**: Already in Section 5.3.1
- **Format**: 3-column table

### Table 4: Hyperparameter Sensitivity
- **Type**: Analysis table
- **Content**: Performance variation across different hyperparameter values
- **Caption**: Table 4. Hyperparameter sensitivity analysis
- **Data**:
  - Learning rate: 0.0005-0.002
  - Number of rules: 30-100
  - Fusion weight: Fixed vs adaptive
- **Format**: 3 sections for each parameter group

### Table 5: Rule Statistics
- **Type**: Analysis table
- **Content**: Statistics on learned fuzzy rules
- **Caption**: Table 5. Statistical analysis of learned fuzzy rules
- **Data**:
  - Average rules activated: 8.2 ± 2.1
  - Features per rule: 3.4 ± 1.2
  - Rule coverage: 94.3%
- **Format**: Multiple rows for different statistics

## Supplementary Figures

### Figure S1: Training Curves
- **Type**: Line plots
- **Content**: Training and validation loss/accuracy curves over epochs
- **Caption**: Fig. S1. Training curves showing convergence and early stopping point
- **Notes**: Include both loss and accuracy plots

### Figure S2: ROC Curves
- **Type**: Line plots
- **Content**: ROC curves for each fault class (one-vs-rest)
- **Caption**: Fig. S2. ROC curves for multi-class classification
- **Notes**: Include AUC values for each class

### Figure S3: Feature Importance Distribution
- **Type**: Bar chart
- **Content**: Frequency of each feature in rule antecedents
- **Caption**: Fig. S3. Feature importance distribution in learned fuzzy rules
- **Notes**: Show top 10 most used features

### Figure S4: Noise Robustness
- **Type**: Line plot
- **Content**: Performance vs. noise level (SNR)
- **Caption**: Fig. S4. Model robustness to input noise
- **Data**: SNR from -20dB to 30dB
- **Notes**: Include both accuracy and explainability stability

## Supplementary Tables

### Table S1: Full Hyperparameter Configuration
- **Type**: Configuration table
- **Content**: Complete list of hyperparameters and their values
- **Caption**: Table S1. Complete hyperparameter configuration
- **Notes**: Include all optimizer, scheduler, and architecture parameters

### Table S2: Computational Resource Usage
- **Type**: Resource table
- **Content**: Memory usage, GPU utilization, energy consumption
- **Caption**: Table S2. Computational resource requirements
- **Data**: Training time per epoch, peak memory, energy cost

### Table S3: Case Study Details
- **Type**: Case study table
- **Content**: Detailed information about safety-critical cases
- **Caption**: Table S3. Detailed analysis of safety-critical case studies
- **Notes**: Include timestamps, confidence scores, expert evaluations

## Figure Preparation Guidelines

1. **Format Requirements**:
   - Resolution: 300 DPI minimum
   - File format: PNG or EPS
   - Font size: 8-12 pt for labels, 12-14 pt for titles
   - Line width: 0.5-1.0 pt

2. **Color Scheme**:
   - Primary: Blue (#1f77b4)
   - Secondary: Orange (#ff7f0e)
   - Success: Green (#2ca02c)
   - Danger: Red (#d62728)
   - Warning: Yellow (#ffec78)

3. **Consistency**:
   - Use consistent fonts throughout
   - Maintain uniform aspect ratios
   - Follow IEEE formatting guidelines

4. **Accessibility**:
   - Ensure color-blind friendly palettes
   - Include legends where necessary
   - Use patterns in addition to colors when possible