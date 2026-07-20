# Figures and Tables List
## LLM-Enhanced Explainable Fault Diagnosis Paper

### Main Figures

#### Figure 1: System Architecture
- **Title**: Four-layer architecture of LLM-Enhanced Explainable Fault Diagnosis Toolkit
- **Type**: Architecture diagram
- **Description**: Shows the flow from Signal Processing Layer → Knowledge Enhancement Layer → LLM Integration Layer → Interactive Interface Layer
- **Location**: Section 3.1

#### Figure 2: Structured-to-Natural Language Mapping
- **Title**: Transformation process from model outputs to natural language explanations
- **Type**: Process flow diagram
- **Description**: Illustrates IR format, template matching, and dynamic prompt assembly
- **Location**: Section 3.2

#### Figure 3: Dialogue State Management
- **Title**: Multi-turn dialogue state machine with 9 query types
- **Type**: State diagram
- **Description**: Shows dialogue transitions and response strategies
- **Location**: Section 3.3

#### Figure 4: Evidence Chain Tracking
- **Title**: Evidence chain architecture preventing hallucination
- **Type**: Network diagram
- **Description**: Shows evidence links, validation, and traceability
- **Location**: Section 3.4

#### Figure 5: User Study Results - Explanation Quality
- **Title**: Five-dimensional explanation quality comparison
- **Type**: Radar chart
- **Data**: Our method vs. Traditional visualization vs. Generic LLM
- **Location**: Section 5.1

#### Figure 6: Diagnostic Performance
- **Title**: Accuracy vs. Response Time trade-off
- **Type**: Scatter plot with confidence intervals
- **Description**: Shows maintained accuracy with reduced decision time
- **Location**: Section 5.2

#### Figure 7: Industrial Deployment Impact
- **Title**: Maintenance cost and downtime reduction over time
- **Type**: Line chart with trend lines
- **Data**: 12-month deployment metrics
- **Location**: Section 5.3

#### Figure 8: Failure Analysis Distribution
- **Title**: Categorization of explanation failures
- **Type**: Pie chart with mitigation strategies
- **Location**: Section 6.4

### Main Tables

#### Table 1: Dataset Characteristics
- **Title**: PHM-Vibench dataset statistics
- **Columns**: Dataset | Samples | Fault Types | Load Conditions | Sampling Rate
- **Location**: Section 4.1

#### Table 2: Model Performance Comparison
- **Title**: Transparent model accuracy and parameters
- **Columns**: Model | Parameters | Accuracy | Response Time | Explanation Type
- **Location**: Section 4.1.2

#### Table 3: User Study Demographics
- **Title**: Participant characteristics
- **Columns**: Group | N | Experience | Role | Background
- **Location**: Section 4.2

#### Table 4: Explanation Quality Metrics
- **Title**: Quantitative evaluation across five dimensions
- **Columns**: Dimension | Our Method | Traditional | Generic LLM | p-value
- **Location**: Section 5.1

#### Table 5: Ablation Study Results
- **Title**: Component and prompt ablation impacts
- **Columns**: Configuration | Accuracy | Understandability | Hallucination Rate
- **Location**: Section 5.3

### Supplementary Figures

#### Figure S1: Additional Architecture Details
- Detailed component interactions
- API specifications
- Data flow diagrams

#### Figure S2: Prompt Template Examples
- Standard template
- Technical template
- Maintenance template
- Emergency template

#### Figure S3: Dialogue Examples
- Simple fault diagnosis
- Complex multi-fault interaction
- Emergency response protocol

#### Figure S4: Knowledge Graph Structure
- Fault hierarchy
- Symptom relationships
- Maintenance action mappings

#### Figure S5: Performance Benchmarks
- Concurrent user load testing
- Response time distribution
- Error rate analysis

### Supplementary Tables

#### Table S1: Full Dataset Statistics
- Detailed breakdown by fault type
- Load condition distributions
- Class balance analysis

#### Table S2: Hyperparameter Settings
- Learning rates
- Batch sizes
- Regularization parameters
- Training convergence

#### Table S3: LLM Configuration Details
- Temperature settings
- Token limits
- Rate limiting
- Cost analysis

#### Table S4: Complete User Study Results
- Individual participant scores
- Qualitative feedback summary
- Statistical analysis details

#### Table S5: Failure Case Analysis
- Detailed failure descriptions
- Root cause analysis
- Mitigation effectiveness

### Charts for Case Studies

#### Wind Turbine Case
- **Chart W1**: Vibration spectrum with fault annotations
- **Chart W2**: Maintenance cost comparison (before/after)
- **Chart W3**: ROI analysis over 5 years
- **Chart W4**: Failure prevention timeline

#### High-Speed Rail Case
- **Chart R1**: Real-time monitoring dashboard
- **Chart R2**: Emergency response flow
- **Chart R3**: Service disruption statistics
- **Chart R4**: System reliability metrics

### Visual Elements for UI

#### UI Mockups
- Conversation interface screenshot
- Evidence visualization
- Interactive charts
- Mobile responsive views

#### Iconography
- Fault type icons
- Severity indicators
- Status badges
- Navigation elements

---

## Figure Generation Scripts

### Python Scripts Location
- `/scripts/generate_figure_1.py` - Architecture diagram
- `/scripts/generate_figure_5.py` - Radar chart
- `/scripts/generate_table_4.py` - Quality metrics table

### Data Files
- `/data/user_study_results.csv` - Raw user study data
- `/data/industrial_metrics.json` - Deployment statistics
- `/data/ablation_results.xlsx` - Ablation study data

---

## Formatting Specifications

### Figure Requirements
- **Format**: Vector graphics (SVG) preferred, PNG minimum 300 DPI
- **Size**: Column width = 85mm, Full width = 174mm
- **Font**: Arial, 8pt for labels, 10pt for titles
- **Color**: Colorblind-friendly palette
- **File naming**: figX_description.ext

### Table Requirements
- **Format**: LaTeX tabular environment
- **Font**: 9pt
- **Alignment**: Numeric columns right-aligned
- **Captions**: Above table, concise description
- **File naming**: tableX_description.tex

### Supplementary Material Organization
```
supplementary/
├── figures/
│   ├── S1_architecture_details.pdf
│   ├── S2_prompt_templates.pdf
│   └── ...
├── tables/
│   ├── S1_dataset_stats.csv
│   ├── S2_hyperparameters.csv
│   └── ...
├── data/
│   ├── user_study_raw.zip
│   ├── experimental_logs.tar.gz
│   └── ...
└── code/
    ├── reproduction/
    ├── analysis/
    └── visualization/
```

---

## Checklist for Submission

### Required Elements
- [ ] All figures in publication-ready format
- [ ] All tables in LaTeX format
- [ ] Figure captions complete
- [ ] Table captions complete
- [ ] In-text citations correct
- [ ] Supplementary material organized
- [ ] Data availability statement

### Optional but Recommended
- [ ] High-resolution versions for review
- [ ] Source code for visualizations
- [ ] Interactive versions (where applicable)
- [ ] Color versions for online publication
- [ ] Grayscale versions for print

---

## Notes for Reviewers

1. **Figure placeholders** in the main text indicate where figures should be inserted
2. **Table references** use LaTeX labels for cross-referencing
3. **Supplementary figures** are referenced as Fig. S1, S2, etc.
4. **Supplementary tables** are referenced as Table S1, S2, etc.
5. **All generated plots** use consistent styling and color schemes
6. **Statistical significance** is indicated with appropriate asterisks (*, **, ***)