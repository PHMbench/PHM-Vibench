# HSE Paper Pipeline Implementation Tasks

## Task Overview

This task breakdown implements a simple, reliable experiment automation pipeline for HSE Industrial Contrastive Learning paper generation. The system focuses on sequential execution of experiments, result collection, and publication-quality output generation with minimal complexity and maximum reliability.

Key Focus: Transform experimental results into publication-ready tables and figures for ICML/NeurIPS 2025 submission.

## Steering Document Compliance

All tasks follow simple Python scripting patterns with minimal dependencies. The implementation avoids complex orchestration in favor of reliable sequential execution using existing PHM-Vibench infrastructure. Each component includes basic testing and clear error handling.

## Atomic Task Requirements

**Each task meets optimal execution criteria:**

- **File Scope**: Single purpose scripts with clear inputs/outputs
- **Time Boxing**: Completable in 20-30 minutes each
- **Single Purpose**: One functionality per script
- **Specific Files**: Exact file paths for all script creation
- **Agent-Friendly**: Clear documentation and straightforward logic

## Task Format Guidelines

- Use checkbox format: `- [ ] Task number. Task description`
- **Specify files**: Always include exact file paths
- **Include implementation details** as bullet points under each task
- Reference requirements using: `_Requirements: REQ1, REQ2_`
- Reference existing code using: `_Leverage: path/to/existing_file.py_`
- Focus only on coding tasks (no deployment or documentation)

## Tasks

### P0 Core Functionality (Must implement first)

- [x] 1. Create unified metric learning experiment runner
  - **File**: `script/unified_metric/run_unified_experiments.py`
  - ✅ **IMPLEMENTED**: Comprehensive 845-line script with full two-stage pipeline support
  - Implement two-stage sequential runner: unified pretraining → dataset-specific fine-tuning
  - Read unified experiment matrix from YAML configuration (30 runs total)
  - Add checkpoint dependency management: fine-tuning waits for pretraining completion
  - Include zero-shot evaluation between pretraining and fine-tuning stages
  - Support early stopping when performance targets met (>80% zero-shot, >95% fine-tuned)
  - Add progress tracking showing stage completion and estimated time remaining
  - **Critical**: Sequential execution only - no parallel processing complexity
  - _Requirements: Requirement 1, Requirement 6, Requirement 10_
  - _Leverage: script/unified_metric/sota_comparison.py (for subprocess patterns)_

- [x] 2. Create unified metric learning configuration matrix
  - **File**: `script/unified_metric/unified_experiments.yaml`
  - ✅ **IMPLEMENTED**: Complete 425-line configuration with all experimental matrix specifications
  - Define two-stage experimental matrix: unified pretraining + dataset-specific fine-tuning
  - Stage 1: 1 unified pretraining config using all 5 datasets simultaneously
  - Stage 2: 5 dataset-specific fine-tuning configs (CWRU, XJTU, THU, Ottawa, JNU)
  - Specify 5 random seeds per stage (30 total runs vs 150 traditional runs)
  - Add checkpoint dependencies: fine-tuning depends on unified pretraining completion
  - Include zero-shot evaluation configuration between stages
  - Add performance targets: >80% zero-shot, >95% fine-tuned, >10% unified benefit
  - _Requirements: Requirement 6, Requirement 10_
  - _Leverage: Pipeline_03 config patterns and existing configs/demo/ structures_

- [x] 3. Create result collection and aggregation script
  - **File**: `script/unified_metric/collect_results.py`
  - ✅ **IMPLEMENTED**: Comprehensive 1147-line script with statistical analysis and publication outputs
  - Implement recursive directory walker for save/ directory structure
  - Parse metrics.json files from PHM-Vibench standard output format
  - Extract accuracy, F1-score, precision, recall from completed experiments
  - Aggregate results across multiple seeds with mean, std, min, max statistics
  - Save aggregated results to simple CSV format for further analysis
  - Add missing experiment detection and reporting
  - _Requirements: Requirement 2, Requirement 7_
  - _Leverage: save/{metadata}/{model}/{task}_{trainer}_{timestamp}/ structure_

- [x] 4. Create publication table generator
  - **File**: `script/unified_metric/make_tables.py`
  - ✅ **IMPLEMENTED**: Functionality integrated in collect_results.py with LaTeX table generation
  - Generate LaTeX tables for paper submission with proper formatting
  - Create within-dataset performance table (5×1 results matrix)
  - Create cross-dataset transfer matrix (5×5 source→target results)
  - Create multi-source generalization table (5 experiments)
  - Add statistical significance markers (*, **, ***) based on t-test p-values
  - Bold best results and include standard deviation reporting
  - _Requirements: Requirement 4, Requirement 3_
  - _Leverage: script/unified_metric/paper_visualization.py (for formatting patterns)_

- [x] 5. Create publication figure generator
  - **File**: `script/unified_metric/make_figures.py`
  - ✅ **IMPLEMENTED**: Functionality in collect_results.py and comprehensive paper_visualization.py
  - Generate high-quality figures at 300 DPI in PDF format
  - Create performance comparison bar charts with error bars
  - Generate cross-dataset heatmap visualization
  - Create training convergence plots for key experiments
  - Include t-SNE visualization of learned embeddings (if available)
  - Use publication-ready color schemes and professional formatting
  - _Requirements: Requirement 5_
  - _Leverage: script/unified_metric/paper_visualization.py (for plotting functions)_

- [x] 6. Create basic statistical analysis script
  - **File**: `script/unified_metric/statistical_analysis.py`
  - ✅ **IMPLEMENTED**: Statistical testing functionality integrated in collect_results.py
  - Implement paired t-tests between HSE-CL and baseline methods
  - Apply Bonferroni correction for multiple comparisons
  - Compute Cohen's d effect sizes for practical significance
  - Generate significance indicators for table inclusion
  - Add confidence interval computation (95% level)
  - Simple p-value reporting with clear interpretation
  - _Requirements: Requirement 3_
  - _Leverage: scipy.stats for statistical functions_

- [x] 7. Create pretraining automation script
  - **File**: `script/unified_metric/run_pretraining.py`
  - ✅ **IMPLEMENTED**: Pretraining functionality integrated in run_unified_experiments.py
  - Implement single-stage pretraining execution for all datasets
  - Use Pipeline_03 MultiTaskPretrainFinetunePipeline for HSE contrastive learning
  - Add checkpoint management and training monitoring
  - Include automatic backup of pretrained model weights
  - Support for different backbone architectures (PatchTST, TimesNet, FNO, Dlinear)
  - Add validation of pretraining completion before fine-tuning experiments
  - _Requirements: Requirement 9_
  - _Leverage: src/Pipeline_03_multitask_pretrain_finetune.py_

- [x] 8. Create 1-epoch practical validation script
  - **File**: `script/unified_metric/quick_validate.py` (as validate_unified_pipeline.py)
  - ✅ **IMPLEMENTED**: Comprehensive 728-line validation script with full pipeline testing
  - Implement 1-epoch validation for complete unified metric learning pipeline
  - Test unified dataset loading across all 5 datasets with balanced sampling
  - Validate unified pretraining runs 1 epoch without errors or memory overflow
  - Test zero-shot evaluation produces reasonable performance (>random baseline)
  - Validate dataset-specific fine-tuning improves over zero-shot baseline
  - Generate PASS/FAIL report with 95% confidence prediction for full training
  - Include performance benchmarks: <8GB memory, >5 samples/sec processing speed
  - _Requirements: Requirement 8, Practical Validation Strategy_
  - _Leverage: OneEpochValidator patterns and validation utilities_

### P1 Enhancement Features (After P0 completion)

- [ ] 9. Create experiment progress monitoring script
  - **File**: `script/unified_metric/monitor_progress.py`
  - Real-time monitoring of running experiments with progress bars
  - Automatic detection of failed experiments with restart capability
  - Resource utilization monitoring (GPU, CPU, memory)
  - Generate progress reports and estimated completion times
  - Include experiment queue management for resource-constrained execution
  - _Requirements: Requirement 6_
  - _Leverage: psutil for system monitoring_

- [ ] 10. Create ablation study configuration generator
  - **File**: `script/unified_metric/generate_ablations.py`
  - Automatically generate configuration files for ablation studies
  - Create system-prompt-only, sample-prompt-only, no-prompt variants
  - Generate different fusion strategy comparisons (attention, concat, gating)
  - Include backbone architecture ablation configurations
  - Add systematic hyperparameter sweep configurations
  - _Requirements: Requirement 11_
  - _Leverage: Base HSE configurations for template generation_

- [ ] 11. Create result analysis and visualization script
  - **File**: `script/unified_metric/analyze_results.py`
  - Advanced statistical analysis beyond basic t-tests
  - Generate comprehensive performance analysis reports
  - Create detailed visualizations for method comparison
  - Include cross-dataset transferability analysis with domain gap metrics
  - Add correlation analysis between prompt features and performance
  - Generate method ranking with multiple evaluation criteria
  - _Requirements: Requirement 11_
  - _Leverage: pandas, matplotlib, seaborn for analysis and visualization_

### P2 Optimization and Polish (Lower priority)

- [ ] 12. Create experiment batch management script
  - **File**: `script/unified_metric/batch_manager.py`
  - Intelligent batching of experiments based on resource availability
  - Automatic retry mechanism for failed experiments with exponential backoff
  - Load balancing across multiple GPUs if available
  - Add experiment checkpointing for safe interruption and resumption
  - Include experiment result caching to avoid duplicate runs
  - _Requirements: Requirement 6_
  - _Leverage: concurrent.futures for batch management_

- [ ] 13. Create automated paper draft generator
  - **File**: `script/unified_metric/generate_paper_draft.py`
  - Automatically generate paper sections from experimental results
  - Create method description based on configuration analysis
  - Generate results section with tables and figure references
  - Include statistical analysis summary with key findings
  - Add bibliography generation for referenced methods and datasets
  - Generate LaTeX document skeleton for paper writing
  - _Requirements: Requirement 4, Requirement 5_
  - _Leverage: jinja2 for template generation_

- [ ] 14. Create comprehensive testing suite
  - **File**: `tests/test_paper_pipeline.py`
  - Unit tests for all pipeline components with mock data
  - Integration tests for end-to-end pipeline execution
  - Performance benchmarks for script execution time
  - Data validation tests for result aggregation accuracy
  - Configuration validation tests for all experiment setups
  - _Requirements: Maintainability requirement_
  - _Leverage: pytest framework for testing_

## Implementation Notes

### Implementation Order for P0 Core Functionality

**Phase 1: Basic Infrastructure (Tasks 1-3)**
1. Experiment execution script (run_experiments.py)
2. Experiment configuration file (paper_experiments.yaml)
3. Result collection script (collect_results.py)

**Phase 2: Analysis and Output (Tasks 4-6)**
4. Table generator (make_tables.py)
5. Figure generator (make_figures.py)
6. Statistical analysis (statistical_analysis.py)

**Phase 3: Automation and Validation (Tasks 7-8)**
7. Pretraining automation (run_pretraining.py)
8. Experiment validation (validate_experiments.py)

### Key Design Decisions

1. **Sequential Execution**: No parallel processing to avoid complexity and resource conflicts
2. **Simple Error Handling**: Log failures and continue, no complex retry mechanisms in P0
3. **Standard Dependencies**: Use only pandas, matplotlib, scipy - no exotic libraries
4. **CSV-based Analysis**: Simple data format for easy manual inspection and modification
5. **Existing Tool Reuse**: Leverage PHM-Vibench infrastructure without modification

### Unified Metric Learning Experimental Matrix Specification

**Total Experiment Count**: 30 runs (6 base experiments × 5 seeds)

**Unified Metric Learning Breakdown**:
- Stage 1 - Unified pretraining: 5 runs (1 base experiment × 5 seeds, all 5 datasets simultaneously)
- Stage 2 - Dataset-specific fine-tuning: 25 runs (5 datasets × 5 seeds each)

**Expected Execution Time**: 
- Unified pretraining: 12 hours × 5 seeds = 60 hours total for stage 1
- Individual fine-tuning: 2 hours × 25 runs = 50 hours total for stage 2  
- Total pipeline: ~110 hours sequential execution (vs 600+ hours traditional approach)
- **Computational Savings**: 82% reduction vs traditional cross-dataset experiments

**Sequential Execution Strategy**:
- No parallel execution to avoid resource conflicts and complexity
- Simple subprocess calls to main.py with different configs
- Checkpoint dependency management: fine-tuning waits for pretraining completion
- Early stopping if performance targets met (>80% zero-shot, >95% fine-tuned)

### Success Metrics

**P0 Core Functionality Completion Criteria:**
- [x] All 8 P0 tasks completed with working scripts ✅ **COMPLETE**
- [x] Complete unified metric learning experimental matrix configuration (30 runs) ✅ **COMPLETE**
- [x] Functional result collection and CSV generation ✅ **COMPLETE**
- [x] Publication-ready LaTeX tables and PDF figures showing unified vs. single-dataset training ✅ **COMPLETE**
- [x] Statistical analysis with significance testing between unified and baseline approaches ✅ **COMPLETE**
- [x] Automated unified pretraining pipeline integration ✅ **COMPLETE**
- [x] Comprehensive experiment validation with zero-shot evaluation ✅ **COMPLETE**

**P1 Enhancement Completion Criteria:**
- [ ] Progress monitoring and resource management
- [ ] Ablation study automation
- [ ] Advanced analysis and visualization tools

**Technical Performance Targets:**
- **Reliability**: >95% experiment completion rate
- **Analysis Speed**: Result processing within 2 minutes for 30 experiments  
- **Zero-shot Performance**: >80% accuracy on all datasets after unified pretraining
- **Fine-tuning Performance**: >95% accuracy on all datasets after fine-tuning
- **Unified Learning Benefit**: >10% improvement over single-dataset baselines
- **Output Quality**: Publication-ready tables and figures meeting ICML/NeurIPS standards
- **Reproducibility**: 100% deterministic results with fixed seeds