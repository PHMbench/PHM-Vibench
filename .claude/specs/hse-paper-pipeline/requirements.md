# Requirements Document - HSE Paper Pipeline

## Introduction

The HSE Paper Pipeline system is a comprehensive experiment execution and analysis framework designed to streamline the workflow from running machine learning experiments to generating publication-ready tables and figures for the HSE Industrial Contrastive Learning paper. This system will automate the entire research pipeline, including experiment execution, metrics collection, statistical analysis, and visualization generation, enabling researchers to focus on scientific insights rather than manual data processing.

The system transforms the existing `unified_metric` infrastructure into a production-level pipeline that ensures reproducibility, statistical rigor, and publication quality outputs for top-tier machine learning conferences.

## Alignment with Product Vision

This feature directly supports the PHM-Vibench platform's mission of providing a comprehensive benchmark for industrial equipment vibration signal analysis by:
- **Research Acceleration**: Automating the complete experimental workflow from data to publication
- **Scientific Rigor**: Ensuring statistical significance testing and proper experimental methodology  
- **Reproducibility**: Creating standardized pipelines for experiment replication
- **Publication Quality**: Generating professional-grade tables and figures meeting conference standards
- **Integration**: Leveraging existing PHM-Vibench factory patterns and configuration systems

## Requirements

### Requirement 1: Automated Experiment Execution Pipeline

**User Story:** As a researcher, I want to execute comprehensive experiment matrices automatically, so that I can run baseline, HSE, and SOTA comparisons efficiently without manual intervention.

#### Acceptance Criteria

1. WHEN I specify an experiment configuration matrix THEN the system SHALL execute all experiments in parallel using available computational resources
2. IF an experiment fails during execution THEN the system SHALL log the failure, continue with remaining experiments, and provide a detailed failure report
3. WHEN experiments are running THEN the system SHALL display real-time progress tracking with completion percentages and estimated time remaining
4. WHEN all experiments complete THEN the system SHALL generate a comprehensive execution summary with success/failure statistics

### Requirement 2: Metrics Collection and Aggregation System

**User Story:** As a researcher, I want automatic metrics collection from multiple experiment runs, so that I can aggregate results across different random seeds and configurations for statistical analysis.

#### Acceptance Criteria

1. WHEN an experiment completes THEN the system SHALL automatically parse metrics.json files and extract performance indicators (accuracy, F1, precision, recall, training time)
2. IF multiple runs exist for the same configuration THEN the system SHALL aggregate metrics across runs and compute mean, standard deviation, and confidence intervals
3. WHEN parsing checkpoint directories THEN the system SHALL identify best performing models and extract their metrics
4. WHEN collecting metrics THEN the system SHALL integrate with MetricsMarkdownReporter from the loop_id branch for automatic table generation

### Requirement 3: Statistical Analysis and Significance Testing

**User Story:** As a researcher, I want statistical significance testing between different methods, so that I can make scientifically valid claims about performance improvements in publications.

#### Acceptance Criteria

1. WHEN comparing two methods THEN the system SHALL perform paired t-tests and report p-values with appropriate significance levels (p < 0.05, p < 0.01, p < 0.001)
2. IF multiple comparisons are made THEN the system SHALL apply Bonferroni correction to control family-wise error rate
3. WHEN analyzing method differences THEN the system SHALL compute Cohen's d effect sizes to quantify practical significance
4. WHEN generating statistical reports THEN the system SHALL include confidence intervals at 95% confidence level for all performance metrics

### Requirement 4: Publication-Quality Table Generation

**User Story:** As a researcher, I want automatic generation of LaTeX tables from experiment results, so that I can directly include professionally formatted tables in my paper submissions.

#### Acceptance Criteria

1. WHEN experiment results are aggregated THEN the system SHALL generate LaTeX tables with proper formatting, bold best results, and statistical significance indicators
2. IF cross-dataset experiments are conducted THEN the system SHALL create cross-dataset performance matrices with source-target accuracy pairs
3. WHEN generating ablation study tables THEN the system SHALL highlight performance differences and statistical significance markers (*, **, ***)
4. WHEN creating SOTA comparison tables THEN the system SHALL include method names, accuracies, standard deviations, and computational efficiency metrics

### Requirement 5: High-Quality Figure Generation

**User Story:** As a researcher, I want automatic generation of publication-quality figures, so that I can include professional visualizations in my paper that meet conference standards.

#### Acceptance Criteria

1. WHEN generating architecture diagrams THEN the system SHALL create clear, publication-ready figures at 300 DPI resolution in PDF format
2. IF t-SNE visualizations are requested THEN the system SHALL generate embedding plots with proper color schemes, legends, and annotations
3. WHEN creating performance comparison plots THEN the system SHALL use colorblind-friendly palettes and include error bars with confidence intervals
4. WHEN generating training curves THEN the system SHALL show convergence behavior with proper axis labels and professional formatting

### Requirement 6: Batch Experiment Management

**User Story:** As a researcher, I want to manage and monitor large-scale experiment batches, so that I can efficiently utilize computational resources and track experiment progress.

#### Acceptance Criteria

1. WHEN launching batch experiments THEN the system SHALL support parallel execution across multiple GPUs with automatic load balancing
2. IF system resources are limited THEN the system SHALL queue experiments and execute them as resources become available
3. WHEN monitoring batch progress THEN the system SHALL provide real-time status updates with completion estimates and resource utilization
4. WHEN experiments fail THEN the system SHALL implement automatic retry mechanisms with exponential backoff for transient failures

### Requirement 7: Result Aggregation and Cross-Run Analysis

**User Story:** As a researcher, I want to aggregate results across multiple experimental runs, so that I can account for random variation and report robust performance statistics.

#### Acceptance Criteria

1. WHEN multiple runs exist for the same configuration THEN the system SHALL aggregate results and compute descriptive statistics (mean, median, std, min, max)
2. IF runs have different random seeds THEN the system SHALL track seed-specific results and analyze variance across seeds
3. WHEN aggregating cross-dataset results THEN the system SHALL create comprehensive performance matrices showing all source-target combinations
4. WHEN computing aggregate statistics THEN the system SHALL detect and flag outlier runs that deviate significantly from the mean

### Requirement 8: Integration with Existing PHM-Vibench Infrastructure

**User Story:** As a researcher, I want seamless integration with existing PHM-Vibench components, so that I can leverage existing models, datasets, and configurations without code duplication.

#### Acceptance Criteria

1. WHEN executing experiments THEN the system SHALL use existing main.py entry points and configuration system without modification
2. IF new components are added THEN they SHALL follow existing factory patterns and registration mechanisms
3. WHEN integrating with MetricsMarkdownReporter THEN the system SHALL merge components from loop_id branch without conflicts
4. WHEN saving results THEN the system SHALL use existing save/ directory structure and naming conventions

### Requirement 9: Two-Stage Training Workflow

**User Story:** As a researcher, I want a clear two-stage training process (unsupervised pretraining → supervised fine-tuning), so that I can evaluate the effectiveness of cross-system transfer learning.

#### Acceptance Criteria

1. WHEN running pretraining stage THEN the system SHALL use all 5 datasets (CWRU, XJTU, THU, Ottawa, JNU) without labels for contrastive learning
2. WHEN fine-tuning stage starts THEN the system SHALL load pretrained backbone weights and freeze all layers except the classification head
3. IF pretraining completes successfully THEN the system SHALL save backbone checkpoints for use in multiple fine-tuning experiments
4. WHEN evaluating two-stage training THEN the system SHALL compare against single-stage training baselines

### Requirement 10: Unified Metric Learning Evaluation

**User Story:** As a researcher, I want to evaluate unified metric learning across industrial datasets, so that I can measure the effectiveness of learning universal representations from multiple systems.

#### Acceptance Criteria

1. WHEN running unified pretraining THEN the system SHALL train on all 5 datasets simultaneously (CWRU, XJTU, THU, Ottawa, JNU) using HSE contrastive learning
2. IF pretraining completes successfully THEN the system SHALL execute 5 separate fine-tuning experiments (one per dataset)
3. WHEN evaluating generalization THEN the system SHALL compute within-dataset performance after unified pretraining and dataset-specific fine-tuning
4. WHEN generating results THEN the system SHALL create performance comparison showing unified vs. single-dataset training

### Requirement 11: Theoretical Metrics and Analysis

**User Story:** As a researcher, I want theoretical metrics that justify HSE contrastive learning, so that I can provide solid theoretical foundation for the approach.

#### Acceptance Criteria

1. WHEN analyzing HSE embeddings THEN the system SHALL compute domain invariance metrics using feature similarity measures
2. IF system prompts are enabled THEN the system SHALL quantify system-awareness through embedding separability analysis
3. WHEN comparing methods THEN the system SHALL report transferability coefficients showing cross-domain effectiveness
4. WHEN generating theoretical analysis THEN the system SHALL include visualization of learned prompt embeddings and their clustering properties

## Non-Functional Requirements

### Performance
- The system SHALL complete unified pretraining stage within 12 hours on a single GPU (training on all 5 datasets)
- Fine-tuning experiments SHALL complete within 2 hours per dataset
- Unified pretraining SHALL achieve zero-shot performance >80% on all 5 datasets
- Fine-tuning SHALL achieve >95% accuracy on CWRU, XJTU, THU, Ottawa, JNU datasets after unified pretraining
- Universal representation SHALL demonstrate >10% improvement over single-dataset training baselines
- Result processing and aggregation SHALL complete within 2 minutes for 30 experiment results
- Statistical analysis and table generation SHALL complete within 1 minute
- Figure generation SHALL produce publication-quality outputs at 300 DPI within 1 minute per figure

### Computational Requirements
- **Minimum Hardware**: Single GPU with 8GB VRAM (NVIDIA GTX 1080 or equivalent)
- **Recommended Hardware**: NVIDIA RTX 3080/4080 with 16GB VRAM for optimal performance
- **System Memory**: 16GB RAM minimum, 32GB recommended for large batch processing
- **Storage**: 200GB available space (100GB for datasets, 100GB for results)
- **CPU**: 8 cores minimum for data preprocessing and result analysis
- **Operating System**: Linux Ubuntu 18.04+ or CentOS 7+ (Windows/macOS supported but not optimized)

### Reliability
- The system SHALL handle individual experiment failures gracefully without stopping the entire batch
- Results SHALL be automatically saved to persistent storage after each experiment completion
- The system SHALL support resuming interrupted batch experiments from checkpoints
- All generated tables and figures SHALL be reproducible given the same input data

### Usability
- The system SHALL provide clear command-line interfaces with intuitive parameter names
- Progress reporting SHALL include estimated completion times and current status for all running experiments
- Error messages SHALL be actionable and include specific guidance for resolution
- Generated outputs SHALL be organized in logical directory structures with clear naming conventions

### Scalability  
- The system SHALL support sequential execution of 30 experiment runs (6 base experiments × 5 random seeds)
- Memory usage SHALL be limited to single-experiment requirements (no batch processing)
- The system SHALL support adding new datasets to unified pretraining by extending the experiment configuration YAML
- Result storage SHALL use simple CSV format for easy analysis and archiving

### Maintainability
- All scripts SHALL use simple Python with standard libraries (pandas, matplotlib, subprocess)
- Each script SHALL have a single clear purpose and be understandable by reading the code
- Configuration files SHALL use simple YAML format compatible with existing PHM-Vibench configs
- Code SHALL avoid complex abstractions and use direct, straightforward implementations

## Success Metrics and Validation

### Quantitative Success Criteria

#### Experimental Coverage Metrics
- **Complete Experiment Matrix**: 30 total runs (6 base experiments × 5 seeds) successfully executed
- **Unified Pretraining**: 1/1 pretraining experiment on all 5 datasets achieves convergence
- **Dataset-Specific Fine-tuning**: 5/5 fine-tuning experiments achieve >95% accuracy
- **Universal Representation Quality**: Zero-shot performance >80% on all datasets before fine-tuning
- **Statistical Significance**: >80% of comparisons show p < 0.01 significance between unified vs. single-dataset approaches

#### Pipeline Reliability Metrics
- **Experiment Success Rate**: >95% of individual experiments complete successfully
- **Result Collection Rate**: >99% of completed experiments have parseable results
- **Reproducibility Rate**: 100% of experiments produce identical results with same seed
- **Error Recovery Rate**: >90% of failed experiments succeed on retry

#### Output Quality Metrics
- **Table Generation**: 100% of LaTeX tables compile without errors in standard document classes
- **Figure Quality**: 100% of figures meet 300 DPI publication standards
- **Statistical Accuracy**: 100% of significance tests use appropriate corrections (Bonferroni)
- **Data Integrity**: 100% of aggregated results match source metrics.json files

### Qualitative Success Criteria

#### Scientific Contribution
- **Theoretical Foundation**: Clear mathematical formulation of prompt-guided contrastive learning
- **Empirical Validation**: Comprehensive ablation studies demonstrate method effectiveness
- **Cross-System Generalization**: Evidence of transferability across industrial datasets
- **SOTA Comparison**: Competitive or superior performance vs. existing methods

#### Publication Readiness
- **Table Standards**: All tables formatted to ICML/NeurIPS submission guidelines
- **Figure Standards**: All figures use colorblind-friendly palettes and clear annotations
- **Statistical Rigor**: All claims supported by appropriate significance tests
- **Reproducibility**: Complete experimental setup documented for replication

### Validation Methodology

#### Pre-Implementation Validation
1. **Requirements Verification**: All functional requirements mapped to implementation tasks
2. **Resource Allocation**: Computational requirements verified against available hardware
3. **Timeline Validation**: Implementation schedule allows sufficient testing time

#### During Implementation Validation
1. **Component Testing**: Each script validated with sample data before integration
2. **Integration Testing**: End-to-end pipeline tested with subset of experiments
3. **Performance Monitoring**: Execution times tracked against target metrics

#### Post-Implementation Validation
1. **Result Verification**: Statistical analysis validated against manual calculations
2. **Output Quality Review**: Tables and figures reviewed by domain experts
3. **Reproducibility Testing**: Complete pipeline executed independently to verify results