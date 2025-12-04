# Requirements Document: Paper Experiment Infrastructure

## Introduction

**Feature Name**: paper-experiment-infrastructure
**Purpose**: Reorganize SOTA comparison methods into PHM-Vibench src architecture and create comprehensive paper experiment plan
**Target Venue**: ICML/NeurIPS 2025

This feature addresses the critical need for a systematic, reproducible experimental infrastructure that integrates seamlessly with the existing HSE contrastive learning implementation. By reorganizing comparison methods into the modular PHM-Vibench architecture and providing comprehensive experiment documentation, researchers can efficiently conduct rigorous academic experiments that meet top-tier publication standards.

**User Value**: Enables researchers to execute standardized, reproducible experiments with integrated SOTA baselines, reducing experiment setup time from days to minutes while ensuring publication-quality results and proper code organization following PHM-Vibench design patterns.

## Alignment with Product Vision

This feature directly supports PHM-Vibench's mission as a comprehensive benchmark platform for industrial equipment vibration signal analysis by:

1. **Enhancing Benchmark Completeness**: Integrating 8 SOTA domain adaptation methods provides comprehensive baseline comparisons essential for academic validation
2. **Improving Reproducibility**: Standardized experiment infrastructure ensures consistent, reproducible results across different research teams
3. **Supporting Academic Excellence**: Streamlined paper experiment workflows enable researchers to focus on innovation rather than infrastructure setup
4. **Maintaining Architecture Consistency**: Proper integration into factory design patterns preserves codebase maintainability and extensibility

## Requirements

### Requirement 1: SOTA Method Architecture Integration [P0]

**User Story**:
As a researcher, I want all comparison methods to follow the unified task_factory pattern, so that I can maintain code consistency, leverage existing PHM-Vibench infrastructure, and execute standardized experiments with consistent parameters across all SOTA baselines.

**Acceptance Criteria**:
- WHEN user configures SOTA method experiments THEN system should load methods through unified task_factory interface
- IF configuration file specifies comparison method name THEN system should automatically instantiate corresponding task implementation  
- WHEN running comparison experiments THEN all methods should use identical data loading and evaluation pipelines
- WHEN integrating new SOTA method THEN it should follow existing registration patterns in task_factory
- IF method requires custom loss functions THEN they should be registered in Components/loss.py following existing patterns

**SOTA Contrastive Learning Methods to Integrate**:
1. InfoNCE/NT-Xent (Noise Contrastive Estimation)
2. TripletLoss (Metric learning with margin)
3. SupConLoss (Supervised Contrastive Learning)
4. PrototypicalLoss (Prototypical Networks for few-shot)
5. BarlowTwinsLoss (Self-supervised without negatives)
6. VICRegLoss (Variance-Invariance-Covariance regularization)

**Note**: Domain adaptation baselines (DANN, CORAL, MMD, CDAN, MCD, SHOT, NRC) will be implemented as comparison methods, but the core requirement is implementing state-of-the-art contrastive learning losses that align with HSE's approach of pulling similar samples together and pushing different samples apart.

### Requirement 2: Paper Experiment Plan Documentation [P0]

**User Story**:
As a paper author, I want a comprehensive experiment plan document with clear execution steps and expected outcomes, so that I can systematically execute all required experiments and ensure complete coverage of validation scenarios for academic publication.

**Acceptance Criteria**:
- WHEN reviewing experiment plan THEN it should include baseline comparisons, ablation studies, cross-dataset generalization, and all experiment types required for paper
- IF executing experiment plan THEN each step should have clear commands and expected outputs documented
- WHEN generating paper results THEN corresponding visualization and table generation instructions should be provided
- IF reproducing experiments THEN all random seeds and environment settings should be specified
- WHEN conducting statistical analysis THEN significance testing methods and thresholds should be documented

### Requirement 3: Data Path Configuration Standardization [P1]

**User Story**:
As an experiment environment administrator, I want all configurations to use standardized data paths, so that experiments can be executed consistently across different environments.

As a user, I want configuration file path settings to be correct and consistent, so that I don't encounter path-related errors when running experiments.

**Acceptance Criteria**:
- WHEN checking configuration files THEN all data_dir should point to "/home/user/data/PHMbenchdata/PHM-Vibench"
- IF running any experiment THEN it should use metadata_file: "metadata_6_1.xlsx"
- WHEN adding new configurations THEN they should automatically inherit standard path settings
- IF deploying to new environment THEN path updates should be centralized and consistent
- WHEN validating configurations THEN system should verify path accessibility and file existence

### Requirement 4: Experiment Results Collection System [P1]

**User Story**:
As a researcher, I want experimental results to be automatically collected, organized, and processed into publication-ready materials, so that I can focus on analysis and interpretation rather than manual result management and formatting.

**Acceptance Criteria**:
- WHEN experiments complete THEN results should be automatically saved to structured directories with consistent naming conventions
- IF running results analysis THEN system should generate statistical significance tests and comparison tables
- WHEN generating paper materials THEN system should output publication-ready figures with proper formatting
- IF comparing multiple runs THEN system should aggregate results with confidence intervals
- WHEN archiving results THEN complete experimental metadata should be preserved for reproducibility

### Requirement 5: Visualization Code Reorganization [P2]

**User Story**:
As a code maintainer, I want visualization code to be functionally organized in modules, so that it follows PHM-Vibench architecture patterns and is maintainable.

As a developer, I want visualization tools to be reusable by other modules, so that I can leverage existing plotting functions in new contexts.

**Acceptance Criteria**:
- WHEN reorganizing visualization code THEN it should be moved to src/utils/visualization/ following factory patterns
- IF using visualization functionality THEN it should be accessible through standard import paths
- WHEN adding new visualizations THEN they should follow unified module structure and API patterns
- IF integrating with experiments THEN visualization should support both programmatic and CLI usage
- WHEN maintaining code THEN visualization modules should have comprehensive documentation and examples

## Non-Functional Requirements

### Non-Functional Requirement 1: Performance
- Comparison method experiment execution time must not exceed 120% of original implementation
- Batch experiments must support parallel execution capabilities
- Memory usage optimization to support large-scale datasets (>100GB)
- System should handle concurrent experiment execution without performance degradation

### Non-Functional Requirement 2: Compatibility
- Must be fully compatible with existing HSE contrastive learning implementation
- Must support all PHM-Vibench datasets (30+ industrial datasets)
- Must maintain backward compatibility with existing configuration file formats
- Must integrate seamlessly with existing PyTorch Lightning 2.0+ training infrastructure

### Non-Functional Requirement 3: Maintainability
- All SOTA methods must follow unified code standards and documentation patterns
- Complete documentation coverage with API documentation and usage examples
- Unit test coverage of core functionality (minimum 80% coverage)
- Code must pass linting and static analysis checks

### Non-Functional Requirement 4: Reproducibility
- Fixed random seeds must ensure consistent results across runs
- Detailed logging of experimental environment and parameters
- Complete dependency version information and environment specifications
- Experiment results must be reproducible within 0.1% variance across identical hardware

## Technical Constraints

### Architecture Constraints
- Must follow PHM-Vibench factory design patterns for all new components
- Must use PyTorch Lightning framework for training orchestration
- Must maintain interface compatibility with existing modules (data_factory, model_factory, trainer_factory)
- Must support the unified configuration system using ConfigWrapper

### Data Constraints
- Must use specified data path: "/home/user/data/PHMbenchdata/PHM-Vibench"
- Must use metadata file: "metadata_6_1.xlsx"
- Must support H5 format data files and existing data preprocessing pipelines
- Must handle datasets with varying sampling rates and sensor configurations

### Configuration Constraints
- Must use YAML format configuration files following existing patterns
- Must support hierarchical configuration override capabilities
- Must implement configuration validation and comprehensive error handling
- Must be compatible with existing config loading utilities

## Acceptance Criteria

### Global Acceptance Criteria
1. **Functional Completeness**: All 6 SOTA contrastive learning losses successfully integrated into Components/ with working implementations
2. **Experimental Executability**: All steps in paper experiment plan execute successfully with documented results  
3. **Result Consistency**: HSE contrastive learning maintains proper architecture with models from model_factory (within 1% variance)
4. **Documentation Completeness**: Complete experiment plan and usage documentation provided and validated
5. **Code Quality**: All code passes review standards and testing requirements with appropriate coverage

### Performance Benchmarks
- Single comparison method experiment execution time < 30 minutes
- Complete SOTA comparison experiment execution time < 4 hours  
- Peak memory usage < 16GB per experiment
- Parallel experiment throughput supports at least 4 concurrent runs

## Risk Assessment

### High Risk
- **Code Refactoring Complexity**: Integrating script-based code into src architecture may require substantial modifications and testing
- **Integration Dependencies**: Ensuring compatibility between different SOTA methods and existing HSE implementation

### Medium Risk  
- **Configuration Compatibility**: Standardizing configuration paths may impact existing experiments requiring migration
- **Performance Regression**: Architecture reorganization may affect execution performance requiring optimization
- **Testing Coverage**: Comprehensive testing of integrated methods may require significant effort

### Low Risk
- **Documentation Updates**: Experiment plan documentation updates are manageable and well-defined
- **Visualization Migration**: Moving plot files follows clear architectural patterns

## Success Metrics

### Technical Metrics
- 6 SOTA contrastive learning losses 100% successfully integrated with passing tests
- HSE contrastive architecture 100% refactored to follow PHM-Vibench patterns
- Configuration file paths 100% updated and validated
- Visualization code 100% reorganized following factory patterns
- All integration tests passing with >95% success rate

### Quality Metrics
- Code test coverage >80% for all new and modified components
- Documentation completeness check passing with comprehensive API coverage
- Performance benchmarks met or exceeded compared to baseline
- Static analysis and linting checks passing for all code

### User Experience Metrics  
- Experiment execution success rate >95% across different environments
- Result consistency validation passing with documented variance analysis
- Documentation clarity rating >4/5 from independent review
- Setup time reduction of >50% compared to manual script execution

## Dependencies and Assumptions

### Dependencies
- PyTorch 2.6.0+ with CUDA support
- PyTorch Lightning for training orchestration
- PHM-Vibench core modules (data_factory, model_factory, trainer_factory)
- Access to specified datasets and metadata files
- Adequate computing resources (16GB+ RAM, GPU support)

### Assumptions
- Existing HSE contrastive learning implementation is functionally correct and stable
- PHM-Vibench framework APIs remain stable during implementation
- Dataset paths and metadata files are accessible and correctly formatted
- Development environment has required dependencies and permissions