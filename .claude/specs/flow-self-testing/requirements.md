# Flow Self-Testing Module Requirements

**Feature Name**: flow-self-testing  
**Requirements Version**: 1.0  
**Creation Date**: 2025-09-02  
**Status**: Draft

## Introduction

This specification defines the requirements for adding comprehensive `if __name__ == "__main__"` self-testing capabilities to the existing Flow pretraining modules in PHM-Vibench. The goal is to provide immediate validation and testing capabilities for each module without requiring external test framework setup.

## Alignment with Product Vision

This specification aligns with PHM-Vibench's core mission of providing a comprehensive benchmark platform for industrial equipment vibration signal analysis by:

- **Quality Assurance**: Ensures each Flow module can be independently validated, supporting the platform's reliability requirements
- **Developer Experience**: Provides immediate feedback during development and debugging, accelerating research workflows
- **Research Enablement**: Facilitates rapid experimentation and validation for academic research, directly supporting the platform's research mission
- **Maintainability**: Establishes consistent testing patterns across all Flow modules, following existing PHM-Vibench architectural principles

The self-testing capabilities directly support the platform's factory design pattern, modular architecture, and configuration-driven experimental approach by providing immediate validation of each component's functionality.

## Requirements

### Requirement 1: Core Self-Testing Infrastructure

**User Story**: As a developer working with Flow modules, I want each module to have comprehensive self-testing capabilities so that I can immediately validate functionality without external setup.

**Acceptance Criteria**:
- **WHEN** a Flow module is executed with `python module_name.py`
- **THEN** it SHALL run a complete suite of self-tests
- **AND** it SHALL provide clear pass/fail status with informative output  
- **AND** it SHALL complete testing within 30 seconds for rapid development feedback
- **AND** it SHALL clean up resources (GPU memory, temporary files) after testing
- **AND** it SHALL integrate with existing pytest markers (@pytest.mark.slow, @pytest.mark.gpu) for CI/CD compatibility

### Requirement 2: FlowPretrainTask Self-Testing

**User Story**: As a researcher using FlowPretrainTask, I want to validate the task configuration and basic functionality so that I can ensure proper integration before running full experiments.

**Acceptance Criteria**:
- **WHEN** `python flow_pretrain.py` is executed
- **THEN** it SHALL test task instantiation with mock configurations leveraging existing conftest.py patterns
- **AND** it SHALL validate forward pass with synthetic data using ModelTestHelper utilities
- **AND** it SHALL test both pure Flow and joint Flow-Contrastive training modes
- **AND** it SHALL verify generation capabilities (conditional/unconditional)
- **AND** it SHALL validate metrics tracking and logging functionality
- **AND** it SHALL test configuration parameter validation and error handling

### Requirement 3: FlowContrastiveLoss Self-Testing

**User Story**: As a developer working with loss functions, I want to validate FlowContrastiveLoss computation accuracy so that I can ensure correct gradient computation and loss weighting.

**Acceptance Criteria**:
- **WHEN** `python flow_contrastive_loss.py` is executed
- **THEN** it SHALL test loss computation with known input/output pairs
- **AND** it SHALL validate gradient flow through both Flow and contrastive components
- **AND** it SHALL test different loss weight combinations (λ_flow, λ_contrastive)
- **AND** it SHALL verify gradient balancing mechanism functionality
- **AND** it SHALL test time-series augmentation effects on loss computation
- **AND** it SHALL validate projection head initialization and functionality

**Priority**: High  
**Risk Level**: Medium - Gradient computation validation complexity

### Requirement 4: FlowMetrics Self-Testing

**User Story**: As a researcher evaluating Flow model quality, I want to validate FlowMetrics calculations so that I can trust the generated quality assessments and performance measurements.

**Acceptance Criteria**:
- **WHEN** `python flow_metrics.py` is executed  
- **THEN** it SHALL test all quality metrics with synthetic ground truth data
- **AND** it SHALL validate statistical tests (KS test, spectral similarity) with known distributions
- **AND** it SHALL test performance monitoring (speed, memory, gradient tracking)
- **AND** it SHALL verify visualization generation without errors
- **AND** it SHALL test metrics persistence (save/load functionality)
- **AND** it SHALL validate convergence analysis with mock training histories

### Requirement 5: Integration Testing Capabilities

**User Story**: As a system integrator, I want to test Flow module interactions so that I can validate end-to-end functionality before deployment.

**Acceptance Criteria**:
- **WHEN** any Flow module self-test is executed
- **THEN** it SHALL test integration with PHM-Vibench factory registration patterns
- **AND** it SHALL validate configuration compatibility with existing YAML templates
- **AND** it SHALL test interoperability between Flow modules using existing registration system
- **AND** it SHALL verify PyTorch Lightning integration points
- **AND** it SHALL validate device compatibility (CPU/GPU switching) following existing patterns

### Requirement 6: Performance Validation

**User Story**: As a performance-conscious developer, I want to validate Flow module performance characteristics so that I can ensure they meet efficiency requirements.

**Acceptance Criteria**:
- **WHEN** self-tests include performance validation
- **THEN** it SHALL measure execution time using consistent methodologies from existing codebase
- **AND** it SHALL monitor memory usage patterns and detect potential leaks
- **AND** it SHALL validate GPU utilization when available using torch.cuda utilities
- **AND** it SHALL test with different batch sizes to identify scaling behavior
- **AND** it SHALL provide relative performance metrics accounting for hardware variations

### Requirement 7: Error Handling and Edge Cases

**User Story**: As a robust system developer, I want comprehensive error handling validation so that Flow modules gracefully handle edge cases and provide meaningful error messages.

**Acceptance Criteria**:
- **WHEN** self-tests encounter error conditions
- **THEN** it SHALL test graceful handling of invalid configurations
- **AND** it SHALL validate appropriate error messages for common misconfigurations
- **AND** it SHALL test edge cases (empty data, extreme parameter values)
- **AND** it SHALL verify resource cleanup in error scenarios
- **AND** it SHALL test recovery mechanisms for transient failures

### Requirement 8: Mock Data Generation and Test Infrastructure

**User Story**: As a developer creating self-tests, I want reusable mock data generation and testing utilities so that I can efficiently create comprehensive test coverage.

**Acceptance Criteria**:
- **WHEN** self-tests require synthetic data
- **THEN** it SHALL leverage existing conftest.py fixture patterns for data generation
- **AND** it SHALL create realistic synthetic vibration signals matching industrial patterns
- **AND** it SHALL provide configurable mock configurations using existing args_* parameter structures  
- **AND** it SHALL implement ModelTestHelper integration for parameter counting and shape validation
- **AND** it SHALL support both CPU and GPU mock environments with proper device management
- **AND** it SHALL maintain consistency with existing 149+ self-testing implementations in the codebase

## Priority and Risk Assessment

### Requirements Priority Matrix
- **High Priority**: Requirements 1-4 (Core infrastructure and individual module testing)
- **Medium Priority**: Requirements 5-7 (Integration, performance, and error handling)  
- **Low Priority**: Requirement 8 (Infrastructure and utilities)

### Risk Analysis by Requirement
- **Low Risk**: Requirements 4, 7, 8 (Well-defined statistical validation, error scenarios, straightforward utilities)
- **Medium Risk**: Requirements 1, 2, 3, 6 (Integration complexity, gradient computation validation, performance measurement accuracy)
- **High Risk**: Requirement 5 (Cross-module integration complexity)

## Non-Functional Requirements

### NFR-1: Performance
- Self-tests SHALL complete within 30 seconds for rapid development feedback
- Memory usage SHALL not exceed 2GB during testing
- GPU memory SHALL be properly cleaned up after each test

### NFR-2: Maintainability  
- Self-testing code SHALL follow existing PHM-Vibench patterns and conventions
- Test implementations SHALL be easily extendable for new functionality
- Mock data generation SHALL be reusable across modules

### NFR-3: Reproducibility
- All tests SHALL use fixed random seeds for consistent results
- Test outputs SHALL be deterministic across different environments
- Performance measurements SHALL account for hardware variations

### NFR-4: Compatibility
- Self-tests SHALL work in both CPU and GPU environments
- Testing SHALL not require external dependencies beyond existing project requirements
- Self-tests SHALL be compatible with existing CI/CD workflows

## Success Metrics

### Development Efficiency
- **Measurement**: Time to validate module functionality after code changes
- **Target**: < 30 seconds per module
- **Baseline**: Current manual testing approach

### Code Quality
- **Measurement**: Number of issues caught by self-tests vs. integration testing
- **Target**: 80% of module-level issues caught by self-tests
- **Tracking**: Issue categorization and resolution tracking

### Developer Adoption
- **Measurement**: Frequency of self-test execution during development
- **Target**: Self-tests run before every commit involving Flow modules
- **Tracking**: Developer workflow surveys and usage analytics

## Risk Analysis

### Technical Risks
- **Complex Mock Data**: Generating realistic synthetic data for testing
  - **Mitigation**: Leverage existing test fixtures and patterns from conftest.py
  - **Impact**: Medium
  
- **GPU Resource Management**: Proper cleanup in testing scenarios  
  - **Mitigation**: Follow existing CUDA memory management patterns
  - **Impact**: Low

### Integration Risks
- **Module Interdependencies**: Testing modules that depend on each other
  - **Mitigation**: Implement proper mocking and isolation strategies
  - **Impact**: Medium

- **Performance Variation**: Inconsistent performance measurements across environments
  - **Mitigation**: Relative performance metrics and environment detection
  - **Impact**: Low

## Assumptions and Constraints

### Assumptions
- Existing PHM-Vibench testing infrastructure patterns will be maintained
- PyTorch and PyTorch Lightning APIs will remain stable
- CUDA availability is optional but beneficial for comprehensive testing

### Constraints
- Must not introduce new external dependencies
- Self-tests must be isolated and not interfere with existing test suites
- Implementation must maintain compatibility with existing module interfaces

## Dependencies

### Internal Dependencies
- Existing PHM-Vibench factory pattern infrastructure
- Current Flow module implementations (FlowPretrainTask, FlowContrastiveLoss, FlowMetrics)
- Test utilities from conftest.py and ModelTestHelper

### External Dependencies
- PyTorch framework (existing)
- PyTorch Lightning (existing)
- NumPy and SciPy for statistical validations (existing)

## Definition of Done

- [ ] All three Flow modules have comprehensive `if __name__ == "__main__"` self-testing
- [ ] Self-tests cover unit, integration, and performance validation scenarios
- [ ] All tests pass consistently across CPU and GPU environments  
- [ ] Documentation includes self-testing usage instructions
- [ ] Self-tests execute within performance targets (< 30 seconds)
- [ ] Error handling and edge cases are thoroughly tested
- [ ] Integration with existing PHM-Vibench patterns is validated