# Flow Self-Testing Implementation Tasks

## Task Overview

This implementation plan creates comprehensive `if __name__ == "__main__"` self-testing capabilities for three PHM-Vibench Flow modules: FlowPretrainTask, FlowContrastiveLoss, and FlowMetrics. The approach leverages existing testing infrastructure patterns from conftest.py and ModelTestHelper while ensuring atomic, agent-friendly tasks that execute within 30 seconds.

## Steering Document Compliance

This implementation follows PHM-Vibench architectural conventions:
- **Factory Pattern Integration**: Validates proper registration with task factory systems
- **Configuration-Driven Architecture**: Uses existing YAML configuration patterns and argparse namespace structures  
- **PyTorch Lightning Compatibility**: Maintains full compatibility with Lightning training workflows
- **Resource Management**: Implements proper CUDA memory cleanup following existing patterns
- **Testing Standards**: Leverages existing conftest.py fixtures, ModelTestHelper utilities, and pytest markers

## Atomic Task Requirements

**Each task must meet these criteria for optimal agent execution:**
- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Must specify exact files to create/modify
- **Agent-Friendly**: Clear input/output with minimal context switching

## Task Format Guidelines

- Use checkbox format: `- [ ] Task number. Task description`
- **Specify files**: Always include exact file paths to create/modify
- **Include implementation details** as bullet points
- Reference requirements using: `_Requirements: X.Y, Z.A_`
- Reference existing code to leverage using: `_Leverage: path/to/file.py, path/to/component.py_`
- Focus only on coding tasks (no deployment, user testing, etc.)
- **Avoid broad terms**: No "system", "integration", "complete" in task titles

## Tasks

### Core Infrastructure Tasks

- [ ] 1. Create self-testing infrastructure base classes in src/task_factory/task/pretrain/self_testing/__init__.py
  - File: src/task_factory/task/pretrain/self_testing/__init__.py
  - Define ValidationResult, TestConfiguration, and PerformanceMetrics dataclasses
  - Implement SelfTestOrchestrator base class with timeout management
  - Add ResourceManager class for proper cleanup following existing patterns
  - Purpose: Establish shared testing infrastructure for all Flow modules
  - _Leverage: test/conftest.py ModelTestHelper patterns, existing device management_
  - _Requirements: 1.1, 8.1_

- [ ] 2. Implement mock data generator in src/task_factory/task/pretrain/self_testing/generators.py
  - File: src/task_factory/task/pretrain/self_testing/generators.py
  - Create FlowMockDataGenerator class with realistic vibration signal generation
  - Implement generate_flow_batch() method using existing synthetic_dataset patterns
  - Add generate_file_ids() for conditional training mock data
  - Purpose: Provide reusable synthetic data for Flow module testing
  - _Leverage: test/conftest.py synthetic_dataset fixture, sample_classification_data_
  - _Requirements: 8.2, 8.3_

- [ ] 3. Create configuration factory in src/task_factory/task/pretrain/self_testing/config_factory.py
  - File: src/task_factory/task/pretrain/self_testing/config_factory.py
  - Implement FlowConfigurationFactory class using existing Namespace patterns
  - Add create_flow_task_config(), create_model_config(), create_trainer_config() methods
  - Use existing basic_model_configs fixture patterns for consistency
  - Purpose: Generate mock configurations compatible with PHM-Vibench patterns
  - _Leverage: test/conftest.py basic_model_configs, argparse Namespace creation_
  - _Requirements: 8.4, 5.2_

### FlowPretrainTask Self-Testing Tasks

- [ ] 4. Create FlowPretrainTask validator in src/task_factory/task/pretrain/self_testing/validators.py
  - File: src/task_factory/task/pretrain/self_testing/validators.py
  - Implement FlowPretrainTaskValidator class with test_instantiation() method
  - Add test_forward_pass() using ModelTestHelper.test_forward_backward patterns
  - Implement test_training_modes() for Flow-only and joint Flow-Contrastive modes
  - Purpose: Validate FlowPretrainTask core functionality with synthetic data
  - _Leverage: test/conftest.py ModelTestHelper.test_forward_backward, device fixture_
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 5. Add generation capability tests to FlowPretrainTaskValidator
  - File: src/task_factory/task/pretrain/self_testing/validators.py (extend from task 4)
  - Implement test_generation_capabilities() for conditional/unconditional generation
  - Add test_metrics_tracking() for logging functionality validation
  - Implement test_configuration_validation() with error handling scenarios
  - Purpose: Complete FlowPretrainTask validation with generation and metrics testing
  - _Leverage: existing generation methods in flow_pretrain.py, metrics integration_
  - _Requirements: 2.4, 2.5, 2.6_

- [ ] 6. Implement FlowPretrainTask main self-test entry point in flow_pretrain.py
  - File: src/task_factory/task/pretrain/flow_pretrain.py (add to existing file)
  - Add if __name__ == "__main__" block with 30-second timeout management
  - Integrate SelfTestOrchestrator with FlowPretrainTaskValidator
  - Implement comprehensive results reporting with Chinese language output
  - Purpose: Provide immediate self-testing capability for FlowPretrainTask
  - _Leverage: self_testing infrastructure classes, existing module structure_
  - _Requirements: 1.1, 2.1_

### FlowContrastiveLoss Self-Testing Tasks

- [ ] 7. Create FlowContrastiveLoss validator in self_testing/validators.py
  - File: src/task_factory/task/pretrain/self_testing/validators.py (extend from task 4)
  - Implement FlowContrastiveLossValidator class with test_loss_computation() method
  - Add test_gradient_flow() using ModelTestHelper gradient checking patterns
  - Implement test_weight_balancing() for different λ_flow and λ_contrastive combinations
  - Purpose: Validate FlowContrastiveLoss computation accuracy and gradient flow
  - _Leverage: ModelTestHelper gradient validation, existing loss computation logic_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 8. Add projection head and augmentation tests to FlowContrastiveLossValidator
  - File: src/task_factory/task/pretrain/self_testing/validators.py (extend from task 7)
  - Implement test_projection_head() for initialization and forward pass validation
  - Add test_augmentation_effects() for time-series augmentation impact on loss
  - Implement test_gradient_balancing() mechanism functionality
  - Purpose: Complete FlowContrastiveLoss validation with projection head and augmentation testing
  - _Leverage: ProjectionHead class, TimeSeriesAugmentation from ContrastiveSSL_
  - _Requirements: 3.4, 3.5, 3.6_

- [ ] 9. Implement FlowContrastiveLoss main self-test entry point in flow_contrastive_loss.py
  - File: src/task_factory/task/pretrain/flow_contrastive_loss.py (add to existing file)
  - Add if __name__ == "__main__" block with SelfTestOrchestrator integration
  - Implement FlowContrastiveLossValidator execution with results reporting
  - Add proper resource cleanup and timeout handling
  - Purpose: Provide immediate self-testing capability for FlowContrastiveLoss
  - _Leverage: self_testing infrastructure, existing FlowContrastiveLoss class_
  - _Requirements: 1.1, 3.1_

### FlowMetrics Self-Testing Tasks

- [ ] 10. Create FlowMetrics validator in self_testing/validators.py
  - File: src/task_factory/task/pretrain/self_testing/validators.py (extend from tasks 4,7)
  - Implement FlowMetricsValidator class with test_quality_metrics() method
  - Add test_statistical_validation() for KS test and spectral similarity with known distributions
  - Implement test_performance_monitoring() for speed, memory, gradient tracking
  - Purpose: Validate FlowMetrics calculation accuracy with synthetic ground truth
  - _Leverage: existing statistical methods, SciPy stats functions_
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 11. Add visualization and persistence tests to FlowMetricsValidator
  - File: src/task_factory/task/pretrain/self_testing/validators.py (extend from task 10)
  - Implement test_visualization_generation() for plot generation without errors
  - Add test_metrics_persistence() for save/load functionality validation
  - Implement test_convergence_analysis() with mock training histories
  - Purpose: Complete FlowMetrics validation with visualization and persistence testing
  - _Leverage: matplotlib plotting functions, existing save/load methods_
  - _Requirements: 4.4, 4.5, 4.6_

- [ ] 12. Implement FlowMetrics main self-test entry point in flow_metrics.py
  - File: src/task_factory/task/pretrain/flow_metrics.py (add to existing file)
  - Add if __name__ == "__main__" block with FlowMetricsValidator integration
  - Implement comprehensive metrics testing with results reporting
  - Add proper matplotlib backend handling for non-GUI environments
  - Purpose: Provide immediate self-testing capability for FlowMetrics
  - _Leverage: self_testing infrastructure, existing FlowMetrics class_
  - _Requirements: 1.1, 4.1_

### Integration and Performance Tasks

- [ ] 13. Create integration test runner in self_testing/integration.py
  - File: src/task_factory/task/pretrain/self_testing/integration.py
  - Implement FlowIntegrationTester class for cross-module communication testing
  - Add test_factory_registration() for PHM-Vibench factory pattern validation
  - Implement test_configuration_compatibility() with existing YAML templates
  - Purpose: Validate Flow module interactions and factory integration
  - _Leverage: existing factory registration patterns, configuration loading_
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 14. Add PyTorch Lightning integration tests to integration.py
  - File: src/task_factory/task/pretrain/self_testing/integration.py (extend from task 13)
  - Implement test_lightning_integration() for trainer compatibility
  - Add test_device_compatibility() for CPU/GPU switching following existing patterns
  - Implement test_interoperability() between Flow modules using registration system
  - Purpose: Complete integration testing with PyTorch Lightning and device management
  - _Leverage: existing device management patterns, PyTorch Lightning utilities_
  - _Requirements: 5.4, 5.5_

- [ ] 15. Create performance benchmarking in self_testing/performance.py
  - File: src/task_factory/task/pretrain/self_testing/performance.py
  - Implement PerformanceBenchmarker class with execution time monitoring
  - Add memory usage tracking and GPU utilization assessment methods
  - Implement batch size scaling behavior tests with relative performance metrics
  - Purpose: Validate Flow module performance characteristics within requirements
  - _Leverage: existing performance monitoring patterns, torch.cuda utilities_
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

### Error Handling and Edge Case Tasks

- [ ] 16. Create error handling validator in self_testing/error_validator.py
  - File: src/task_factory/task/pretrain/self_testing/error_validator.py
  - Implement ErrorHandlingValidator class with test_invalid_configurations() method
  - Add test_edge_cases() for empty data and extreme parameter values
  - Implement test_resource_cleanup() in error scenarios
  - Purpose: Validate comprehensive error handling and edge case management
  - _Leverage: existing error handling patterns, resource cleanup mechanisms_
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 17. Add recovery mechanism tests to error_validator.py
  - File: src/task_factory/task/pretrain/self_testing/error_validator.py (extend from task 16)
  - Implement test_recovery_mechanisms() for transient failure handling
  - Add test_meaningful_error_messages() for common misconfiguration scenarios
  - Implement test_graceful_degradation() with reduced functionality modes
  - Purpose: Complete error handling validation with recovery and graceful degradation
  - _Leverage: existing error recovery patterns, fallback mechanisms_
  - _Requirements: 7.4, 7.5_

### Documentation and Final Integration Tasks

- [ ] 18. Create comprehensive test documentation in self_testing/README.md
  - File: src/task_factory/task/pretrain/self_testing/README.md
  - Document self-testing usage instructions for each Flow module
  - Include examples of running individual module self-tests
  - Add troubleshooting guide for common issues and solutions
  - Purpose: Provide clear documentation for self-testing capabilities
  - _Leverage: existing documentation patterns, module usage examples_
  - _Requirements: Definition of Done documentation requirement_

- [ ] 19. Implement pytest integration wrapper in test/test_flow_self_testing.py
  - File: test/test_flow_self_testing.py
  - Create pytest wrappers for each Flow module self-test with proper markers
  - Add @pytest.mark.slow and @pytest.mark.gpu markers for CI/CD integration
  - Implement pytest_collection_modifyitems integration for automatic marker assignment
  - Purpose: Enable Flow self-tests to run within existing CI/CD workflows
  - _Leverage: test/conftest.py pytest markers, existing integration patterns_
  - _Requirements: 1.6, CI/CD compatibility_

- [ ] 20. Create comprehensive validation script in validate_flow_self_tests.py
  - File: src/task_factory/task/pretrain/validate_flow_self_tests.py
  - Implement comprehensive validation runner for all three Flow modules
  - Add parallel execution with timeout management for rapid validation
  - Include summary reporting with pass/fail statistics and performance metrics
  - Purpose: Provide single-command validation of all Flow module self-testing capabilities
  - _Leverage: self_testing infrastructure, existing validation patterns_
  - _Requirements: Definition of Done validation requirement_

## Implementation Guidelines

### File Structure
```
src/task_factory/task/pretrain/
├── self_testing/
│   ├── __init__.py           # Core infrastructure (Task 1)
│   ├── generators.py         # Mock data generation (Task 2)
│   ├── config_factory.py     # Configuration factory (Task 3)
│   ├── validators.py         # Module validators (Tasks 4,5,7,8,10,11)
│   ├── integration.py        # Integration testing (Tasks 13,14)
│   ├── performance.py        # Performance benchmarking (Task 15)
│   ├── error_validator.py    # Error handling (Tasks 16,17)
│   └── README.md            # Documentation (Task 18)
├── flow_pretrain.py         # Add self-test entry point (Task 6)
├── flow_contrastive_loss.py # Add self-test entry point (Task 9)
├── flow_metrics.py          # Add self-test entry point (Task 12)
└── validate_flow_self_tests.py # Comprehensive validator (Task 20)
```

### Validation Criteria

Each task must produce:
- **Testable Output**: Code that can be executed and validated
- **Error Handling**: Proper exception handling and resource cleanup
- **Documentation**: Inline comments explaining implementation decisions
- **Integration**: Compatibility with existing PHM-Vibench patterns
- **Performance**: Execution within 30-second timeout requirements

### Dependencies Between Tasks

- **Sequential Dependencies**: Tasks 1-3 must complete before tasks 4-17
- **Module Dependencies**: Tasks 4-6 (FlowPretrainTask), 7-9 (FlowContrastiveLoss), 10-12 (FlowMetrics) can be completed in parallel
- **Integration Dependencies**: Tasks 13-17 require completion of module-specific validators
- **Final Tasks**: Tasks 18-20 require completion of all implementation tasks

This task breakdown ensures atomic, implementable units that leverage existing PHM-Vibench infrastructure while providing comprehensive self-testing capabilities for all Flow modules within the 30-second execution requirement.