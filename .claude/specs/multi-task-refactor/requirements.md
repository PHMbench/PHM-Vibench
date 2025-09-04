# Multi-Task Refactor Requirements

## Introduction

This specification defines the requirements for refactoring PHM-Vibench's multi-task learning module from its current standalone implementation (`multi_task_lightning.py`) into the standardized task factory architecture. The refactor will reorganize the module under the `In_distribution` task category and ensure it inherits from `Default_task` to leverage existing infrastructure and maintain consistency with other task implementations.

**User Value**: This refactoring provides developers and researchers with a unified, maintainable approach to multi-task learning that follows established patterns, reduces code duplication, and ensures seamless integration with the PHM-Vibench pipeline system.

## Alignment with Product Vision

This refactoring directly supports PHM-Vibench's core architectural principles:
- **Modular Factory Design**: Ensures multi-task learning follows the same factory patterns as all other components
- **Code Reusability**: Leverages `Default_task` infrastructure to eliminate duplicate optimizer, scheduler, and logging code
- **Consistency**: Maintains uniform task interfaces across classification, domain generalization, few-shot learning, and multi-task scenarios
- **Extensibility**: Positions multi-task learning within the standard framework for future enhancements

## Background and Context

### Current Problems
1. **Non-standard Module Location**: `multi_task_lightning.py` is located in the task_factory root directory, violating the organizational structure used by other tasks
2. **Missing Unified Interface**: MultiTaskLightningModule directly inherits from pl.LightningModule, bypassing Default_task's infrastructure
3. **Factory Loading Failure**: Missing 'task' export variable prevents task_factory from correctly loading the module
4. **Code Duplication**: Reimplements optimizer configuration, logging, and other functionality already provided by Default_task

### Solution Objectives
- Standardize module organization following established task categorization patterns
- Inherit from Default_task to reuse infrastructure and reduce code duplication
- Ensure task_factory can correctly load and instantiate the multi-task module
- No backward compatibility required - direct replacement of existing implementation

## Functional Requirements

### Requirement 1: Module Structure Reorganization
**User Story**: As a developer, I want the multi-task module to follow standard task organization structure, so that it's easy to maintain and extend

**Acceptance Criteria**:
- WHEN multi-task configuration specifies type="In_distribution", name="multi_task_phm"
- THEN task_factory correctly resolves the path src.task_factory.task.In_distribution.multi_task_phm
- AND successfully instantiates the task module within 5 seconds
- AND IF module loading fails THEN raises ImportError with specific path information

### Requirement 2: Default_task Infrastructure Inheritance
**User Story**: As a developer, I want the multi-task module to reuse Default_task's infrastructure, so that I avoid code duplication and maintain consistency

**Acceptance Criteria**:
- WHEN MultiTaskPHM inherits from Default_task
- THEN automatically inherits optimizer configuration, learning rate scheduling, and logging functionality
- AND only needs to override multi-task specific methods (training_step, validation_step, etc.)
- AND training throughput remains >= 95% of current implementation performance

### Requirement 3: Multi-Task Training Support
**User Story**: As a researcher, I want to simultaneously train classification, anomaly detection, signal prediction, and RUL prediction tasks, so that I can develop comprehensive foundation models

**Acceptance Criteria**:
- WHEN configuration contains multiple tasks (classification, anomaly_detection, signal_prediction, rul_prediction)
- THEN module computes individual loss and metrics for each enabled task
- AND supports configurable task weights for loss balancing with weights summing to reasonable ranges
- AND correctly handles different label formats and output dimensions for each task type
- AND logs separate metrics for each task (e.g., classification_acc, rul_mae, anomaly_f1)

### Requirement 4: Task Registration and Discovery
**User Story**: As the PHM-Vibench system, I need to automatically discover and register the multi-task module, so that it integrates seamlessly with the factory pattern

**Acceptance Criteria**:
- WHEN using @register_task("In_distribution", "multi_task_phm") decorator
- THEN task module automatically registers to TASK_REGISTRY
- AND task_factory can quickly lookup the task class without filesystem imports
- AND registration occurs at module import time without additional initialization

### Requirement 5: Pipeline Integration
**User Story**: As a data scientist, I want multi-task models to integrate with existing Pipeline_03_multitask workflows, so that I can leverage pretrained foundation models without workflow changes

**Acceptance Criteria**:
- WHEN Pipeline_03_multitask_pretrain_finetune loads multi-task configuration
- THEN new implementation processes batches in identical format to original
- AND supports all existing multi-task configuration parameters
- AND maintains compatibility with ISFM foundation model loading
- AND preserves wandb logging integration for multi-task metrics

## Technical Requirements

### Technical Requirement 1: File Organization Structure
**Specification**:
```
src/task_factory/
├── task/
│   └── In_distribution/
│       └── multi_task_phm.py  # New multi-task implementation
└── multi_task_lightning.py     # DELETE this file
```

**Validation**: Directory structure must match exactly, with multi_task_lightning.py completely removed

### Technical Requirement 2: Class Inheritance Hierarchy
**Specification**:
```python
from ...Default_task import Default_task

@register_task("In_distribution", "multi_task_phm")
class MultiTaskPHM(Default_task):
    """Multi-task PHM implementation inheriting Default_task infrastructure"""
    # Override multi-task specific methods only
```

**Validation**: Class must inherit Default_task, not pl.LightningModule directly

### Technical Requirement 3: Configuration Format Compatibility
**Specification**:
```yaml
task:
  type: "In_distribution"
  name: "multi_task_phm"
  enabled_tasks: ["classification", "anomaly_detection", "signal_prediction", "rul_prediction"]
  task_weights:
    classification: 1.0
    anomaly_detection: 0.6
    signal_prediction: 0.7
    rul_prediction: 0.8
```

**Validation**: Configuration parser must accept all existing multi-task parameters without modification

### Technical Requirement 4: Module Export Interface
**Specification**:
```python
# At end of multi_task_phm.py file
task = MultiTaskPHM
```

**Validation**: task_factory must be able to import and instantiate using `task_module.task`

## 非功能需求

### NR1: 性能要求
- 多任务训练性能不低于原实现
- 内存使用保持在相同水平

### NR2: 可维护性
- 代码结构清晰，职责明确
- 遵循项目既有的编码规范
- 充分利用 Default_task 的基础设施

### NR3: 可测试性
- 支持单独测试每个任务组件
- End-to-end multi-task training test support

## Constraints

### Constraint 1: No Backward Compatibility Required
**Details**:
- Direct deletion of `multi_task_lightning.py` is acceptable
- Configuration files must be updated to new format
- Existing multi-task experiments require configuration migration
- No support for legacy configuration formats

**Impact**: Simplifies implementation by eliminating compatibility layers

### Constraint 2: Must Follow Established Patterns
**Details**:
- Inherit from Default_task, not directly from pl.LightningModule
- Use @register_task decorator for task registration
- Follow In_distribution task organizational structure
- Adhere to existing task_factory module resolution logic

**Rationale**: Maintains consistency with PHM-Vibench architectural patterns

### Constraint 3: Integration Requirements
**Details**:
- Must work with existing Pipeline_03_multitask_pretrain_finetune
- Must support ISFM foundation model loading patterns
- Must maintain wandb/swanlab logging integration
- Must support all current multi-task configuration parameters

**Validation**: End-to-end testing with existing pipeline workflows

## Success Criteria

### Quantitative Success Metrics
1. **Performance**: Multi-task training throughput >= 95% of original implementation
2. **Resource Usage**: Memory consumption <= 110% of current implementation
3. **Loading Speed**: Task factory instantiation < 2 seconds
4. **Code Quality**: <20% code duplication compared to Default_task baseline
5. **Test Coverage**: 100% pass rate for existing multi-task test scenarios

### Qualitative Success Criteria
1. **Structural Compliance**: Module follows In_distribution organization pattern exactly
2. **Integration Seamless**: Works with Pipeline_03_multitask without modification
3. **Configuration Compatible**: All existing multitask_*.yaml files work with minimal changes
4. **Code Maintainable**: Passes code review with PHM-Vibench maintainers
5. **Documentation Complete**: Inline documentation covers all multi-task specific logic

### Validation Methods
- **Automated Testing**: Run full test suite with multi-task configurations
- **Performance Benchmarking**: Compare against baseline measurements
- **Integration Testing**: End-to-end pipeline execution with foundation models
- **Code Review**: Peer review focusing on adherence to established patterns

## Risk Analysis and Mitigation

### Risk 1: Functional Regression
**Probability**: Medium | **Impact**: High
**Mitigation**: 
- Detailed comparison of original implementation functionality
- Comprehensive test suite covering all multi-task scenarios
- Side-by-side validation with original implementation
- Rollback plan if critical issues discovered

### Risk 2: Configuration Migration Complexity  
**Probability**: Low | **Impact**: Medium
**Mitigation**:
- Clear configuration migration guide with examples
- Automated configuration validation
- Gradual migration path for existing experiments

### Risk 3: Default_task Insufficient Functionality
**Probability**: Low | **Impact**: Medium  
**Mitigation**:
- Thorough analysis of Default_task capabilities vs multi-task requirements
- Extension through inheritance and method overriding
- Fallback to selective code duplication if absolutely necessary

### Risk 4: Performance Degradation
**Probability**: Low | **Impact**: High
**Mitigation**:
- Performance benchmarking throughout development
- Profiling to identify bottlenecks
- Optimization of critical paths if needed