# HSE Industrial Contrastive Learning Implementation Plan

## Task Overview

This plan implements a Prompt-guided Contrastive Learning system for industrial vibration analysis, focusing on System Information Prompt encoding combined with SOTA contrastive learning methods. The implementation creates two-level prompts (System+Sample, NO fault-level since Label is prediction target), independent M_02_ISFM_Prompt model, two-stage training, and complete self-testing infrastructure.

Key Innovation: First-ever combination of system metadata as learnable prompts with contrastive learning for cross-system fault diagnosis generalization, integrated with Pipeline_03_multitask_pretrain_finetune.py workflow.

## Steering Document Compliance

All tasks follow PHM-Vibench factory patterns with strict component registration and self-testing requirements. The implementation creates completely isolated M_02_ISFM_Prompt and E_01_HSE_v2 models to avoid any conflicts with existing code. Integration with Pipeline_03's MultiTaskPretrainFinetunePipeline ensures reuse of mature training infrastructure. Each component includes `if __name__ == '__main__':` self-test sections.

## Atomic Task Requirements

**Each task meets optimal agent execution criteria:**

- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes by experienced developer
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Exact file paths specified for create/modify operations
- **Agent-Friendly**: Clear inputs/outputs with minimal context switching

## Task Format Guidelines

- Use checkbox format: `- [ ] Task number. Task description`
- **Specify files**: Always include exact file paths to create/modify
- **Include implementation details** as bullet points under each task
- Reference requirements using: `_Requirements: FR1, FR2_`
- Reference existing code using: `_Leverage: path/to/existing_file.py_`
- Focus only on coding tasks (no deployment or user testing)
- **Avoid broad terms**: No "system", "integration", "complete" in task titles

## Tasks

### P0 Core Functionality (Must implement first)

- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/components/SystemPromptEncoder.py
  - Implement two-level prompt encoding: Dataset_id+Domain_id (system), Sample_rate (sample)
  - Use embedding tables for categorical features, linear projection for numerical features
  - Add multi-head self-attention for level fusion with final aggregation
  - Include comprehensive self-test with metadata dict creation utilities
  - **Critical**: NO fault-level prompts since Label is prediction target
  - _Requirements: FR2_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (for parameter patterns)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/components/PromptFusion.py
  - Implement three fusion strategies: concatenation, cross-attention, adaptive gating
  - Add residual connections for attention-based fusion to preserve original signal features
  - Include dimension validation and automatic shape matching
  - Add comprehensive self-test for all three fusion strategies with gradient flow verification
  - _Requirements: FR1, FR2_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (for attention patterns)_
- [ ] 
  - **Files**:
    - src/model_factory/ISFM_Prompt/__init__.py
    - src/model_factory/ISFM_Prompt/components/__init__.py
    - src/model_factory/ISFM_Prompt/embedding/__init__.py
  - Initialize module structure with proper imports and component registration
  - Add docstrings explaining Prompt-guided architecture innovation
  - Set up component dictionaries for factory pattern integration
  - _Requirements: FR3_
  - _Leverage: src/model_factory/ISFM/__init__.py (for factory patterns)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/embedding/E_01_HSE_v2.py
  - Create completely new HSE implementation with SystemPromptEncoder and PromptFusion integration
  - DO NOT inherit from or modify existing E_01_HSE.py - this is a fresh implementation
  - Add metadata parameter to forward() method for two-level prompt processing (System+Sample only)
  - Implement training_stage control with prompt freezing for Pipeline_03 integration
  - Include fallback to signal-only processing when metadata unavailable
  - Add complete self-test with metadata validation and stage switching
  - **Critical**: Ensure zero dependencies on existing E_01_HSE.py
  - _Requirements: FR2, FR3_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (for reference only, not inheritance)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/M_02_ISFM_Prompt.py
  - Implement complete ISFM model with Prompt-guided embedding support
  - Add component dictionaries for PromptEmbedding, Backbone (reuse existing), TaskHead (reuse existing)
  - Include training stage control and metadata forwarding throughout model pipeline
  - Add graceful degradation when prompt features unavailable
  - Include comprehensive self-test with multiple embedding/backbone/taskhead combinations
  - _Requirements: FR1, FR3_
  - _Leverage: src/model_factory/ISFM/M_01_ISFM.py (for model structure patterns)_
- [ ] 
  - **File**: configs/pipeline_03/hse_prompt_multitask_config.yaml
  - Configure Pipeline_03 MultiTaskPretrainFinetunePipeline with HSE prompt integration
  - Set stage_1_pretraining with M_02_ISFM_Prompt and E_01_HSE_v2 embedding
  - Configure backbone comparison: ['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO']
  - Set multi-source target_systems: [1, 5, 6, 13, 19] for pretraining
  - Add HSE prompt specific task configuration with contrast_weight: 0.15
  - Configure stage_2_finetuning with prompt freezing and classification tasks
  - Include Pipeline_03 environment and evaluation configurations
  - _Requirements: FR5, FR4_
  - _Leverage: Pipeline_03 configuration patterns (create_pretraining_config, create_finetuning_config)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/test_prompt_components.py
  - Implement comprehensive component testing for SystemPromptEncoder, PromptFusion, E_01_HSE_v2
  - Test integration between components with proper metadata flow and two-level prompt processing
  - Verify Pipeline_03 integration functionality and prompt freezing behavior
  - Add performance benchmarking for latency and memory usage requirements
  - Include cross-component compatibility testing and complete model isolation validation
  - Test that E_01_HSE_v2 has zero dependencies on original E_01_HSE.py
  - _Requirements: FR2, FR3, FR5_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (for testing patterns only)_
- [ ] 
  - **File**: src/utils/pipeline_config/hse_prompt_integration.py
  - Implement HSEPromptPipelineIntegration adapter for Pipeline_03 integration
  - Add create_hse_prompt_pretraining_config() using Pipeline_03's utilities
  - Add create_hse_prompt_finetuning_config() using Pipeline_03's utilities
  - Include adapt_checkpoint_loading() for Pipeline_03 checkpoint format compatibility
  - Add parameter freezing utilities for prompt-related components during finetuning
  - Include comprehensive self-test for all Pipeline_03 integration functions
  - **Critical**: This is essential for Pipeline_03 integration - must be P0
  - _Requirements: FR5_
  - _Leverage: src/Pipeline_03_multitask_pretrain_finetune.py (for integration patterns)_
- [ ] 
  - **File**: src/task_factory/Components/prompt_contrastive.py
  - Implement universal wrapper for all 6 existing contrastive losses with prompt guidance
  - Add system-aware positive/negative sampling using metadata system_ids
  - Include prompt similarity loss to encourage system-invariant representations
  - Add comprehensive self-test for all LOSS_MAPPING combinations with prompt features
  - Support graceful fallback to standard contrastive learning when prompts unavailable
  - **Critical**: Core functionality for prompt-guided contrastive learning - must be P0
  - _Requirements: FR1_
  - _Leverage: src/task_factory/Components/contrastive_losses.py (for all existing losses)_
- [ ] 
  - **File**: src/task_factory/task/CDDG/hse_contrastive.py
  - Implement task class integrating prompt-guided contrastive learning
  - Add metadata preprocessing for system information extraction
  - Include loss combination logic: classification + prompt-guided contrastive
  - Add comprehensive self-test for complete training workflow
  - Support both pretraining and finetuning modes through configuration
  - **Critical**: Essential task implementation for training workflow - must be P0
  - _Requirements: FR1, FR3_
  - _Leverage: src/task_factory/task/CDDG/ (for existing CDDG patterns)_
- [ ] 
  - **File**: scripts/test_pipeline03_integration.py
  - Implement end-to-end testing of HSE Prompt with Pipeline_03 workflow
  - Test create_pretraining_config and create_finetuning_config integration
  - Verify checkpoint loading and parameter freezing in Pipeline_03 context
  - Test multi-backbone comparison experiments with HSE prompt features
  - Include Pipeline_03 configuration validation and error handling
  - Add comparison tests with baseline Pipeline_03 runs (no prompts)
  - _Requirements: FR5_
  - _Leverage: src/Pipeline_03_multitask_pretrain_finetune.py (for integration testing)_

### P1 Feature Enhancement (Implement after P0)

**Note**: Critical Pipeline_03 integration tasks have been moved to P0 for core functionality. P1 now focuses on enhancements, experiments, and optimizations.
- [ ] 
  - **File**: src/utils/config/hse_prompt_validator.py
  - Implement configuration validation for HSE prompt-guided training
  - Add automatic path standardization and metadata file verification
  - Include fusion strategy validation and parameter range checking
  - Add configuration fixing utilities with clear error reporting
  - Support both pretraining and finetuning configuration validation
  - _Requirements: FR4_
  - _Leverage: src/utils/config/path_standardizer.py (for path handling patterns)_
- [ ] 
  - **Files**:
    - configs/pipeline_03/ablation/hse_system_prompt_only.yaml
    - configs/pipeline_03/ablation/hse_sample_prompt_only.yaml 
    - configs/pipeline_03/ablation/hse_no_prompt_baseline.yaml
  - Create Pipeline_03 compatible ablation study configurations
  - Configure different two-level prompt combinations: system-only, sample-only, none
  - Ensure all ablation configs use identical Pipeline_03 training settings
  - Add proper experimental controls with same backbone architectures and hyperparameters
  - Include configuration for standard contrastive learning baseline (no prompts)
  - _Requirements: FR6_
  - _Leverage: configs/pipeline_03/hse_prompt_multitask_config.yaml (for base structure)_
- [ ] 
  - **File**: scripts/run_hse_prompt_pipeline03.py
  - Implement HSE Prompt experiments using Pipeline_03 workflow
  - Add support for cross-dataset backbone comparison experiments
  - Include HSEPromptPipelineIntegration adapter for seamless Pipeline_03 usage
  - Add automated result collection from Pipeline_03's standardized output format
  - Support stage-specific execution (pretraining-only, finetuning-only, complete)
  - Include comprehensive logging and experiment tracking
  - _Requirements: FR5, FR6_
  - _Leverage: src/Pipeline_03_multitask_pretrain_finetune.py (main execution patterns)_

### P2 Performance Optimization (Lower priority)

- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/components/MixedPrecisionWrapper.py
  - Implement FP16 mixed precision wrapper for memory efficiency
  - Add gradient scaling and unscaling for stable training
  - Include compatibility checks for different PyTorch versions
  - Add performance benchmarking utilities for speed/memory comparison
  - Support automatic fallback to FP32 when hardware incompatible
  - _Requirements: NFR-P2 (memory efficiency)_
  - _Leverage: src/trainer_factory/ (for training optimization patterns)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/components/MemoryOptimizedFusion.py
  - Implement memory-optimized version of PromptFusion with gradient checkpointing
  - Add dynamic batch size adjustment based on available GPU memory
  - Include memory profiling utilities for optimization tuning
  - Add fallback to standard fusion when memory sufficient
  - Support memory usage monitoring and automatic optimization
  - _Requirements: NFR-P2 (memory efficiency)_
  - _Leverage: src/model_factory/ISFM_Prompt/components/PromptFusion.py (as base)_
- [ ] 
  - **File**: tests/performance/prompt_benchmarks.py
  - Implement comprehensive performance testing for all prompt components
  - Add latency benchmarking with different input sizes and batch sizes
  - Include memory usage profiling with peak usage tracking
  - Add throughput testing for real-time inference requirements
  - Include comparative analysis with baseline methods
  - _Requirements: NFR-P1, NFR-P2, NFR-P3_
  - _Leverage: src/model_factory/ISFM_Prompt/ (for component testing)_
- [ ] 
  - **File**: tests/integration/test_hse_prompt_workflow.py
  - Implement end-to-end workflow testing for two-stage training
  - Add cross-system generalization testing with multiple datasets
  - Include ablation study automation with statistical significance testing
  - Add configuration compatibility testing for all supported combinations
  - Support automated regression testing for continuous integration
  - _Requirements: FR6, NFR-R2_
  - _Leverage: configs/demo/HSE_Contrastive/ (for configuration testing)_

## Implementation Notes

### Implementation Order for P0 Core Functionality

**Phase 1: Foundation (Tasks 1-4)**
1. Module structure initialization (__init__.py files)
2. SystemPromptEncoder.py (two-level prompt encoding)
3. PromptFusion.py (fusion strategies) 
4. E_01_HSE_v2.py (completely new HSE implementation)

**Phase 2: Model Integration (Task 5)**
5. M_02_ISFM_Prompt.py (main model with prompt support)

**Phase 3: Pipeline_03 Integration (Tasks 6-8)**  
6. hse_prompt_integration.py (Pipeline_03 adapter)
7. prompt_contrastive.py (contrastive loss wrapper)
8. hse_contrastive.py (task implementation)

**Phase 4: Configuration & Testing (Tasks 9-11)**
9. Pipeline_03 configuration YAML
10. Component testing
11. End-to-end integration testing

### Key Design Decisions

1. **Two-Level Prompts Only**: System-level (Dataset_id + Domain_id) + Sample-level (Sample_rate) prompts. NO fault-level prompts since Label is the prediction target.
2. **Complete Model Isolation**: E_01_HSE_v2.py and M_02_ISFM_Prompt.py are completely independent from existing E_01_HSE.py and M_01_ISFM.py with zero code sharing or dependencies.
3. **Pipeline_03 Integration**: Leverage MultiTaskPretrainFinetunePipeline's mature two-stage workflow instead of custom TwoStageController.
4. **Factory Pattern Compliance**: All components registered in independent ISFM_Prompt namespace with proper component dictionaries.
5. **Self-Testing Requirements**: Every component must include `if __name__ == '__main__':` self-test with comprehensive validation of functionality.
6. **Configuration Reuse**: Utilize Pipeline_03's create_pretraining_config() and create_finetuning_config() utilities for seamless integration.

### Risk Mitigation

- **Backward Compatibility**: Independent M_02_ISFM_Prompt ensures no disruption to existing functionality
- **Graceful Degradation**: All components include fallback behavior when metadata unavailable
- **Comprehensive Testing**: Each component and integration thoroughly tested with multiple scenarios
- **Configuration Validation**: Automatic validation and fixing for common configuration errors

### Success Metrics

**P0 Core Functionality Completion Criteria:**
- [ ] All 11 P0 tasks completed with passing self-tests
- [ ] E_01_HSE_v2.py completely independent from E_01_HSE.py (zero dependencies)
- [ ] Pipeline_03 integration working with all backbone comparisons
- [ ] Two-level prompt system functional (System + Sample, no fault-level)
- [ ] End-to-end training workflow operational (pretrain → finetune via Pipeline_03)
- [ ] Complete model isolation verified (ISFM_Prompt namespace)

**P1 Enhancement Completion Criteria:**
- [ ] Ablation studies and cross-system experiments operational  
- [ ] Configuration validation and automated experiment scripts functional
- [ ] Performance benchmarking and optimization features complete

**Technical Performance Targets:**
- **Cross-System Accuracy**: >85% accuracy on cross-dataset generalization tasks
- **Performance Targets**: <100ms inference latency, <8GB memory usage, >50 samples/second throughput
- **Integration**: 100% compatibility with existing PHM-Vibench workflows
