# HSE Industrial Contrastive Learning Implementation Plan

## Task Overview

This plan implements a Prompt-guided Contrastive Learning system for industrial vibration analysis, focusing on System Information Prompt encoding combined with SOTA contrastive learning methods. The implementation creates two-level prompts (System+Sample, NO fault-level since Label is prediction target), independent M_02_ISFM_Prompt model, two-stage training, and complete self-testing infrastructure.

Key Innovation: First-ever combination of system metadata as learnable prompts with contrastive learning for cross-system fault diagnosis generalization.

## Steering Document Compliance

All tasks follow PHM-Vibench factory patterns with strict component registration and self-testing requirements. The implementation maintains backward compatibility while introducing the M_02_ISFM_Prompt as an independent model to avoid conflicts. Each component includes `if __name__ == '__main__':` self-test sections and integrates with existing contrastive loss infrastructure.

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
  - **File**: src/model_factory/ISFM_Prompt/embedding/E_01_HSE_Prompt.py
  - Extend E_01_HSE with SystemPromptEncoder and PromptFusion integration
  - Add metadata parameter to forward() method for system information processing
  - Implement training_stage control with prompt freezing for two-stage training
  - Include fallback to original HSE behavior when metadata unavailable
  - Add complete self-test with metadata validation and stage switching
  - _Requirements: FR1, FR2_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (as base class)_
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
  - **File**: configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml
  - Configure M_02_ISFM_Prompt with E_01_HSE_Prompt embedding for pretraining stage
  - Set training_stage: 'pretrain', freeze_prompt: false, contrast_weight: 0.15
  - Configure multi-source domain data loading with proper metadata handling
  - Include model components: B_08_PatchTST backbone, H_10_ProjectionHead for contrastive learning
  - Add contrastive loss configuration with InfoNCE as default
  - _Requirements: FR1, FR4_
  - _Leverage: configs/demo/Single_DG/CWRU.yaml (for structure patterns)_
- [ ] 
  - **File**: configs/demo/HSE_Contrastive/hse_prompt_finetune.yaml
  - Configure M_02_ISFM_Prompt for finetuning stage with prompt freezing
  - Set training_stage: 'finetune', freeze_prompt: true, contrast_weight: 0.0
  - Use H_01_Linear_cla taskhead for classification, disable contrastive learning
  - Configure lower learning rate (1e-4) for finetuning phase
  - _Requirements: FR1, FR4_
  - _Leverage: configs/demo/Single_DG/CWRU.yaml (for classification config patterns)_
- [ ] 
  - **File**: src/model_factory/ISFM_Prompt/test_prompt_components.py
  - Implement comprehensive component testing for SystemPromptEncoder, PromptFusion, E_01_HSE_Prompt
  - Test integration between components with proper metadata flow
  - Verify two-stage training functionality and prompt freezing behavior
  - Add performance benchmarking for latency and memory usage requirements
  - Include cross-component compatibility testing
  - _Requirements: FR1, FR2, FR3_
  - _Leverage: src/model_factory/ISFM/embedding/E_01_HSE.py (for testing patterns)_

### P1 Feature Enhancement (Implement after P0)

- [ ] 
  - **File**: src/task_factory/Components/prompt_contrastive.py
  - Implement universal wrapper for all 6 existing contrastive losses with prompt guidance
  - Add system-aware positive/negative sampling using metadata system_ids
  - Include prompt similarity loss to encourage system-invariant representations
  - Add comprehensive self-test for all LOSS_MAPPING combinations with prompt features
  - Support graceful fallback to standard contrastive learning when prompts unavailable
  - _Requirements: FR1_
  - _Leverage: src/task_factory/Components/contrastive_losses.py (for all existing losses)_
- [ ] 
  - **File**: src/utils/training/TwoStageController.py
  - Implement automated two-stage training workflow: pretrain → finetune
  - Add checkpoint management with best model selection and recovery capability
  - Include parameter freezing utilities for prompt-related components
  - Add experiment state management for interruption recovery
  - Include progress tracking and logging for both training stages
  - _Requirements: FR1, FR5_
  - _Leverage: src/trainer_factory/ (for training patterns)_
- [ ] 
  - **File**: src/task_factory/task/CDDG/hse_contrastive.py
  - Implement task class integrating prompt-guided contrastive learning
  - Add metadata preprocessing for system information extraction
  - Include loss combination logic: classification + prompt-guided contrastive
  - Add comprehensive self-test for complete training workflow
  - Support both pretraining and finetuning modes through configuration
  - _Requirements: FR1, FR3_
  - _Leverage: src/task_factory/task/CDDG/ (for existing CDDG patterns)_
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
    - configs/demo/HSE_Contrastive/ablation/system_only_prompt.yaml
    - configs/demo/HSE_Contrastive/ablation/sample_only_prompt.yaml
    - configs/demo/HSE_Contrastive/ablation/no_prompt_baseline.yaml
  - Create ablation study configurations for systematic component evaluation
  - Configure different prompt level combinations for contribution analysis
  - Add baseline configuration without prompts for performance comparison
  - Include proper experimental controls with identical hyperparameters
  - _Requirements: FR6_
  - _Leverage: configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml (for base structure)_
- [ ] 
  - **File**: scripts/run_cross_system_experiments.py
  - Implement automated cross-dataset experiment execution
  - Add support for source-target domain combinations (CWRU→XJTU, etc.)
  - Include experiment progress tracking with interruption recovery
  - Add result collection and standardized reporting functionality
  - Support parallel experiment execution for efficiency
  - _Requirements: FR5, FR6_
  - _Leverage: main.py (for experiment execution patterns)_

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

### Key Design Decisions

1. **Two-Level Prompts Only**: System-level (Dataset_id + Domain_id) + Sample-level (Sample_rate) prompts. NO fault-level prompts since Label is the prediction target.
2. **Independent Model Architecture**: M_02_ISFM_Prompt avoids conflicts with existing M_01_ISFM.py by being completely independent while reusing backbone and taskhead components.
3. **Factory Pattern Compliance**: All components registered in appropriate factories with proper component dictionaries and configuration-driven initialization.
4. **Self-Testing Requirements**: Every component must include `if __name__ == '__main__':` self-test with comprehensive validation of functionality.
5. **Two-Stage Training**: Pretraining with contrastive learning enabled, finetuning with frozen prompts and classification-only loss.

### Risk Mitigation

- **Backward Compatibility**: Independent M_02_ISFM_Prompt ensures no disruption to existing functionality
- **Graceful Degradation**: All components include fallback behavior when metadata unavailable
- **Comprehensive Testing**: Each component and integration thoroughly tested with multiple scenarios
- **Configuration Validation**: Automatic validation and fixing for common configuration errors

### Success Metrics

- **P0 Completion**: Basic prompt-guided contrastive learning functional with self-tests passing
- **P1 Completion**: Two-stage training and ablation studies operational
- **P2 Completion**: Performance optimizations and comprehensive benchmarking complete
- **Cross-System Accuracy**: >85% accuracy on cross-dataset generalization tasks
- **Performance Targets**: <100ms inference latency, <8GB memory usage, >50 samples/second throughput
