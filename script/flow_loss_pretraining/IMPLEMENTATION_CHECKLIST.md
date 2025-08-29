# Flow Loss Pretraining Implementation Checklist

## Phase 1: Core Components Implementation

### 1.1 Flow Network Module ✓ Ready to implement
- [ ] Create directory: `src/model_factory/ISFM/flow_net/`
- [ ] Implement `src/model_factory/ISFM/flow_net/__init__.py`
- [ ] Implement `src/model_factory/ISFM/flow_net/F_01_RectifiedFlow.py`
  - [ ] Time embedding network
  - [ ] Main flow prediction network with SiLU activations
  - [ ] Forward method for velocity prediction
  - [ ] Input validation and shape checking
- [ ] Add base class `src/model_factory/ISFM/flow_net/base_flow.py`

### 1.2 Conditional Encoder Module ✓ Ready to implement
- [ ] Create directory: `src/model_factory/ISFM/encoder/` (if not exists)
- [ ] Implement `src/model_factory/ISFM/encoder/__init__.py`
- [ ] Implement `src/model_factory/ISFM/encoder/E_03_ConditionalEncoder.py`
  - [ ] Domain embedding layer
  - [ ] System embedding layer
  - [ ] Conditional encoding network
  - [ ] Forward method with domain/system conditioning
- [ ] Add to existing embedding dictionary

### 1.3 Enhanced ISFM Model ✓ Ready to implement
- [ ] Implement `src/model_factory/ISFM/M_04_ISFM_Flow.py`
  - [ ] Initialize conditional encoder
  - [ ] Initialize decoder network
  - [ ] Initialize flow network
  - [ ] Optional classifier initialization
  - [ ] Forward method with flow prediction
  - [ ] Sampling method for generation
- [ ] Register in ISFM model dictionary

## Phase 2: Loss Function Implementation

### 2.1 Flow Pretrain Loss ✓ Ready to implement
- [ ] Implement `src/task_factory/Components/flow_pretrain_loss.py`
  - [ ] FlowPretrainLossCfg dataclass with all hyperparameters
  - [ ] FlowPretrainLoss class extending nn.Module
  - [ ] Reconstruction loss computation
  - [ ] Flow matching loss computation
  - [ ] Contrastive flow loss computation
  - [ ] Hierarchical loss computation methods:
    - [ ] Domain separation loss
    - [ ] System cohesion loss  
    - [ ] Hierarchical margin loss
  - [ ] Regularization loss (radius + centering)
  - [ ] Optional classification loss
  - [ ] Total loss aggregation
  - [ ] Loss statistics tracking

### 2.2 Loss Component Utilities ✓ Ready to implement
- [ ] Implement helper functions in `flow_pretrain_loss.py`:
  - [ ] `compute_pairwise_distances()`
  - [ ] `compute_domain_centers()`
  - [ ] `compute_system_cohesion()`
  - [ ] `apply_margin_constraint()`

## Phase 3: Task Integration

### 3.1 Flow Pretraining Task ✓ Ready to implement
- [ ] Create directory: `src/task_factory/task/pretrain/`
- [ ] Implement `src/task_factory/task/pretrain/__init__.py`
- [ ] Implement `src/task_factory/task/pretrain/flow_pretrain_task.py`
  - [ ] FlowPretrainTask class extending pl.LightningModule
  - [ ] Initialize loss function and metrics
  - [ ] training_step method:
    - [ ] Batch processing with metadata extraction
    - [ ] Domain/system ID extraction
    - [ ] Forward pass through model
    - [ ] Loss computation
    - [ ] Metric logging
  - [ ] validation_step method
  - [ ] configure_optimizers method
  - [ ] Learning rate scheduling

### 3.2 Task Factory Registration ✓ Ready to implement
- [ ] Update `src/task_factory/task_factory.py`:
  - [ ] Import FlowPretrainTask
  - [ ] Add to task_dict: `("pretrain", "flow_pretrain"): FlowPretrainTask`
- [ ] Update `src/task_factory/__init__.py` if needed

## Phase 4: Model Factory Integration

### 4.1 Component Registration ✓ Ready to implement
- [ ] Update `src/model_factory/ISFM/__init__.py`:
  - [ ] Import new components
  - [ ] Add E_03_ConditionalEncoder to Embedding_dict
  - [ ] Add F_01_RectifiedFlow to new FlowNet_dict
  - [ ] Update Model selection logic for M_04_ISFM_Flow

### 4.2 Factory Method Updates ✓ Ready to implement
- [ ] Update `src/model_factory/model_factory.py`:
  - [ ] Handle flow-enabled models
  - [ ] Pass flow network configuration
  - [ ] Validate flow model requirements

## Phase 5: Pipeline Integration

### 5.1 Enhanced Pretraining Pipeline ✓ Ready to implement
- [ ] Update `src/Pipeline_03_multitask_pretrain_finetune.py`:
  - [ ] Add `run_flow_pretraining_stage()` method
  - [ ] Flow configuration setup
  - [ ] Integration with existing pipeline stages
  - [ ] Checkpoint management for flow models
  - [ ] Results tracking and logging

### 5.2 Pipeline Configuration ✓ Ready to implement  
- [ ] Update `src/utils/pipeline_config.py`:
  - [ ] Add `create_flow_pretraining_config()` function
  - [ ] Flow-specific parameter handling
  - [ ] Validation for flow configurations

## Phase 6: Configuration System

### 6.1 Configuration Templates ✓ Ready to implement
- [ ] Create `configs/demo/Pretraining/flow_pretrain.yaml`
  - [ ] Environment configuration
  - [ ] Data configuration with metadata
  - [ ] Model configuration for M_04_ISFM_Flow
  - [ ] Task configuration with loss hyperparameters
  - [ ] Trainer configuration
- [ ] Create variants:
  - [ ] `flow_pretrain_basic.yaml` - Simple configuration
  - [ ] `flow_pretrain_advanced.yaml` - Full hierarchical setup

### 6.2 Configuration Validation ✓ Ready to implement
- [ ] Add to `src/utils/config_validator.py`:
  - [ ] Flow model configuration validation
  - [ ] Loss hyperparameter range checking
  - [ ] Domain/system metadata requirements
  - [ ] Compatibility checks

## Phase 7: Data Processing Enhancements

### 7.1 Metadata Enhancement ✓ Ready to implement
- [ ] Update `src/data_factory/ID_dataset.py`:
  - [ ] Ensure domain ID extraction from metadata
  - [ ] Ensure system ID extraction from metadata  
  - [ ] Batch formatting for hierarchical information
  - [ ] Metadata validation

### 7.2 Data Factory Integration ✓ Ready to implement
- [ ] Update data factory components if needed:
  - [ ] Domain/system ID mapping
  - [ ] Metadata preprocessing
  - [ ] Batch construction with hierarchical info

## Phase 8: Testing and Validation

### 8.1 Unit Tests ✓ Ready to implement
- [ ] Create `test/model_factory/test_flow_network.py`:
  - [ ] Test F_01_RectifiedFlow forward pass
  - [ ] Test input/output shapes
  - [ ] Test time embedding
  - [ ] Test gradient flow

- [ ] Create `test/model_factory/test_conditional_encoder.py`:
  - [ ] Test E_03_ConditionalEncoder forward pass
  - [ ] Test domain/system embedding
  - [ ] Test conditional encoding

- [ ] Create `test/task_factory/test_flow_pretrain_loss.py`:
  - [ ] Test individual loss components
  - [ ] Test hierarchical loss computation
  - [ ] Test loss aggregation
  - [ ] Test gradient computation

- [ ] Create `test/task_factory/test_flow_pretrain_task.py`:
  - [ ] Test training step
  - [ ] Test validation step
  - [ ] Test optimizer configuration

### 8.2 Integration Tests ✓ Ready to implement
- [ ] Create `test/integration/test_flow_pretraining_pipeline.py`:
  - [ ] Test full pretraining pipeline
  - [ ] Test model checkpoint saving/loading
  - [ ] Test configuration loading
  - [ ] Test multi-domain training

### 8.3 Performance Tests ✓ Ready to implement
- [ ] Create `test/performance/test_flow_memory_usage.py`:
  - [ ] Memory usage profiling
  - [ ] Training time benchmarking
  - [ ] GPU utilization testing

## Phase 9: Documentation and Examples

### 9.1 Code Documentation ✓ Ready to implement
- [ ] Add comprehensive docstrings to all new classes
- [ ] Add type hints throughout codebase
- [ ] Add inline comments for complex algorithms
- [ ] Update existing CLAUDE.md files in relevant modules

### 9.2 Usage Examples ✓ Ready to implement
- [ ] Create `examples/flow_pretraining_basic.py`:
  - [ ] Simple flow pretraining example
  - [ ] Basic configuration setup
  - [ ] Training loop demonstration

- [ ] Create `examples/flow_pretraining_advanced.py`:
  - [ ] Advanced hierarchical pretraining
  - [ ] Multi-domain/system setup
  - [ ] Custom loss configuration

### 9.3 Tutorial Documentation ✓ Ready to implement
- [ ] Create `docs/tutorials/flow_pretraining_tutorial.md`:
  - [ ] Step-by-step tutorial
  - [ ] Configuration explanations
  - [ ] Troubleshooting guide
  - [ ] Performance optimization tips

## Phase 10: Optimization and Performance

### 10.1 Performance Optimization ✓ For future implementation
- [ ] Profile memory usage and optimize
- [ ] Implement gradient checkpointing if needed
- [ ] Optimize batch processing for hierarchical losses
- [ ] Add mixed precision training support

### 10.2 Hyperparameter Tuning ✓ For future implementation
- [ ] Implement hyperparameter search utilities
- [ ] Create default parameter recommendations
- [ ] Add automated hyperparameter validation
- [ ] Performance benchmarking against baselines

## Implementation Priority Order

### High Priority (Core Functionality)
1. **Flow Network Module** (Phase 1.1) - Essential component
2. **Conditional Encoder Module** (Phase 1.2) - Required for hierarchical conditioning
3. **Enhanced ISFM Model** (Phase 1.3) - Integrates everything
4. **Flow Pretrain Loss** (Phase 2.1) - Core loss function
5. **Flow Pretraining Task** (Phase 3.1) - Lightning module
6. **Model Factory Integration** (Phase 4) - Registration
7. **Configuration Templates** (Phase 6.1) - Basic configs

### Medium Priority (Integration)
8. **Task Factory Registration** (Phase 3.2) - Task registration
9. **Pipeline Integration** (Phase 5) - Pipeline modifications
10. **Data Processing Enhancements** (Phase 7) - Metadata handling
11. **Unit Tests** (Phase 8.1) - Basic testing

### Lower Priority (Polish)
12. **Integration Tests** (Phase 8.2) - Comprehensive testing
13. **Documentation** (Phase 9) - Documentation and examples
14. **Configuration Validation** (Phase 6.2) - Advanced validation
15. **Performance Tests** (Phase 8.3) - Performance testing
16. **Optimization** (Phase 10) - Performance improvements

## Key Dependencies

### Internal Dependencies
- Existing ISFM architecture (M_01-03_ISFM.py)
- Task factory framework (task_factory.py)
- Model factory framework (model_factory.py)
- Configuration system (config_utils.py)
- Data factory (ID_dataset.py)

### External Dependencies
- PyTorch Lightning for training orchestration
- PyTorch for neural networks and autograd
- NumPy for numerical operations
- Dataclasses for configuration management

## Success Criteria

### Functional Requirements ✓
- [ ] Flow network successfully predicts velocities
- [ ] Hierarchical losses organize latent space as intended
- [ ] Full training pipeline runs without errors
- [ ] Models can be saved and loaded properly
- [ ] Configurations are properly validated

### Performance Requirements ✓
- [ ] Training converges within reasonable epochs (< 1000)
- [ ] Memory usage is reasonable for typical batch sizes
- [ ] Latent space shows hierarchical organization (domain > system > instance)
- [ ] Reconstruction quality is maintained
- [ ] Flow sampling generates realistic data

### Integration Requirements ✓
- [ ] Compatible with existing PHM-Vibench architecture
- [ ] Follows factory pattern conventions
- [ ] Proper error handling and logging
- [ ] Comprehensive testing coverage (>90%)
- [ ] Clear documentation and examples

This checklist provides a comprehensive roadmap for implementing flow loss pretraining in PHM-Vibench while maintaining compatibility with the existing framework architecture.