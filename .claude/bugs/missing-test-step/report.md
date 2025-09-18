# Bug Report

## Bug Summary
MisconfigurationException: Multi-task PHM Lightning module missing `test_step()` method, preventing model testing phase during Pipeline_01_default execution.

## Bug Details

### Expected Behavior
The multi-task PHM training should complete successfully including the testing phase. After training, the pipeline should load the best checkpoint and run `trainer.test()` to evaluate model performance on test data.

### Actual Behavior  
Training completes successfully, but when attempting to run the test phase (`trainer.test()`), PyTorch Lightning throws a MisconfigurationException because the multi-task PHM Lightning module does not implement the required `test_step()` method.

### Steps to Reproduce
1. Run multi-task training with debug config: `python main_LQ.py --config script/Vibench_paper/foundation_model/multitask_B_04_Dlinear_debug.yaml`
2. Training completes successfully with validation metrics logged
3. Pipeline attempts to load best checkpoint and run testing phase
4. `trainer.test()` called in Pipeline_01_default.py line 136
5. Exception raised: "No `test_step()` method defined to run `Trainer.test`"

### Environment
- **Version**: PHM-Vibench v5.0
- **Platform**: Linux (Python 3.8+, PyTorch 2.6.0, PyTorch Lightning)
- **Configuration**: Multi-task B_04_Dlinear debug configuration
- **Task Type**: In_distribution multi-task PHM training

## Impact Assessment

### Severity
- [x] High - Major functionality broken

### Affected Users
Users running multi-task PHM experiments who need test phase evaluation results.

### Affected Features
- Multi-task foundation model testing and evaluation
- Pipeline_01_default test phase execution
- Model performance assessment on test datasets
- Complete experiment workflow completion

## Additional Context

### Error Messages
```
lightning_fabric.utilities.exceptions.MisconfigurationException: No `test_step()` method defined to run `Trainer.test`.
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/src/Pipeline_01_default.py", line 136, in pipeline
    result = trainer.test(task, data_factory.get_dataloader('test'))
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 45, in main
    results = pipeline.pipeline(args)
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 51, in <module>
    main()
```

### Related Issues
- Multi-task PHM Lightning module (src/task_factory/task/In_distribution/multi_task_phm.py) implements `training_step()` and `validation_step()` but lacks `test_step()`
- Other task types (Default_task subclasses) have proper `test_step()` implementations
- Pipeline_01_default assumes all Lightning modules implement the complete training/validation/testing interface

### Code Context
**Affected File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
- Has `training_step()` (line ~200) - working
- Has `validation_step()` (line ~220) - working  
- **Missing**: `test_step()` method - causing failure

**Pipeline Code**: `src/Pipeline_01_default.py:136`
```python
# This line fails when test_step() is not implemented
result = trainer.test(task, data_factory.get_dataloader('test'))
```

## Initial Analysis

### Suspected Root Cause
The multi-task PHM Lightning module (`task` class in multi_task_phm.py) was refactored from the original multi_task_lightning.py but the `test_step()` method was not implemented during the migration. The module implements training and validation steps for multi-task learning but lacks the testing interface.

### Affected Components
- **Primary**: `src/task_factory/task/In_distribution/multi_task_phm.py` - missing `test_step()`
- **Secondary**: `src/Pipeline_01_default.py` - calls `trainer.test()` expecting complete Lightning interface
- **Related**: Multi-task head implementations that need testing evaluation