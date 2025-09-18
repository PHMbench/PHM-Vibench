# Bug Report

## Bug Summary
TypeError in multitask model initialization due to NaN values in label data causing num_classes calculation to fail

## Bug Details

### Expected Behavior
- Multi-task models should initialize successfully with proper class count calculation
- `get_num_classes()` method should return valid integer values for all datasets
- Training should start without TypeError exceptions

### Actual Behavior  
- TypeError occurs during model initialization: `empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType)`
- Error originates in `MultiTaskHead` when `n_classes` is None
- Training fails to start

### Steps to Reproduce
1. Run multi-task experiment: `bash "script/Vibench_paper/foundation_model/test_multitask.sh"`
2. System attempts to initialize B_04_Dlinear model with MultiTaskHead
3. `M_01_ISFM.get_num_classes()` encounters NaN values in Label column
4. Returns None instead of valid class count
5. `MultiTaskHead.__init__()` fails with TypeError

### Environment
- **Version**: PHM-Vibench v5.0
- **Platform**: Linux 6.8.0-65-generic
- **Configuration**: Multi-task training configuration in script/Vibench_paper/foundation_model/

## Impact Assessment

### Severity
- [x] Critical - System unusable
- [ ] High - Major functionality broken
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
- Researchers using multi-task learning experiments
- Users working with datasets containing NaN or -1 label values
- Anyone running foundation model multi-task training

### Affected Features
- Multi-task model initialization (M_01_ISFM with MultiTaskHead)
- Cross-dataset experiments with inconsistent labeling
- All 4 planned multi-task models (B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO)

## Additional Context

### Error Messages
```
TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType)
```

### Screenshots/Media
Error occurs in file: `/src/model_factory/ISFM/task_head/multi_task_head.py:131`
Code location: `nn.Linear(self.hidden_dim // 2, n_classes)`

### Related Issues
- Label column contains NaN values in some datasets
- Some datasets use -1 to mark anomalous or unlabeled samples
- `max()` function returns NaN when applied to columns containing NaN values

## Initial Analysis

### Suspected Root Cause
The `get_num_classes()` method in `M_01_ISFM.py` (lines 66-70) fails to handle NaN and -1 values in Label columns:

```python
def get_num_classes(self):
    num_classes = {}
    for key in np.unique(self.metadata.df['Dataset_id']):
        num_classes[key] = max(self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']) + 1
    return num_classes
```

### Affected Components
- `/src/model_factory/ISFM/M_01_ISFM.py` - Root cause in get_num_classes()
- `/src/model_factory/ISFM/task_head/multi_task_head.py` - Fails when receiving None
- Dataset metadata - Contains inconsistent label formats across datasets

### Data Analysis Results
```
Label column info:
- Data type: float64
- Has NaN values: True
- Dataset_id 1: Label range 0.0 to 3.0, Has NaN: True
- Dataset_id 2: Label range -1.0 to 15.0, Has NaN: False  
- Dataset_id 3: Label range -1.0 to 17.0, Has NaN: True
```