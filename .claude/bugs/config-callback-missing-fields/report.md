# Bug Report: Configuration Callback Fields Missing

## Bug Summary
Trainer callback initialization fails when optional configuration fields (`pruning`, `early_stopping`, `monitor`, `patience`) are missing from YAML configuration files, causing AttributeError exceptions.

## Bug Details

### Expected Behavior
- Configuration should work with minimal required fields
- Optional callback fields should have sensible defaults
- Users should not need to specify all callback parameters in every config

### Actual Behavior  
- AttributeError occurs when accessing missing fields: `args.pruning`, `args.early_stopping`, `args.monitor`, `args.patience`
- Training fails to start when config files don't include optional callback parameters
- Code directly accesses attributes without checking existence

### Steps to Reproduce
1. Create YAML config without optional fields like `pruning` or `early_stopping`
2. Run experiment with this config
3. Trainer initialization fails with AttributeError
4. System crashes before training can begin

### Environment
- **Version**: PHM-Vibench v5.0
- **Platform**: Linux 6.8.0-65-generic
- **Configuration**: Any trainer configuration missing optional callback fields

## Impact Assessment

### Severity
- [x] High - Major functionality broken
- [ ] Critical - System unusable
- [ ] Medium - Feature impaired but workaround exists  
- [ ] Low - Minor issue or cosmetic

### Affected Users
- All users creating custom configurations
- Users following minimal configuration examples
- Anyone not using the full default.yaml template

### Affected Features
- Trainer callback initialization
- All experiment types using Default_trainer
- Configuration flexibility and user experience

## Additional Context

### Error Messages
```
AttributeError: 'types.SimpleNamespace' object has no attribute 'pruning'
AttributeError: 'types.SimpleNamespace' object has no attribute 'early_stopping'
AttributeError: 'types.SimpleNamespace' object has no attribute 'patience'
AttributeError: 'types.SimpleNamespace' object has no attribute 'monitor'
```

### Code Location
Error occurs in: `/src/trainer_factory/Default_trainer.py`
- Lines 93, 98: Direct attribute access to optional fields
- Lines 83, 143, 145: Required fields without defaults

### User Suggestion
Original user suggested: "按照config 类先默认指定一个yaml 比如 configs/default.yaml 再根据实际的 yaml 进行overwite" (Use a default YAML config first, then override with actual YAML)

## Root Cause Analysis

### Suspected Root Cause
The callback creation code uses direct attribute access (`args.pruning`, `args.early_stopping`) instead of safe access with defaults (`getattr(args, 'pruning', False)`). This breaks when users create minimal configurations.

### Affected Components
- `src/trainer_factory/Default_trainer.py` - Callback creation functions
- All YAML configurations missing optional callback fields
- User experience with configuration flexibility