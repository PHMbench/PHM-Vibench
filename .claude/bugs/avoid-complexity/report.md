# Bug Report: Avoiding Unnecessary Complexity

## Bug Summary
The codebase contains multiple instances of unnecessary complexity that violate the project's core principle of "简单 > 聪明" (simplicity over cleverness). These patterns make the code harder to maintain, debug, and extend, going against the CLAUDE.md coding guidelines.

## Bug Details

### Expected Behavior
Code should follow the project's core principles from CLAUDE.md:
- **直白实现**: 简单 > 聪明；能用朴素结构就别引入框架
- **可读第一**: 代码自解释；注释讲"为何"不讲"如何"
- **接口最小**: 公共 API 小而稳；不暴露内部状态
- **失败尽早**: 前置校验、边界检查、显式异常

### Actual Behavior
Several modules exhibit unnecessary complexity patterns:
- Global state manipulation for configuration
- Wildcard imports obscuring dependencies
- Complex conditional logic without clear abstraction
- Magic strings and implicit behavior

### Steps to Reproduce
1. Examine `src/Pipeline_03_multitask_pretrain_finetune.py` lines 69-71
2. Review ISFM model imports in `src/model_factory/ISFM/M_01_ISFM.py`
3. Check configuration handling patterns across pipelines
4. Observe the complexity in model instantiation logic

### Environment
- **Version**: PHM-Vibench current version
- **Platform**: Python codebase
- **Configuration**: All modules affected

## Impact Assessment

### Severity
- [x] Medium - Feature impaired but workaround exists

### Affected Users
- **Developers**: Difficulty understanding and modifying code
- **Maintainers**: Increased debugging and refactoring overhead
- **New Contributors**: Higher barrier to entry

### Affected Features
- Configuration management system
- Model factory instantiation
- Pipeline orchestration
- Import resolution

## Additional Context

### Error Messages
While not generating runtime errors, these patterns create maintenance burdens:
```python
# Complex global state manipulation
if '_override_params' in globals():
    overrides = globals()['_override_params']
```

### Code Examples

#### 1. Global State Manipulation (Pipeline_03)
**File**: `src/Pipeline_03_multitask_pretrain_finetune.py:69-71`
```python
# Problematic: Global state access
overrides = {}
if '_override_params' in globals():
    overrides = globals()['_override_params']
    print(f"[INFO] 在Pipeline_03中覆盖配置参数: {overrides}")
```

**Issues**:
- Hidden dependencies on global state
- Implicit configuration mechanism
- Difficult to test and reason about

#### 2. Wildcard Imports (ISFM Models)
**File**: `src/model_factory/ISFM/M_01_ISFM.py:3-6`
```python
# Problematic: Wildcard imports
from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
```

**Issues**:
- Namespace pollution
- Hidden dependencies
- Import conflicts potential
- IDE/tooling support impaired

#### 3. Dictionary-Based Component Registration
**File**: `src/model_factory/ISFM/M_01_ISFM.py:14-30`
```python
# Complex dictionary mapping
Embedding_dict = {
    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,
    'E_03_Patch_DPOT': E_03_Patch_DPOT,
}
Backbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
    'B_03_FITS': B_03_FITS,
    # ... many more entries
}
```

**Issues**:
- Magic strings for component names
- Manual maintenance required
- No validation or error handling
- Duplication across model files

### Related Issues
- Code review standards not consistently applied
- Technical debt accumulation from rapid development
- Inconsistent patterns across similar modules

## Initial Analysis

### Suspected Root Cause
1. **Rapid development pressure** leading to quick fixes
2. **Lack of consistent code review standards**
3. **Missing architectural guidelines** for complex features
4. **Copy-paste programming** spreading complexity patterns

### Affected Components
- **Pipeline modules**: `Pipeline_01_default.py`, `Pipeline_02_pretrain_fewshot.py`, `Pipeline_03_multitask_pretrain_finetune.py`
- **Model factory**: All ISFM model files (`M_01_ISFM.py`, `M_02_ISFM.py`, `M_03_ISFM.py`)
- **Configuration system**: Cross-cutting concern affecting multiple modules
- **Component registration**: Factory pattern implementations

### Improvement Opportunities
1. **Replace global state** with explicit parameter passing
2. **Convert wildcard imports** to explicit imports
3. **Implement proper component registry** with validation
4. **Add configuration validation** with clear error messages
5. **Extract complex conditional logic** into well-named methods
6. **Standardize error handling patterns** across modules

### Priority Areas
1. **High Priority**: Pipeline configuration handling - affects all experiments
2. **Medium Priority**: ISFM model imports - affects model instantiation reliability  
3. **Medium Priority**: Component registration patterns - affects extensibility
4. **Low Priority**: Code style consistency - affects long-term maintainability

---

**Note**: This bug report serves as a foundation for systematic refactoring to improve code quality and maintainability while preserving functionality.