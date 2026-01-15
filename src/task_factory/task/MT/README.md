# Multi-Task Learning (MT)

> **Status**: ⚠️ **Deprecated**
>
> This module has been superseded by `In_distribution/multi_task_phm.py` and is kept for reference only.

---

## Overview

This directory contains `multi_task_lightning.py`, the original multi-task learning implementation for PHM-Vibench.

## Deprecation Notice

This module is **deprecated** and should not be used for new experiments. The code has been refactored and improved in:

**Replacement**: `src/task_factory/task/In_distribution/multi_task_phm.py`

## Migration Guide

If you have existing code using this module:

```python
# Old (deprecated)
from src.task_factory.task.MT.multi_task_lightning import MultiTaskLightningModule

# New (recommended)
from src.task_factory.task.In_distribution.multi_task_phm import MultiTaskModule
```

## Files

| File | Description | Status |
|------|-------------|--------|
| `multi_task_lightning.py` | Original multi-task implementation (21KB) | Deprecated |

## Historical Notes

- This was the initial multi-task implementation for PHM-Vibench
- Refactored to improve code organization and maintainability
- Kept in repository for historical reference and potential rollback needs

---

## Related Documentation

- [@../In_distribution/README.md] - Updated multi-task implementation
- [@../README.md] - Task Factory Overview
- [@../../README.md] - Task Factory Root Documentation

---

## Recommendation

**Do not use this module for new experiments.** Use the refactored version in `In_distribution/` or the standard `Default_task` with multi-task support.
