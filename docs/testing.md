# Testing Guide

## Pytest

```bash
python -m pytest test/
```

## Legacy runner (optional; historical matrix)

`dev/test_history/` contains a historical test runner used during earlier refactors. It may require additional
dependencies and is not part of the maintained workflow.

```bash
python dev/test_history/run_tests.py --unit
```
