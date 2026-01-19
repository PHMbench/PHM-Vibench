# Tests

This repo maintains a **minimal default test suite** under `test/` that should run in all environments.

Policy (UXFD merge):
- Keep only basic, dependency-free regression tests enabled by default.
- Move larger integration / MultiTaskPHM / heavy configuration tests into `test/TODO/` until they are stabilized.

Run:
```bash
python -m pytest test/
```

Run a TODO test explicitly:
```bash
python -m pytest test/TODO/todo_end_to_end_integration.py
```

