# TODO Tests (Parked)

This folder contains tests that are **temporarily parked** and excluded from the default `pytest test/` run.

Reasons (typical):
- Depend on unavailable config paths / external assets
- Require additional datasets or optional dependencies
- Exercise unstable MultiTaskPHM behaviors that need spec alignment

How to run one test explicitly:
```bash
python -m pytest test/TODO/todo_multi_task_phm_comprehensive.py
```

