# Quickstart: PHM 2025+ Literature Integration

Run the literature inventory module:

```bash
python -m scripts.phm_literature_matrix --min-count 50
```

Run focused tests:

```bash
python -m pytest -q test/test_phm_literature_matrix.py
```

Run repository validation gates relevant to this slice:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
bash scripts/run_demo_matrix.sh --mode smoke
```

Expected result: all commands exit 0, and the literature command reports at
least 50 entries with year >= 2025.
