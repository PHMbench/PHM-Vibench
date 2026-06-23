# Contract: PHM 2025+ Literature Inventory

## Command

```bash
python -m scripts.phm_literature_matrix --min-count 50
```

## Required Behavior

- Reads `docs/literature/phm_2025_plus.csv`.
- Validates required fields, year boundary, uniqueness, and support status.
- Prints a Markdown report grouped by task family, method family, and support
  status.
- Exits non-zero on malformed inventory data.

## Optional Arguments

- `--inventory <path>`: validate an alternate CSV.
- `--min-count <n>`: minimum number of entries; default 50.
- `--format markdown|json`: output report format; default markdown.

## Non-Goals

- Does not fetch network data.
- Does not declare a paper's method implemented unless existing registry/test
  evidence already supports the mapped surface.
