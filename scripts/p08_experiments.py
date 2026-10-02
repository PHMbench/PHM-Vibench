"""Retired P08 protocol entrypoint; retained only to report the migration."""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "The alternate P08 native protocol is retired. Use python -m scripts.p08_physical "
        "with configs/experiments/p08/physical_conditioning.yaml. The schemas and "
        "selection rules differ; do not reuse source_only.yaml or reinterpret old outputs. "
        "The complete alternate implementation remains in Git commit 838ad940. "
        "See docs/P08_EXPERIMENT_A.md."
    )


if __name__ == "__main__":
    main()
