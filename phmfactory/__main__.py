"""Process entrypoint for ``python -m phmfactory``.

The programmatic router may return dictionaries, lists, or Pipeline results. This module
uses the dedicated integer process entrypoint so successful structured results always
produce operating-system exit code ``0``.
"""

from __future__ import annotations

from phmfactory.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
