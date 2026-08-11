"""Compatibility process entrypoint for ``python main.py``.

The repository launcher intentionally shares the same integer process boundary as the
installed ``phmfactory`` command and ``python -m phmfactory``. Programmatic callers that
need the structured command or Pipeline result should import ``phmfactory.cli.main``.
"""

from __future__ import annotations

from phmfactory.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
