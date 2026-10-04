"""Direct output files for the existing symbolic-diagnosis protocol."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Iterator

import numpy as np
import sklearn
from phmfactory import __version__


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


@contextmanager
def run_output(path: Path, configuration: dict) -> Iterator[Path]:
    """Refuse replacement and retain a failed run; installed packages need no Git."""
    path.mkdir(parents=True, exist_ok=False)
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'scikit-learn': sklearn.__version__, 'phmfactory': __version__}
    write_json(path / 'config.json', {'arguments': configuration, 'environment': environment})
    start = datetime.now(timezone.utc).isoformat()
    try:
        yield path
    except BaseException as error:
        write_json(path / 'error.json', {'type': type(error).__name__, 'message': str(error)})
        write_json(path / 'run_status.json', {'status': 'failed', 'started': start,
                   'finished': datetime.now(timezone.utc).isoformat()})
        raise
    else:
        write_json(path / 'run_status.json', {'status': 'completed', 'started': start,
                   'finished': datetime.now(timezone.utc).isoformat()})
