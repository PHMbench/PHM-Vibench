"""Identity validation before scientific grouping or coercion."""
from numbers import Real
import numpy as np


def validate_identifiers(values, name: str) -> list:
    """Check identities before sorting, casting or grouping; numeric zero is valid.

    Mixed text/numeric identities need a caller-provided provenance mapping, not
    implicit string conversion that can merge records or stringify missing data.
    """
    values = list(values)
    if not values:
        raise ValueError(f'{name}: nonempty identifiers required')
    kinds = set()
    for value in values:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f'{name}: blank identifier')
            kinds.add('text')
        elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
            if not np.isfinite(value):
                raise ValueError(f'{name}: nonfinite identifier')
            kinds.add('numeric')
        else:
            raise ValueError(f'{name}: missing or unsupported identifier {value!r}')
    if len(kinds) != 1:
        raise ValueError(f'{name}: mixed text/numeric identifiers require an explicit mapping')
    return values
