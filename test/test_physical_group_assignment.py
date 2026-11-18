"""Deterministic physical-group isolation contract for the shared splitting module.

These tests pin the sorted-identifier-then-PCG64 convention shared with
resolve_data_splits so record-order changes or RNG-consumption changes cannot
silently alter predeclared role inventories.
"""
import numpy as np
import pytest

from src.data_factory.splitting import assign_physical_groups
from src.data_factory.tii_sampling import assign_physical_groups as reexported


ALLOCATION = [('source_train', 7), ('source_val', 3)]


def _groups():
    return [f'bearing-{i:02d}' for i in range(10) for _ in range(2)]


def test_import_paths_expose_one_implementation():
    assert reexported is assign_physical_groups


def test_same_seed_same_roles_and_disjoint_coverage():
    roles = assign_physical_groups(_groups(), ALLOCATION, seed=7)
    assert len(roles) == 20
    train = {g for g, r in zip(_groups(), roles) if r == 'source_train'}
    val = {g for g, r in zip(_groups(), roles) if r == 'source_val'}
    assert len(train) == 7 and len(val) == 3
    assert not train & val
    assert train | val == set(_groups())


def test_record_order_does_not_change_group_roles():
    groups = _groups()
    roles = assign_physical_groups(groups, ALLOCATION, seed=7)
    reversed_roles = assign_physical_groups(groups[::-1], ALLOCATION, seed=7)
    by_group = dict(zip(groups, roles))
    assert [by_group[g] for g in groups[::-1]] == reversed_roles


def test_explicit_rng_sequence_is_sorted_unique_pcg64():
    groups = _groups()
    roles = assign_physical_groups(groups, ALLOCATION, seed=7)
    unique = sorted(set(groups))
    permutation = np.random.Generator(np.random.PCG64(7)).permutation(len(unique))
    expected = {}
    start = 0
    for role, count in ALLOCATION:
        for index in permutation[start:start + count]:
            expected[unique[int(index)]] = role
        start += count
    assert [expected[g] for g in groups] == roles


@pytest.mark.parametrize('allocation', [
    [('source_train', 6), ('source_val', 3)],
    [('source_train', 7), ('source_val', 3), ('unused', 1)],
    [('source_train', 7), ('source_train', 3)],
])
def test_inexact_or_duplicate_coverage_rejected(allocation):
    with pytest.raises(ValueError):
        assign_physical_groups(_groups(), allocation, seed=7)


def test_seed_must_be_nonnegative_integer():
    for bad in (-1, True, 1.5, None):
        with pytest.raises(ValueError):
            assign_physical_groups(_groups(), ALLOCATION, seed=bad)
