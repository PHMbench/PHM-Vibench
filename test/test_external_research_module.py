"""Downstream task selection must not require a framework-owned paper registry."""
import sys
from types import ModuleType
import pytest
from phmfactory.commands import research


def settings(module):
    return {'task': {'type': 'classification', 'name': 'external',
                     'execution': 'research', 'module': module}}


def test_explicit_downstream_module_is_selected_without_executing(monkeypatch):
    module = ModuleType('downstream_experiment')
    def execute(*args, **kwargs):
        pytest.fail('Resolution must not execute an experiment')
    module.execute = execute
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert research.task_module(settings(module.__name__)) is module


@pytest.mark.parametrize('value', [None, '', '../experiment', 'file.py:execute', 'module..name', 7])
def test_invalid_module_never_falls_back(value):
    with pytest.raises(ValueError, match='dotted Python module'):
        research.task_module(settings(value))


def test_missing_explicit_module_is_not_replaced_by_builtin():
    with pytest.raises(ModuleNotFoundError):
        research.task_module(settings('uninstalled_downstream_experiment'))


def test_downstream_module_cannot_enter_ordinary_training():
    with pytest.raises(ValueError, match='research'):
        research.reject_scientific_factory_task(settings('downstream_experiment'))
