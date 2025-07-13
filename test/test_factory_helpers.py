from types import SimpleNamespace
from src.task_factory import resolve_task_module, register_task, TASK_REGISTRY
from src.trainer_factory import (
    resolve_trainer_module,
    register_trainer,
    TRAINER_REGISTRY,
)
from src.data_factory import (
    build_data,
    register_data_factory,
    resolve_data_factory_class,
    IDDataFactory,
)
from src.model_factory import build_model, register_model


def test_register_task():
    @register_task("x", "y")
    class Dummy:
        pass
    assert TASK_REGISTRY.get("x.y") is Dummy


def test_register_trainer():
    @register_trainer("foo")
    def build(*args, **kw):
        return 1
    assert TRAINER_REGISTRY.get("foo") is build


def test_resolve_task_module_default():
    args = SimpleNamespace(type='Default_task', name='Default_task')
    assert resolve_task_module(args) == 'src.task_factory.Default_task'


def test_resolve_trainer_module_default():
    args = SimpleNamespace(trainer_name='Default_trainer')
    assert resolve_trainer_module(args) == 'src.trainer_factory.Default_trainer'


def test_build_data_registered():
    @register_data_factory("dummy")
    class DummyFactory:
        def __init__(self, *a, **k):
            pass

    args = SimpleNamespace(factory_name="dummy")
    obj = build_data(args, SimpleNamespace())
    assert isinstance(obj, DummyFactory)


def test_build_model_registered():
    @register_model("t", "dummy")
    class DummyModel:
        def __init__(self, *a, **k):
            pass

    args = SimpleNamespace(type="t", name="dummy")
    model = build_model(args, metadata=None)
    assert isinstance(model, DummyModel)


def test_builtin_id_factory():
    from src.task_factory import TASK_REGISTRY
    cls = resolve_data_factory_class("id")
    assert cls is IDDataFactory

    import src.task_factory.ID_task as _  # ensure registration
    assert TASK_REGISTRY.get("Default_task.ID_task") is not None

