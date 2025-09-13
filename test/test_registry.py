from src.utils.registry import Registry


def test_registry_basic():
    reg = Registry()
    @reg.register("foo")
    class A:
        pass
    assert reg.get("foo") is A
    assert "foo" in reg.available()
