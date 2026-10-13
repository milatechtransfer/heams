from importlib import import_module


def test_heams_package_is_importable() -> None:
    module = import_module("heams")

    assert module is not None
