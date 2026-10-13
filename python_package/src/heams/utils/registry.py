"""A simple registry system for managing classes in the project."""

REGISTRY = {}


def register(name: str):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator
