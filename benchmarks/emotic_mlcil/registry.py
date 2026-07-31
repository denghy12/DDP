"""Method registry with explicit names and no eager external imports."""

from __future__ import annotations

from typing import Dict, Iterable, Type

from .method_base import BenchmarkMethod


_METHODS: Dict[str, Type[BenchmarkMethod]] = {}


def register_method(name: str, method_class: Type[BenchmarkMethod]) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Method name must not be empty")
    if not issubclass(method_class, BenchmarkMethod):
        raise TypeError("Registered methods must subclass BenchmarkMethod")
    existing = _METHODS.get(normalized)
    if existing is not None and existing is not method_class:
        raise ValueError(f"Method '{normalized}' is already registered")
    _METHODS[normalized] = method_class


def method_names() -> Iterable[str]:
    return tuple(sorted(_METHODS))


def method_class(name: str) -> Type[BenchmarkMethod]:
    normalized = name.strip().lower()
    try:
        return _METHODS[normalized]
    except KeyError as exc:
        available = ", ".join(method_names()) or "<none>"
        raise KeyError(
            f"Unknown benchmark method '{name}'. Registered: {available}"
        ) from exc
