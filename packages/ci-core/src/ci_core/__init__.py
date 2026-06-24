"""ci-core: shared library for CI tools."""

from .llm import Adapter, AdapterError, AdapterFactory

__all__ = ["Adapter", "AdapterError", "AdapterFactory"]
