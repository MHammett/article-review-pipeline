"""ci_core.llm — LLM adapter abstraction."""

from ._base import Adapter, AdapterError
from ._factory import AdapterFactory

__all__ = ["Adapter", "AdapterError", "AdapterFactory"]
