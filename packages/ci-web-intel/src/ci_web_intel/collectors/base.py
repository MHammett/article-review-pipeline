"""Base classes and data model for voice-profile collectors."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Document:
    text: str
    source: str
    register: str
    date: str
    url_or_id: str
    word_count: int
    content_hash: str
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, source: str, register: str, date: str,
                  url_or_id: str, metadata: dict | None = None) -> "Document":
        cleaned = text.strip()
        word_count = len(cleaned.split()) if cleaned else 0
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return cls(
            text=cleaned,
            source=source,
            register=register,
            date=date,
            url_or_id=url_or_id,
            word_count=word_count,
            content_hash=content_hash,
            metadata=metadata or {},
        )


class CollectorError(Exception):
    def __init__(self, source_name: str, message: str):
        self.source_name = source_name
        super().__init__(f"[{source_name}] {message}")


class ConfigError(Exception):
    def __init__(self, source_name: str, missing_keys: list[str] | None = None, message: str = ""):
        self.source_name = source_name
        self.missing_keys = missing_keys or []
        if missing_keys:
            msg = f"[{source_name}] Missing required config keys: {', '.join(missing_keys)}"
        else:
            msg = f"[{source_name}] {message}"
        super().__init__(msg)


class Collector(ABC):
    SOURCE_NAME: str = ""

    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def validate_config(cls, config: dict) -> None:
        """Raise ConfigError on missing required keys."""

    def estimate_count(self) -> int | None:
        return None

    @abstractmethod
    def fetch(self, since: str | None = None) -> Iterator[Document]:
        """Yield Documents. Raises CollectorError on auth failure or quota exhaustion."""
