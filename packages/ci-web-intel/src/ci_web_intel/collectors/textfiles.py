"""Text file and Markdown collector."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from collectors.base import Collector, CollectorError, ConfigError, Document

log = logging.getLogger(__name__)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def _strip_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        log.warning("python-docx not installed; skipping %s", path)
        return ""
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return ""


class TextFilesCollector(Collector):
    SOURCE_NAME = "textfiles"

    REQUIRED_KEYS = ["path"]

    @classmethod
    def validate_config(cls, config: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ConfigError(cls.SOURCE_NAME, missing_keys=missing)

    def fetch(self, since: str | None = None) -> Iterator[Document]:
        base_path = Path(self.config["path"]).expanduser()
        if not base_path.exists():
            raise CollectorError(self.SOURCE_NAME, f"Path does not exist: {base_path}")

        glob_pattern = self.config.get("glob", "**/*.{txt,md}")
        register = self.config.get("register", "long_form")
        max_files = self.config.get("max_files")

        # Expand brace patterns manually since pathlib doesn't support them
        if "{" in glob_pattern:
            exts = re.findall(r"\{([^}]+)\}", glob_pattern)
            base_glob = re.sub(r"\{[^}]+\}", "*", glob_pattern)
            allowed_exts = set()
            for ext_group in exts:
                for ext in ext_group.split(","):
                    allowed_exts.add("." + ext.strip().lstrip("."))
            candidates = list(base_path.glob(base_glob))
            if allowed_exts:
                candidates = [p for p in candidates if p.suffix.lower() in allowed_exts]
        else:
            candidates = list(base_path.glob(glob_pattern))

        if max_files:
            candidates = candidates[:max_files]

        for path in sorted(candidates):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".docx":
                    text = _strip_docx(path)
                elif suffix == ".md":
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    text = _strip_markdown(raw)
                else:
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                log.warning("TextFiles: failed to read %s: %s", path, e)
                continue

            if not text.strip():
                continue

            mtime = path.stat().st_mtime
            import datetime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

            doc = Document.from_text(
                text=text,
                source=self.SOURCE_NAME,
                register=register,
                date=date_str,
                url_or_id=str(path.absolute()),
                metadata={
                    "file_path": str(path.absolute()),
                    "file_size": path.stat().st_size,
                },
            )
            yield doc
