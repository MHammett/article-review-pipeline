"""Output formatters and file writing for synthesized voice profiles."""

from __future__ import annotations

import difflib
import json
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import LiteralScalarString
    _RUAMEL_AVAILABLE = True
except ImportError:
    log.warning("ruamel.yaml not installed; falling back to PyYAML (comments in existing YAML will be lost)")
    import yaml as _pyyaml
    _RUAMEL_AVAILABLE = False


def _make_ruamel():
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    yaml.width = 120
    return yaml


def _literal(text: str) -> "LiteralScalarString":
    """Wrap a multiline string as a YAML literal block scalar."""
    return LiteralScalarString(text)


class Formatter(ABC):
    @abstractmethod
    def format(self, profile: dict, mode: str = "canonical") -> str:
        ...


class PublicationYamlFormatter(Formatter):
    """Formats the synthesized profile for embedding in publication.yaml."""

    def format(self, profile: dict, mode: str = "canonical") -> str:
        if _RUAMEL_AVAILABLE:
            return self._format_ruamel(profile, mode)
        return self._format_pyyaml(profile, mode)

    def _build_voice_block(self, profile: dict, mode: str) -> dict:
        """Build the voice section dict ready for YAML serialization."""
        canonical = profile if mode == "canonical" else profile.get("canonical", profile)

        voice_profile_text = canonical.get("voice_profile", "")
        audience_primary = canonical.get("audience_primary", "")
        audience_secondary = canonical.get("audience_secondary")
        banned_words = canonical.get("banned_words", [])
        banned_phrases = canonical.get("banned_phrases", [])
        positive_rules = canonical.get("positive_rules", [])

        block: dict = {}

        if _RUAMEL_AVAILABLE:
            block["voice_profile"] = _literal(voice_profile_text + "\n") if voice_profile_text else ""
            block["audience"] = {}
            if audience_primary:
                block["audience"]["primary"] = audience_primary
            if audience_secondary:
                block["audience"]["secondary"] = audience_secondary
        else:
            block["voice_profile"] = voice_profile_text
            block["audience"] = {}
            if audience_primary:
                block["audience"]["primary"] = audience_primary
            if audience_secondary:
                block["audience"]["secondary"] = audience_secondary

        block["style_rules"] = {
            "banned_words": banned_words,
            "banned_phrases": banned_phrases,
            "positive_rules": positive_rules,
        }

        # Add voice_profiles block for detect/per-source modes
        # INTERNAL KEY: detected_voices → EMITTED KEY: voice_profiles
        if mode in ("detect", "per-source"):
            detected = profile.get("detected_voices", {})
            voice_profiles: dict = {}
            for label, vdata in detected.items():
                vp_text = vdata.get("voice_profile", "")
                entry: dict = {}
                if _RUAMEL_AVAILABLE:
                    entry["voice_profile"] = _literal(vp_text + "\n") if vp_text else ""
                else:
                    entry["voice_profile"] = vp_text
                if vdata.get("additional_banned_words"):
                    entry["additional_banned_words"] = vdata["additional_banned_words"]
                if vdata.get("additional_positive_rules"):
                    entry["additional_positive_rules"] = vdata["additional_positive_rules"]
                if vdata.get("voice_notes"):
                    entry["voice_notes"] = vdata["voice_notes"]
                if vdata.get("source_distribution"):
                    entry["source_distribution"] = vdata["source_distribution"]
                if vdata.get("doc_count") is not None:
                    entry["doc_count"] = vdata["doc_count"]
                if vdata.get("confidence"):
                    entry["confidence"] = vdata["confidence"]
                voice_profiles[label] = entry

            if voice_profiles:
                block["voice_profiles"] = voice_profiles

        return block

    def _format_ruamel(self, profile: dict, mode: str) -> str:
        import io
        yaml = _make_ruamel()
        block = self._build_voice_block(profile, mode)
        buf = io.StringIO()
        yaml.dump(block, buf)
        return buf.getvalue()

    def _format_pyyaml(self, profile: dict, mode: str) -> str:
        block = self._build_voice_block(profile, mode)
        return _pyyaml.dump(block, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def merge_into_existing(self, existing_path: str, profile: dict, mode: str = "canonical") -> str:
        """Load existing YAML, replace voice sections, preserve all other keys."""
        existing = Path(existing_path)
        new_voice_block = self._build_voice_block(profile, mode)

        if not existing.exists():
            # Create fresh file with voice block
            return self.format(profile, mode)

        if _RUAMEL_AVAILABLE:
            return self._merge_ruamel(existing, new_voice_block)
        return self._merge_pyyaml(existing, new_voice_block)

    def _merge_ruamel(self, existing_path: Path, new_voice_block: dict) -> str:
        import io
        yaml = _make_ruamel()
        with open(existing_path, encoding="utf-8") as f:
            data = yaml.load(f) or {}

        voice_keys = ["voice_profile", "audience", "style_rules", "voice_profiles"]
        for key in voice_keys:
            if key in data:
                del data[key]

        for key, value in new_voice_block.items():
            data[key] = value

        buf = io.StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()

    def _merge_pyyaml(self, existing_path: Path, new_voice_block: dict) -> str:
        with open(existing_path, encoding="utf-8") as f:
            data = _pyyaml.safe_load(f) or {}

        voice_keys = ["voice_profile", "audience", "style_rules", "voice_profiles"]
        for key in voice_keys:
            data.pop(key, None)

        data.update(new_voice_block)
        return _pyyaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def diff_voice_sections(self, existing_path: str, new_yaml: str) -> str:
        """Return unified diff of voice sections between existing file and new YAML."""
        existing = Path(existing_path)
        if not existing.exists():
            return ""
        old_lines = existing.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = new_yaml.splitlines(keepends=True)

        def _extract_voice_lines(lines: list[str]) -> list[str]:
            voice_keys = {"voice_profile:", "audience:", "style_rules:", "voice_profiles:"}
            result = []
            in_voice = False
            for line in lines:
                stripped = line.lstrip()
                if any(stripped.startswith(k) for k in voice_keys):
                    in_voice = True
                elif not line.startswith(" ") and not line.startswith("\t") and stripped and not stripped.startswith("#"):
                    in_voice = False
                if in_voice:
                    result.append(line)
            return result

        old_voice = _extract_voice_lines(old_lines)
        new_voice = _extract_voice_lines(new_lines)
        return "".join(difflib.unified_diff(
            old_voice, new_voice,
            fromfile=f"{existing_path} (existing)",
            tofile="new profile",
        ))


class MarkdownReportFormatter(Formatter):
    """Human-readable Markdown summary of the synthesized profile."""

    def format(self, profile: dict, mode: str = "canonical") -> str:
        canonical = profile if mode == "canonical" else profile.get("canonical", profile)
        lines = ["# Voice Profile Report", ""]

        vp = canonical.get("voice_profile", "")
        if vp:
            lines += ["## Voice Profile", "", vp, ""]

        audience = canonical.get("audience") or {}
        if not audience:
            primary = canonical.get("audience_primary")
            secondary = canonical.get("audience_secondary")
        else:
            primary = audience.get("primary")
            secondary = audience.get("secondary")

        if primary or secondary:
            lines += ["## Audience", ""]
            if primary:
                lines += [f"**Primary:** {primary}", ""]
            if secondary:
                lines += [f"**Secondary:** {secondary}", ""]

        style = canonical.get("style_rules") or {}
        banned_words = style.get("banned_words") or canonical.get("banned_words", [])
        banned_phrases = style.get("banned_phrases") or canonical.get("banned_phrases", [])
        positive_rules = style.get("positive_rules") or canonical.get("positive_rules", [])

        if banned_words or banned_phrases:
            lines += ["## Banned Words & Phrases", ""]
            if banned_words:
                lines += ["| Banned Word |", "|-------------|"]
                for w in banned_words:
                    lines += [f"| {w} |"]
                lines += [""]
            if banned_phrases:
                lines += ["| Banned Phrase |", "|---------------|"]
                for p in banned_phrases:
                    lines += [f"| {p} |"]
                lines += [""]

        if positive_rules:
            lines += ["## Style Rules", ""]
            for i, rule in enumerate(positive_rules, 1):
                lines += [f"{i}. {rule}"]
            lines += [""]

        # Per-voice section
        if mode in ("detect", "per-source"):
            detected = profile.get("detected_voices") or profile.get("voice_profiles", {})
            if detected:
                lines += ["## Detected Voices", ""]
                for label, vdata in detected.items():
                    lines += [f"### {label}", ""]
                    if vdata.get("voice_profile"):
                        lines += [vdata["voice_profile"], ""]
                    if vdata.get("source_distribution"):
                        lines += ["**Source distribution:**", ""]
                        lines += ["| Source | % |", "|--------|---|"]
                        for src, pct in sorted(vdata["source_distribution"].items(), key=lambda x: -x[1]):
                            lines += [f"| {src} | {pct*100:.0f}% |"]
                        lines += [""]
                    if vdata.get("voice_notes"):
                        lines += [f"*{vdata['voice_notes']}*", ""]

        return "\n".join(lines)


class JsonFormatter(Formatter):
    """Raw JSON dump of the profile dict."""

    def format(self, profile: dict, mode: str = "canonical") -> str:
        return json.dumps(profile, indent=2, ensure_ascii=False)


def write_atomic(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write content atomically using a temp file + os.replace()."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    log.info("Wrote profile to %s", output_path)


def save_versioned_snapshot(content: str, publication: str | None, output_yaml: str | None) -> Path:
    """Save a timestamped copy of the profile."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    profiles_dir = Path(__file__).parent / "profiles"

    if publication:
        snap_dir = profiles_dir / publication
    elif output_yaml:
        stem = Path(output_yaml).stem
        snap_dir = profiles_dir / "_output" / stem
    else:
        snap_dir = profiles_dir / "_unknown"

    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{ts}.yaml"
    write_atomic(snap_path, content)
    log.info("Versioned snapshot: %s", snap_path)
    return snap_path
