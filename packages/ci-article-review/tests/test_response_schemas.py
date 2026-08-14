"""Schemas must stay in step with the prompts they transcribe."""

import json
import pathlib

import pytest

from ci_article_review import pipeline, response_schemas
from ci_core.llm import schema_format

PROMPTS = pathlib.Path(pipeline.__file__).parent / "prompts"


class TestSchemasMatchPrompts:
    @pytest.mark.parametrize("domain", sorted(response_schemas.BY_DOMAIN))
    def test_every_top_level_key_appears_in_the_prompt(self, domain):
        """The prompt's RETURN FORMAT is the contract; the schema transcribes it.

        If someone edits a prompt's shape without editing the schema, the
        provider will reject the model's own correct output. This is the guard.
        """
        text = (PROMPTS / pipeline._DOMAIN_PROMPTS[domain]).read_text(encoding="utf-8")
        for key in response_schemas.BY_DOMAIN[domain]["properties"]:
            assert f'"{key}"' in text, f"{domain}: schema has {key!r}, prompt does not"

    @pytest.mark.parametrize("domain", sorted(response_schemas.BY_DOMAIN))
    def test_schema_is_strict_compatible(self, domain):
        """OpenAI strict mode: every property required, no extra properties."""

        def check(node, path="root"):
            if node.get("type") == "object":
                props = set(node.get("properties") or {})
                assert node.get("additionalProperties") is False, path
                assert set(node.get("required") or []) == props, path
                for k, v in (node.get("properties") or {}).items():
                    check(v, f"{path}.{k}")
            elif node.get("type") == "array":
                check(node["items"], f"{path}[]")

        check(response_schemas.BY_DOMAIN[domain])

    def test_custom_domains_get_no_schema(self):
        """A publication's own prompt has no shape we can enforce."""
        assert response_schemas.for_domain("some_custom_domain") is None


class TestProviderTranslation:
    @pytest.mark.parametrize("domain", sorted(response_schemas.BY_DOMAIN))
    def test_gemini_conversion_succeeds_for_every_domain(self, domain):
        out = schema_format.gemini_response_schema(response_schemas.BY_DOMAIN[domain])
        assert out is not None, f"{domain} failed to convert"
        assert out["responseMimeType"] == "application/json"
        assert out["responseSchema"]["type"] == "OBJECT"

    def test_nullable_fields_survive_the_gemini_conversion(self):
        out = schema_format.gemini_response_schema(response_schemas.FACT_CHECK)
        note = out["responseSchema"]["properties"]["confirmed"]["items"]["properties"][
            "note"
        ]
        assert note == {"type": "STRING", "nullable": True}

    def test_openai_wrapper_is_strict(self):
        out = schema_format.openai_text_format(response_schemas.RED_TEAM)
        assert out["format"]["strict"] is True
        assert out["format"]["type"] == "json_schema"

    def test_chat_completions_wrapper_shape(self):
        out = schema_format.chat_completions_response_format(
            response_schemas.VOICE_STYLE
        )
        assert out["type"] == "json_schema"
        assert out["json_schema"]["schema"] is response_schemas.VOICE_STYLE

    def test_unconvertible_schema_returns_none_rather_than_guessing(self):
        assert (
            schema_format.gemini_response_schema({"type": ["string", "integer"]})
            is None
        )
        assert schema_format.gemini_response_schema({"oneOf": []}) is None

    def test_schemas_are_json_serialisable(self):
        for domain, schema in response_schemas.BY_DOMAIN.items():
            json.dumps(schema), domain


class TestCacheFriendlyLayout:
    def test_system_prefix_is_identical_across_domains(self):
        """That identity is the entire mechanism — if it varies, nothing caches."""
        a = pipeline._cache_friendly_layout("fact check instructions", "DRAFT")
        b = pipeline._cache_friendly_layout("red team instructions", "DRAFT")
        assert a[0] == b[0]

    def test_the_draft_leads_the_user_message(self):
        _, user = pipeline._cache_friendly_layout("TASK TEXT", "THE DRAFT")
        assert user.startswith("THE DRAFT")

    def test_the_domain_instruction_is_preserved_verbatim(self):
        """Relocated, not summarised — the model must still get all of it."""
        _, user = pipeline._cache_friendly_layout("TASK TEXT", "THE DRAFT")
        assert "TASK TEXT" in user

    def test_two_domains_share_a_byte_identical_prefix(self):
        sa, ua = pipeline._cache_friendly_layout("domain A", "SAME DRAFT")
        sb, ub = pipeline._cache_friendly_layout("domain B", "SAME DRAFT")
        shared = len(ua) - len(ua.lstrip())
        assert sa == sb
        assert ua[: len("SAME DRAFT")] == ub[: len("SAME DRAFT")] == "SAME DRAFT"
        assert shared == 0
