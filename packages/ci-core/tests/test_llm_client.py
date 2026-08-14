"""Unit tests for the litellm shim in ci_core.llm.client.

What is tested here is the seam, not litellm. litellm's own suite covers SSE
framing, provider auth, and error classes; re-testing those through a mock would
only assert that our mock matches our assumptions. What this file covers is
everything that would break silently if the shim mapped something wrong:

  * the read-gap timeout — the single most load-bearing knob in the layer, and
    the one whose regression looks like "the provider got slow" rather than
    like a bug;
  * ``num_retries=0`` — because litellm calls credit exhaustion a rate limit,
    and its retry loop would sit on a dead account until the run's wall clock
    ran out;
  * OpenAI on ``responses()`` — measured at 79s of total silence on
    ``completion()``, which the read-gap timeout cannot survive;
  * the provider extras the pipeline reads back out (citations, grounding
    chunks, cached tokens, truncation), each of which fails by going quietly
    empty rather than by raising.
"""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from ci_core.llm import client


# ---------------------------------------------------------------------------
# litellm-shaped mock streams
# ---------------------------------------------------------------------------


def _delta(content=None, finish_reason=None):
    return SimpleNamespace(
        delta=SimpleNamespace(content=content), finish_reason=finish_reason
    )


def _usage(prompt=100, completion=50, cached=None, anthropic_cached=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=details,
        cache_read_input_tokens=anthropic_cached,
    )


def _chunk(content=None, finish_reason=None, usage=None, **extras):
    """One streamed chunk. Absent attributes must read as None, as litellm's do."""
    base = {
        "choices": [_delta(content, finish_reason)] if content or finish_reason else [],
        "usage": usage,
        "citations": None,
        "search_results": None,
        "vertex_ai_grounding_metadata": None,
    }
    base.update(extras)
    return SimpleNamespace(**base)


def _completion_stream(
    text='{"flags": []}', finish_reason="stop", usage=None, **extras
):
    """A completion stream that emits ``text`` one chunk at a time, then usage."""
    chunks = [_chunk(content=piece) for piece in text]
    chunks.append(_chunk(finish_reason=finish_reason))
    chunks.append(_chunk(usage=usage if usage is not None else _usage(), **extras))
    return chunks


def _responses_stream(text='{"flags": []}', status="completed", usage=None):
    """An OpenAI Responses API event stream."""
    events = [SimpleNamespace(type="response.created")]
    events.append(
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thinking…")
    )
    for piece in text:
        events.append(SimpleNamespace(type="response.output_text.delta", delta=piece))
    events.append(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=usage if usage is not None else _usage(),
                status=status,
                incomplete_details=None,
            ),
        )
    )
    return events


def _http_error(status, body=""):
    """An exception shaped like the ones litellm raises for HTTP failures."""
    exc = Exception(f"HTTP {status}")
    exc.status_code = status
    exc.response = SimpleNamespace(status_code=status, text=body)
    return exc


def _call(provider="mistral", **kwargs):
    defaults = dict(
        system_prompt="sys",
        user_prompt="user",
        api_key="key",
        retry=False,
        retry_delay=0,
    )
    defaults.update(kwargs)
    return client.call(provider, **defaults)


# ---------------------------------------------------------------------------
# The read-gap timeout
# ---------------------------------------------------------------------------


class TestReadGapTimeout:
    """The read timeout is the gap BETWEEN chunks, not the generation budget.

    Everything about the layer's tolerance for slow models rests on this. If it
    ever silently becomes a total-time bound, healthy long generations start
    dying at the ceiling and it reads like a provider regression.
    """

    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openai", 120),
            ("mistral", 120),
            ("grok", 120),
            ("claude", 120),
            # Grounded providers search before the first token arrives.
            ("gemini", 160),
            ("perplexity", 160),
        ],
    )
    def test_provider_default_read_gap(self, provider, expected):
        timeout = client._stream_timeout(
            None, client._PROVIDERS[provider]["read_timeout"]
        )
        assert timeout.read == expected

    @pytest.mark.parametrize("provider", client.PROVIDERS)
    def test_stream_read_timeout_override_is_honored(self, provider):
        """The per-model override in presets.yaml must reach the socket.

        Every value in presets.yaml (mistral 200, perplexity 500, gemini 260)
        was set in response to a real production timeout. An override that
        stopped being plumbed through would re-open every one of those.
        """
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _responses_stream() if provider == "openai" else _completion_stream()

        target = "responses" if provider == "openai" else "completion"
        with patch.object(client.litellm, target, side_effect=_capture):
            _call(provider, provider_config={"stream_read_timeout": 222})

        assert isinstance(seen["timeout"], httpx.Timeout)
        assert seen["timeout"].read == 222, (
            f"{provider} ignored stream_read_timeout — got {seen['timeout'].read}"
        )

    def test_read_gap_is_not_derived_from_timeout_seconds(self):
        """``timeout_seconds`` is the pipeline's wall-clock backstop, not a socket
        timeout. Feeding it to the socket would kill long healthy generations."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("mistral", provider_config={"timeout_seconds": 900})

        assert seen["timeout"].read == 120

    def test_connect_timeout_stays_small_and_constant(self):
        timeout = client._stream_timeout({"stream_read_timeout": 500}, 120)
        assert timeout.connect == 30
        assert timeout.read == 500


# ---------------------------------------------------------------------------
# Call surfaces and routing
# ---------------------------------------------------------------------------


class TestCallSurface:
    def test_openai_uses_responses_not_completion(self):
        """Measured: completion() sends zero bytes for 79s on a reasoning model.

        The read-gap timeout cannot tell that apart from a hung socket, so this
        is a correctness constraint, not a preference.
        """
        with (
            patch.object(
                client.litellm, "responses", return_value=_responses_stream()
            ) as responses,
            patch.object(client.litellm, "completion") as completion,
        ):
            result = _call("openai")

        assert responses.called
        assert not completion.called
        assert result["failed"] is False

    @pytest.mark.parametrize(
        "provider", ["gemini", "mistral", "grok", "claude", "perplexity"]
    )
    def test_other_providers_use_completion(self, provider):
        with (
            patch.object(
                client.litellm, "completion", return_value=_completion_stream()
            ) as completion,
            patch.object(client.litellm, "responses") as responses,
        ):
            _call(provider)

        assert completion.called
        assert not responses.called

    @pytest.mark.parametrize(
        "provider,model,expected",
        [
            ("gemini", "gemini-2.5-pro", "gemini/gemini-2.5-pro"),
            ("grok", "grok-4.3", "xai/grok-4.3"),
            ("claude", "claude-opus-4-8", "anthropic/claude-opus-4-8"),
            ("perplexity", "sonar-pro", "perplexity/sonar-pro"),
            ("mistral", "mistral-large-latest", "mistral/mistral-large-latest"),
            # OpenAI is litellm's default route and takes no prefix.
            ("openai", "gpt-5.4", "gpt-5.4"),
        ],
    )
    def test_model_ids_get_litellm_provider_prefix(self, provider, model, expected):
        assert client._qualified(provider, model) == expected

    def test_already_qualified_model_is_left_alone(self):
        """So an operator can pin an exact litellm route in user.yaml."""
        assert client._qualified("gemini", "vertex_ai/gemini-2.5-pro") == (
            "vertex_ai/gemini-2.5-pro"
        )

    def test_unknown_provider_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown provider"):
            _call("not-a-provider")


class TestRetryPolicy:
    def test_num_retries_zero_is_always_passed(self):
        """litellm reports credit exhaustion as a RateLimitError.

        Left to its own retry loop it would treat a dead account as a transient
        limit and burn the wall-clock budget rediscovering that.
        """
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("grok")

        assert seen["num_retries"] == 0

    def test_transient_status_retried_once(self):
        attempts = []

        def _flaky(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise _http_error(429)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_flaky):
            result = _call("mistral", retry=True, retry_delay=0)

        assert len(attempts) == 2
        assert result["failed"] is False

    def test_non_transient_status_not_retried(self):
        """Retrying a 401 spends the budget twice to learn the same thing."""
        attempts = []

        def _always_401(**kwargs):
            attempts.append(1)
            raise _http_error(401, '{"error": "invalid api key"}')

        with patch.object(client.litellm, "completion", side_effect=_always_401):
            result = _call("mistral", retry=True, retry_delay=0)

        assert len(attempts) == 1
        assert result["failed"] is True


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


class TestResultContract:
    def test_successful_call_shape(self):
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream('{"flags": [{"passage": "a"}]}'),
        ):
            result = _call("mistral", model="mistral-large-latest")

        assert result["failed"] is False
        assert result["data"] == {"flags": [{"passage": "a"}]}
        assert result["raw"] == '{"flags": [{"passage": "a"}]}'
        assert result["model"] == "mistral-large-latest"
        assert result["tokens"] == {"prompt": 100, "completion": 50, "cached": 0}
        assert isinstance(result["elapsed_seconds"], float)

    def test_malformed_json_is_a_failure_that_keeps_the_text(self):
        """The review pipeline needs structured findings, so prose is a failure —
        but call_text() rescues the text, so it must survive on ``raw``."""
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream("I'd be happy to help!"),
        ):
            result = _call("mistral")

        assert result["failed"] is True
        assert result["error"] == "Malformed JSON response"
        assert result["raw"] == "I'd be happy to help!"

    def test_missing_usage_does_not_fail_the_call(self):
        """A missing token count must not turn a usable response into a failure."""
        no_usage = [_chunk(content='{"flags": []}'), _chunk(finish_reason="stop")]
        with patch.object(client.litellm, "completion", return_value=no_usage):
            result = _call("mistral")

        assert result["failed"] is False
        assert result["tokens"] == {"prompt": 0, "completion": 0, "cached": 0}

    def test_error_body_is_captured(self):
        """A bare 401 status cannot distinguish an invalid key from a revoked one
        or from an account out of credit. Providers put that in the body."""
        with patch.object(
            client.litellm,
            "completion",
            side_effect=_http_error(
                401, '{"error": {"message": "insufficient credit"}}'
            ),
        ):
            result = _call("mistral")

        assert result["failed"] is True
        assert "insufficient credit" in result["error_body"]

    def test_api_key_in_error_url_is_redacted(self):
        """Gemini carries the API key as a URL query parameter, so the key lands
        in the exception text — and from there in a log or a report."""
        exc = _http_error(400, "POST https://x.googleapis.com/v1?key=SECRET123 failed")
        exc.args = ("https://x.googleapis.com/v1beta?key=SECRET123 returned 400",)

        with patch.object(client.litellm, "completion", side_effect=exc):
            result = _call("gemini")

        assert "SECRET123" not in result["error"]
        assert "SECRET123" not in result["error_body"]
        assert "[REDACTED]" in result["error"]


class TestTokens:
    def test_cached_tokens_read_from_details(self):
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream(usage=_usage(cached=800)),
        ):
            result = _call("grok")

        assert result["tokens"]["cached"] == 800

    def test_anthropic_cache_key_is_read(self):
        """Anthropic reports cache hits on its own key rather than in the
        details object; litellm passes the name straight through."""
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream(usage=_usage(anthropic_cached=640)),
        ):
            result = _call("claude")

        assert result["tokens"]["cached"] == 640

    def test_no_cache_reports_zero_not_none(self):
        """cost.calculate multiplies this; None would raise mid-report."""
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            result = _call("mistral")

        assert result["tokens"]["cached"] == 0


class TestTruncation:
    def test_finish_reason_length_marks_truncated(self):
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream('{"flags": []}', finish_reason="length"),
        ):
            result = _call("mistral")

        assert result["failed"] is False
        assert result["truncated"] is True

    def test_salvage_is_reachable_through_the_shim(self):
        """json_utils' salvage recovers complete array elements from a response
        cut off at the output-token ceiling. litellm has no equivalent, so the
        shim has to route through it — an easy thing to drop and never notice,
        because the call just starts failing as 'malformed JSON'."""
        cut_off = '{"flags": [{"passage": "one"}, {"passage": "two"}, {"pass'
        with patch.object(
            client.litellm,
            "completion",
            return_value=_completion_stream(cut_off, finish_reason="length"),
        ):
            result = _call("mistral")

        assert result["failed"] is False
        assert result["truncated"] is True
        assert result["data"] == {"flags": [{"passage": "one"}, {"passage": "two"}]}

    def test_responses_incomplete_maps_to_truncated(self):
        """The Responses API reports the ceiling as an incomplete status with
        reason max_output_tokens, not as finish_reason='length'."""
        events = _responses_stream()
        events[-1].response.status = "incomplete"
        events[-1].response.incomplete_details = SimpleNamespace(
            reason="max_output_tokens"
        )

        with patch.object(client.litellm, "responses", return_value=events):
            result = _call("openai")

        assert result["truncated"] is True

    def test_clean_response_not_marked_truncated(self):
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            result = _call("mistral")

        assert "truncated" not in result


# ---------------------------------------------------------------------------
# Provider extras
# ---------------------------------------------------------------------------


class TestPerplexityExtras:
    def test_citations_and_search_results_surface(self):
        stream = _completion_stream(
            citations=["https://epa.gov/a", "https://nasa.gov/b"],
            search_results=[{"title": "A", "url": "https://epa.gov/a"}],
        )
        with patch.object(client.litellm, "completion", return_value=stream):
            result = _call("perplexity")

        assert result["citations"] == ["https://epa.gov/a", "https://nasa.gov/b"]
        assert result["search_results"] == [{"title": "A", "url": "https://epa.gov/a"}]
        assert result["grounding_available"] is True

    def test_no_citations_reports_grounding_unavailable(self):
        """Section 9 of the report reads this; a silent True would claim
        grounded sources that were never fetched."""
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            result = _call("perplexity")

        assert result["citations"] == []
        assert result["grounding_available"] is False


class TestGeminiGrounding:
    _META = {
        "groundingChunks": [
            {
                "web": {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ1",
                    "title": "epa.gov",
                }
            },
            {
                "web": {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ2",
                    "title": "harvard.edu",
                }
            },
        ]
    }

    def test_grounding_chunks_extracted_in_resolver_shape(self):
        """``[{"uri", "title"}, ...]`` is a contract: resolve_grounding_urls
        consumes exactly this, and every uri is a redirect wrapper that expires
        in ~30 days, so it must be resolved before anything stores one."""
        stream = _completion_stream(vertex_ai_grounding_metadata=[self._META])
        with patch.object(client.litellm, "completion", return_value=stream):
            result = _call("gemini")

        assert result["grounding_chunks"] == [
            {
                "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ1",
                "title": "epa.gov",
            },
            {
                "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ2",
                "title": "harvard.edu",
            },
        ]
        assert result["grounding_available"] is True

    def test_last_non_empty_metadata_wins(self):
        """Verified against a live call: Gemini emits an empty {} metadata
        object on an early chunk and the populated one last. Keeping the first
        non-None value yields no sources at all on a call that really did
        ground — and it fails silently, as an empty list."""
        chunks = _completion_stream()
        chunks.insert(0, _chunk(vertex_ai_grounding_metadata=[{}]))
        chunks.append(_chunk(vertex_ai_grounding_metadata=[self._META]))

        with patch.object(client.litellm, "completion", return_value=chunks):
            result = _call("gemini")

        assert len(result["grounding_chunks"]) == 2

    def test_ungrounded_call_reports_no_chunks(self):
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            result = _call("gemini")

        assert result["grounding_chunks"] == []
        assert result["grounding_available"] is False

    def test_search_grounding_tool_is_requested(self):
        """Without it Gemini answers fact_check from training recall, which is
        the entire reason it is in that ensemble."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("gemini")

        assert seen["tools"] == [{"googleSearch": {}}]


# ---------------------------------------------------------------------------
# Operator-facing guidance
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_capacity_error_falls_back_and_says_so(self):
        """A quietly degraded review is worse than a failed one: it still
        produces findings, and they still look authoritative."""
        calls = []

        def _first_model_at_capacity(**kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise _http_error(503, "model overloaded")
            return _completion_stream()

        with patch.object(
            client.litellm, "completion", side_effect=_first_model_at_capacity
        ):
            result = _call("claude", model="claude-opus-4-8")

        assert result["failed"] is False
        assert result["fallback_from"] == "claude-opus-4-8"
        assert result["model"] == "claude-sonnet-4-6"

    def test_successful_primary_reports_no_fallback(self):
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            result = _call("claude", model="claude-opus-4-8")

        assert "fallback_from" not in result

    def test_non_capacity_failure_does_not_walk_the_chain(self):
        """A bad key fails on every model; trying three of them just triples
        the time to the same answer."""
        calls = []

        def _bad_key(**kwargs):
            calls.append(kwargs["model"])
            raise _http_error(401, "invalid key")

        with patch.object(client.litellm, "completion", side_effect=_bad_key):
            result = _call("claude")

        assert len(calls) == 1
        assert result["failed"] is True


class TestMisconfigurationRetry:
    def test_reasoning_rejection_retries_without_it_and_warns(self):
        """The preset asked for a reasoning model and user.yaml overrode it with
        one that cannot reason. Run the domain anyway, but say so — the output
        is degraded and it is the config that needs fixing, not the run."""
        calls = []

        def _reject_reasoning(**kwargs):
            calls.append(kwargs)
            if "reasoning_effort" in kwargs:
                raise _http_error(
                    400,
                    '{"error": {"code": "unknown_parameter", "param": "reasoning"}}',
                )
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_reject_reasoning):
            result = _call("mistral", provider_config={"reasoning_effort": "high"})

        assert result["failed"] is False
        assert "[MISCONFIGURATION]" in result["misconfiguration_warning"]
        assert "reasoning_effort" in result["misconfiguration_warning"]
        assert len(calls) == 2

    def test_unrelated_400_does_not_trigger_the_retry(self):
        calls = []

        def _other_400(**kwargs):
            calls.append(kwargs)
            raise _http_error(400, '{"error": {"message": "context length exceeded"}}')

        with patch.object(client.litellm, "completion", side_effect=_other_400):
            result = _call("mistral", provider_config={"reasoning_effort": "high"})

        assert result["failed"] is True
        assert "misconfiguration_warning" not in result
        assert len(calls) == 1


class TestReasoningParameters:
    def test_openai_requests_reasoning_summary(self):
        """summary="auto" is what makes the thinking phase audible on the wire.
        Without it this surface goes as quiet as Chat Completions, and the whole
        reason for using responses() is gone."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _responses_stream()

        with patch.object(client.litellm, "responses", side_effect=_capture):
            _call("openai", provider_config={"reasoning_effort": "xhigh"})

        assert seen["reasoning"] == {"effort": "xhigh", "summary": "auto"}

    def test_openai_without_reasoning_sends_temperature(self):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _responses_stream()

        with patch.object(client.litellm, "responses", side_effect=_capture):
            _call("openai")

        assert "reasoning" not in seen
        assert seen["temperature"] == 0.2

    def test_claude_effort_maps_to_reasoning_effort(self):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("claude", provider_config={"effort": "high"})

        assert seen["reasoning_effort"] == "high"

    def test_gemini_thinking_budget_is_forwarded(self):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("gemini", provider_config={"thinking_budget": 16000})

        assert seen["thinking"] == {"type": "enabled", "budget_tokens": 16000}

    def test_unknown_config_keys_are_not_forwarded(self):
        """A stray key reaching a provider is a 400 mid-run, which costs a whole
        domain's review — so the mapping is an allowlist, not a passthrough."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call(
                "mistral",
                provider_config={
                    "model": "mistral-medium-3-5",
                    "timeout_seconds": 300,
                    "stream_read_timeout": 200,
                    "enabled": True,
                    "prompts": ["fact_check"],
                },
            )

        for leaked in ("timeout_seconds", "stream_read_timeout", "enabled", "prompts"):
            assert leaked not in seen


class TestMistralChunkedContent:
    def test_typed_content_chunks_keep_only_text(self):
        """Mistral's reasoning models stream content as a list of typed chunks
        rather than a string; the thinking parts are not part of the answer."""
        chunks = [
            _chunk(
                content=[
                    {"type": "thinking", "text": "Let me consider the schema…"},
                    {"type": "text", "text": '{"flags": []}'},
                ]
            ),
            _chunk(finish_reason="stop"),
            _chunk(usage=_usage()),
        ]
        with patch.object(client.litellm, "completion", return_value=chunks):
            result = _call("mistral")

        assert result["raw"] == '{"flags": []}'
        assert result["data"] == {"flags": []}


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
#
# These three cases were found by running 25 concurrent calls against real
# providers, not by reading the code. Each one arrives with a status code that
# means something different from what it says.


class TestFailureClassification:
    def test_exhausted_account_is_not_retried(self):
        """Measured on litellm 1.96.2: an OpenAI account with no credits raises
        RateLimitError with status 429 — identical by status to a per-minute
        limit a short wait would clear. Waiting does not refill a wallet."""
        attempts = []

        def _dead_account(**kwargs):
            attempts.append(1)
            raise _http_error(
                429,
                "You have no credits remaining. Add credits to continue using the API.",
            )

        with patch.object(client.litellm, "completion", side_effect=_dead_account):
            result = _call("mistral", retry=True, retry_delay=0)

        assert len(attempts) == 1, "retried an exhausted account"
        assert result["failed"] is True

    def test_genuine_rate_limit_is_still_retried(self):
        """The guard must not swallow real rate limits — those do clear."""
        attempts = []

        def _throttled(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise _http_error(429, "Request rate limit exceeded, try again later.")
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_throttled):
            result = _call("perplexity", retry=True, retry_delay=0)

        assert len(attempts) == 2
        assert result["failed"] is False

    def test_dropped_keepalive_connection_is_retried(self):
        """The failure this retry exists for. A cached HTTP client handing back
        a connection the provider already closed surfaces as status 500 with
        WinError 10038; it hit ~10% of same-host calls spaced a second or two
        apart, which is exactly this pipeline's spacing. A new socket fixes it."""
        attempts = []

        def _stale_socket(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise _http_error(
                    500,
                    "[WinError 10038] An operation was attempted on something "
                    "that is not a socket",
                )
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_stale_socket):
            result = _call("mistral", retry=True, retry_delay=0)

        assert len(attempts) == 2
        assert result["failed"] is False

    def test_dropped_connection_does_not_walk_the_fallback_chain(self):
        """litellm wraps mid-stream failures in MidStreamFallbackError, which
        synthesises status 503 when the underlying error carries none. Matching
        "503" in the message would read a dropped socket as "model at capacity"
        and silently downgrade to a weaker model over a network blip."""
        calls = []

        def _mid_stream_drop(**kwargs):
            calls.append(kwargs["model"])
            raise _http_error(
                500, "MidStreamFallbackError: APIConnectionError - connection reset"
            )

        with patch.object(client.litellm, "completion", side_effect=_mid_stream_drop):
            result = _call("claude", model="claude-opus-4-8", retry=False)

        assert len(calls) == 1, (
            f"walked the fallback chain on a connection error: {calls}"
        )
        assert result["failed"] is True
        assert "fallback_from" not in result

    def test_internal_status_is_not_leaked_to_callers(self):
        """`_status` is a routing detail; the report renders the result dict."""
        with patch.object(
            client.litellm, "completion", side_effect=_http_error(401, "bad key")
        ):
            failure = _call("mistral")
        with patch.object(
            client.litellm, "completion", return_value=_completion_stream()
        ):
            success = _call("mistral")

        for key in ("_status", "_terminal"):
            assert key not in failure
            assert key not in success

    def test_exhausted_account_is_not_retried_when_wrapped_as_503(self):
        """Mid-stream, litellm wraps the same dead account in
        MidStreamFallbackError, which synthesises status 503. Guarding only 429
        lets the wrapped form through and spends a retry_delay per call — 30 of
        them on a maximum run, all to relearn that the wallet is empty."""
        attempts = []

        def _wrapped_dead_account(**kwargs):
            attempts.append(1)
            raise _http_error(
                503,
                "MidStreamFallbackError: litellm.APIError: You have no credits "
                "remaining. Add credits to continue using the API.",
            )

        with patch.object(
            client.litellm, "completion", side_effect=_wrapped_dead_account
        ):
            result = _call("mistral", retry=True, retry_delay=0)

        assert len(attempts) == 1, (
            "a wrapped exhausted-account error was retried or walked the "
            f"fallback chain — {len(attempts)} attempts"
        )
        assert result["failed"] is True

    def test_genuine_503_capacity_error_is_still_retried(self):
        """The guard is content-based, so a real overload must still retry."""
        attempts = []

        def _overloaded(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise _http_error(503, "model is overloaded, try again")
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_overloaded):
            result = _call("claude", retry=True, retry_delay=0)

        assert len(attempts) == 2
        assert result["failed"] is False


class TestTemperature:
    """Anthropic rejects any temperature but 1 on its reasoning models.

    Sending 0.2 to claude-opus-4-8 is a hard 400 — it broke every Claude call in
    the ensemble on the first real run after the migration, in 0.07s, before a
    single token. The adapter this replaced never sent a temperature to
    Anthropic; this keeps that.
    """

    @pytest.mark.parametrize("provider", ["gemini", "mistral", "grok", "perplexity"])
    def test_providers_that_accept_temperature_get_it(self, provider):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call(provider)

        assert seen["temperature"] == 0.2

    def test_claude_is_sent_no_temperature(self):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("claude", provider_config={"effort": "high"})

        assert "temperature" not in seen, (
            "claude-opus-4-8 rejects temperature != 1 with a 400 before any token"
        )

    def test_openai_reasoning_path_sends_no_temperature(self):
        """temperature is incompatible with reasoning on the Responses API."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _responses_stream()

        with patch.object(client.litellm, "responses", side_effect=_capture):
            _call("openai", provider_config={"reasoning_effort": "high"})

        assert "temperature" not in seen


class TestMistralReasoningEscapeHatch:
    """litellm's allowlist is stricter than Mistral's actual API.

    ``reasoning_effort`` on mistral-medium-3-5 raises UnsupportedParamsError
    client-side, in 0.05s, before any request goes out — it failed all five
    domains on the first real maximum-preset run. The parameter is genuinely
    supported: the model accepts "high" and "none" and 400s on "low"/"medium",
    a distinction only the provider could be drawing.
    """

    def test_reasoning_effort_is_allowed_through(self):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("mistral", provider_config={"reasoning_effort": "high"})

        assert seen["reasoning_effort"] == "high"
        assert seen["allowed_openai_params"] == ["reasoning_effort"]

    def test_no_escape_hatch_when_no_reasoning_requested(self):
        """Nothing to allow through, so nothing is asked for."""
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return _completion_stream()

        with patch.object(client.litellm, "completion", side_effect=_capture):
            _call("mistral")

        assert "allowed_openai_params" not in seen

    def test_drop_params_stays_off_globally(self):
        """drop_params=True would make this call succeed by silently discarding
        reasoning — a quiet quality regression instead of a loud failure — and
        it is global, so it would mask the next mismatch too."""
        assert client.litellm.drop_params is False
