"""Tests for analysis.seo_content — the search-reader structure review.

The model call is mocked throughout, same as test_seo_suggest.py.
"""

from unittest.mock import patch

from ci_article_review.analysis import seo_content

_ARTICLE = "\n".join(
    [
        "# Interconnection Queues Decide the Timeline",
        "",
        "The queue is the constraint, not generation.",
        "",
        "## The Bigger Picture",
        "",
        " ".join(["word"] * 200),
    ]
)

_HANDOFF = {
    "title": "Interconnection Queues Decide the Timeline",
    "primary_claim": "Queues, not generation, are the constraint.",
    "target_audience": "Municipal officials.",
}

_KEYS = {"mistral": {"api_key": "k"}}

_FINDING = {
    "type": "heading",
    "target": "The Bigger Picture",
    "problem": "Could sit above any section of any article.",
    "suggestion": "How a queue position becomes a five-year wait",
}


def _model_response(data):
    return {
        "failed": False,
        "data": data,
        "model": "mistral-small-latest",
        "tokens": {"prompt": 1400, "completion": 180},
        "elapsed_seconds": 1.6,
    }


def _review(data, pub_config=None, **kwargs):
    with patch(
        "ci_article_review.analysis.seo_content.mistral.call",
        return_value=_model_response(data),
    ) as mock_call:
        result, call_log = seo_content.review(
            _ARTICLE,
            handoff=_HANDOFF,
            pub_config=pub_config if pub_config is not None else {},
            api_keys=_KEYS,
            **kwargs,
        )
    return result, call_log, mock_call


class TestReview:
    def test_findings_are_returned(self):
        result, _, _ = _review({"findings": [_FINDING]})
        assert result["status"] == "ok"
        assert result["findings"] == [_FINDING]

    def test_empty_findings_is_a_success_not_a_failure(self):
        # A structurally sound article is the expected happy path — it must not
        # look like the pass broke.
        result, call_log, _ = _review({"findings": []})
        assert result["status"] == "ok"
        assert result["findings"] == []
        assert call_log["failed"] is False

    def test_call_log_is_separately_attributable(self):
        from ci_core.llm import cost as cost_analysis

        _, call_log, _ = _review({"findings": []})
        assert call_log["pass"] == "seo_content_review"
        assert call_log["pass"] != "seo_suggestions"
        assert call_log["model"] == "mistral-small-latest"

        summary = cost_analysis.calculate([call_log])
        assert summary["pricing_known"] is True
        assert summary["by_pass"][0]["pass"] == "seo_content_review"

    def test_keyword_candidates_frame_the_search_intent(self):
        suggestions = {
            "keyword_candidates": [
                {"keyword": "interconnection queue"},
                {"keyword": "grid capacity"},
            ]
        }
        _, _, mock_call = _review({"findings": []}, suggestions=suggestions)
        prompt = mock_call.call_args.args[1]
        assert "interconnection queue" in prompt
        assert "grid capacity" in prompt

    def test_runs_without_suggestions(self):
        result, _, _ = _review({"findings": [_FINDING]}, suggestions=None)
        assert result["status"] == "ok"

    def test_uses_the_small_fast_model(self):
        _, _, mock_call = _review({"findings": []})
        assert mock_call.call_args.kwargs["model"] == "mistral-small-latest"


class TestStaysInItsLane:
    """The ensemble already covers argument, completeness, and voice. A finding
    outside this pass's three questions is dropped rather than duplicated."""

    def test_unknown_finding_type_is_dropped(self):
        result, _, _ = _review(
            {
                "findings": [
                    _FINDING,
                    {
                        "type": "completeness",
                        "problem": "No mention of the competing proposal.",
                    },
                    {"type": "voice", "problem": "Reads like AI-speak."},
                ]
            }
        )
        assert [f["type"] for f in result["findings"]] == ["heading"]

    def test_prompt_tells_the_model_to_leave_those_domains_alone(self):
        _, _, mock_call = _review({"findings": []})
        system = mock_call.call_args.args[0]
        assert "Do NOT flag missing information" in system
        assert "EMPTY findings list" in system

    def test_finding_without_a_problem_is_dropped(self):
        result, _, _ = _review(
            {"findings": [{"type": "heading", "target": "x", "problem": "  "}]}
        )
        assert result["findings"] == []

    def test_findings_are_capped(self):
        many = [{**_FINDING, "target": f"H{i}"} for i in range(20)]
        result, _, _ = _review({"findings": many})
        assert len(result["findings"]) == 8


class TestGracefulDegradation:
    def test_failed_call_still_logs_cost(self):
        with patch(
            "ci_article_review.analysis.seo_content.mistral.call",
            return_value={
                "failed": True,
                "error": "503 Service Unavailable",
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.3,
            },
        ):
            result, call_log = seo_content.review(
                _ARTICLE, handoff=_HANDOFF, pub_config={}, api_keys=_KEYS
            )

        assert result["status"] == "failed"
        assert "503" in result["reason"]
        assert call_log["pass"] == "seo_content_review"
        assert call_log["failed"] is True

    def test_raising_call_is_caught(self):
        with patch(
            "ci_article_review.analysis.seo_content.mistral.call",
            side_effect=RuntimeError("socket exploded"),
        ):
            result, call_log = seo_content.review(
                _ARTICLE, handoff=_HANDOFF, pub_config={}, api_keys=_KEYS
            )

        assert result["status"] == "failed"
        assert "socket exploded" in result["reason"]
        assert call_log is None

    def test_non_dict_payload_is_a_failure(self):
        result, call_log = None, None
        with patch(
            "ci_article_review.analysis.seo_content.mistral.call",
            return_value=_model_response(["not", "an", "object"]),
        ):
            result, call_log = seo_content.review(
                _ARTICLE, handoff=_HANDOFF, pub_config={}, api_keys=_KEYS
            )
        assert result["status"] == "failed"
        assert call_log is not None

    def test_non_list_findings_do_not_crash(self):
        result, _, _ = _review({"findings": "a big string"})
        assert result["status"] == "ok"
        assert result["findings"] == []

    def test_missing_api_key_skips_without_calling(self):
        with patch("ci_article_review.analysis.seo_content.mistral.call") as mock_call:
            result, call_log = seo_content.review(
                _ARTICLE, handoff=_HANDOFF, pub_config={}, api_keys={}
            )
        mock_call.assert_not_called()
        assert result["status"] == "skipped"
        assert call_log is None


class TestDisabling:
    def test_content_review_flag_off_skips_without_calling(self):
        with patch("ci_article_review.analysis.seo_content.mistral.call") as mock_call:
            result, call_log = seo_content.review(
                _ARTICLE,
                handoff=_HANDOFF,
                pub_config={"seo_rules": {"content_review": False}},
                api_keys=_KEYS,
            )
        mock_call.assert_not_called()
        assert result["status"] == "skipped"
        assert "disabled" in result["reason"]
        assert call_log is None

    def test_independent_of_the_suggestions_flag(self):
        # Turning off metadata suggestions alone must not silence this pass —
        # they answer different questions and each has its own key.
        with patch(
            "ci_article_review.analysis.seo_content.mistral.call",
            return_value=_model_response({"findings": [_FINDING]}),
        ) as mock_call:
            result, _ = seo_content.review(
                _ARTICLE,
                handoff=_HANDOFF,
                pub_config={"seo_rules": {"suggestions": False}},
                api_keys=_KEYS,
            )
        mock_call.assert_called_once()
        assert result["status"] == "ok"

    def test_on_by_default(self):
        _, _, mock_call = _review({"findings": []})
        mock_call.assert_called_once()
