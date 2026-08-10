"""Tests for analysis.seo_suggest — the SEO suggestion pass.

The model call is mocked everywhere, following test_resolver.py's treatment of
the citation relevance verifier this pass is modelled on.
"""

from unittest.mock import patch

from ci_article_review.analysis import seo as seo_analysis
from ci_article_review.analysis import seo_suggest

_ARTICLE = "\n".join(
    [
        "# Data Centers and the Grid",
        "",
        "Opening paragraph that frames the piece.",
        "",
        "## What the load actually looks like",
        "",
        " ".join(["word"] * 400),
        "",
        "### A sub-point",
        "",
        "Closing paragraph.",
    ]
)

_HANDOFF = {
    "title": "Data Centers and the Grid",
    "primary_claim": "Interconnection queues, not generation, are the constraint.",
    "target_audience": "Municipal officials evaluating data center proposals.",
}

_PUB_CONFIG = {
    "publication_description": "Infrastructure policy for Northern Illinois.",
    "audience": {"primary": "Local officials and network operators."},
}

_KEYS = {"mistral": {"api_key": "k"}}


def _model_response(data, **overrides):
    """A successful adapter result dict, in the shape mistral.call returns."""
    result = {
        "failed": False,
        "data": data,
        "model": "mistral-small-latest",
        "tokens": {"prompt": 900, "completion": 120},
        "elapsed_seconds": 1.2,
    }
    result.update(overrides)
    return result


_GOOD_DATA = {
    "keyword_candidates": [
        {
            "keyword": "data center interconnection queue",
            "rationale": "what officials search",
        },
        {"keyword": "grid capacity data centers", "rationale": "broader intent"},
        {"keyword": "northern illinois data center power", "rationale": "local intent"},
    ],
    "meta_description": "Interconnection queues, not generation capacity, decide how fast a data center can connect.",
}


class TestGenerateHappyPath:
    def test_returns_candidates_and_measured_description(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ):
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["status"] == "ok"
        assert [c["keyword"] for c in suggestions["keyword_candidates"]] == [
            "data center interconnection queue",
            "grid capacity data centers",
            "northern illinois data center power",
        ]
        assert all(c["rationale"] for c in suggestions["keyword_candidates"])
        assert suggestions["meta_description"] == _GOOD_DATA["meta_description"]
        assert suggestions["meta_description_chars"] == len(
            _GOOD_DATA["meta_description"]
        )
        assert suggestions["meta_description_over_limit"] is False

    def test_call_log_entry_is_attributable_and_priced(self):
        from ci_core.llm import cost as cost_analysis

        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ):
            _, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert call_log["pass"] == "seo_suggestions"
        assert call_log["model"] == "mistral-small-latest"
        assert call_log["failed"] is False
        assert call_log["tokens"] == {"prompt": 900, "completion": 120}

        # The entry has to price like any other api_call_log entry, which is the
        # whole point of matching that shape.
        summary = cost_analysis.calculate([call_log])
        assert summary["pricing_known"] is True
        assert summary["total_usd"] > 0
        assert summary["by_pass"][0]["pass"] == "seo_suggestions"

    def test_prompt_carries_the_material_the_pipeline_already_has(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ) as mock_call:
            seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        user_prompt = mock_call.call_args.args[1]
        assert "Data Centers and the Grid" in user_prompt
        assert "Interconnection queues, not generation" in user_prompt
        assert "Municipal officials evaluating" in user_prompt
        assert "Infrastructure policy for Northern Illinois." in user_prompt
        # The outline, not just a blind head+tail slice of the body.
        assert "## What the load actually looks like" in user_prompt
        assert "### A sub-point" in user_prompt
        assert "Opening paragraph that frames the piece." in user_prompt

    def test_uses_the_small_fast_model(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ) as mock_call:
            seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert mock_call.call_args.kwargs["model"] == "mistral-small-latest"


class TestMetaDescriptionLengthConstraint:
    def test_prompt_states_the_limit(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ) as mock_call:
            seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert "under 155 characters" in mock_call.call_args.args[1]

    def test_publication_config_governs_the_limit(self):
        pub_config = {**_PUB_CONFIG, "seo_rules": {"meta_description_max_chars": 120}}
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ) as mock_call:
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=pub_config, api_keys=_KEYS
            )

        assert "under 120 characters" in mock_call.call_args.args[1]
        assert suggestions["meta_description_limit"] == 120

    def test_over_limit_description_is_flagged_not_silently_shipped(self):
        long_description = "x" * 200
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(
                {**_GOOD_DATA, "meta_description": long_description}
            ),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["meta_description_over_limit"] is True
        assert suggestions["meta_description_chars"] == 200
        assert suggestions["meta_description_limit"] == 155
        # Reported in full rather than truncated into a dangling clause — the
        # author trims it.
        assert suggestions["meta_description"] == long_description

    def test_description_at_the_limit_is_not_flagged(self):
        exact = "x" * 155
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response({**_GOOD_DATA, "meta_description": exact}),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["meta_description_over_limit"] is False


class TestOgTitle:
    def _long_title_seo_result(self):
        handoff = {**_HANDOFF, "title": "A" * 71}
        return handoff, seo_analysis.analyze(_ARTICLE, handoff)

    def test_requested_and_returned_when_the_title_is_too_long(self):
        handoff, seo_result = self._long_title_seo_result()
        assert "title_too_long" in [i["type"] for i in seo_result["issues"]]

        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response({**_GOOD_DATA, "og_title": "A Shorter Title"}),
        ) as mock_call:
            suggestions, _ = seo_suggest.generate(
                _ARTICLE,
                handoff=handoff,
                pub_config=_PUB_CONFIG,
                api_keys=_KEYS,
                seo_result=seo_result,
            )

        assert "OG TITLE REQUESTED" in mock_call.call_args.args[1]
        assert suggestions["og_title"] == "A Shorter Title"
        assert suggestions["og_title_chars"] == len("A Shorter Title")
        assert suggestions["og_title_over_limit"] is False

    def test_over_long_og_title_is_flagged(self):
        handoff, seo_result = self._long_title_seo_result()
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response({**_GOOD_DATA, "og_title": "B" * 65}),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE,
                handoff=handoff,
                pub_config=_PUB_CONFIG,
                api_keys=_KEYS,
                seo_result=seo_result,
            )

        assert suggestions["og_title_over_limit"] is True
        assert suggestions["og_title_limit"] == 60

    def test_not_requested_when_the_title_fits(self):
        seo_result = seo_analysis.analyze(_ARTICLE, _HANDOFF)
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response({**_GOOD_DATA, "og_title": "Unasked For"}),
        ) as mock_call:
            suggestions, _ = seo_suggest.generate(
                _ARTICLE,
                handoff=_HANDOFF,
                pub_config=_PUB_CONFIG,
                api_keys=_KEYS,
                seo_result=seo_result,
            )

        assert "OG TITLE REQUESTED" not in mock_call.call_args.args[1]
        assert "og_title" not in suggestions


class TestGracefulDegradation:
    """A suggestion is a nicety — no failure here may cost the author a run."""

    def test_failed_call_returns_failed_status_and_still_logs_cost(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value={
                "failed": True,
                "error": "503 Service Unavailable",
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.4,
            },
        ):
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["status"] == "failed"
        assert "503" in suggestions["reason"]
        assert "keyword_candidates" not in suggestions
        # The attempt still cost something, so it is still logged.
        assert call_log["pass"] == "seo_suggestions"
        assert call_log["failed"] is True

    def test_raising_call_is_caught(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            side_effect=RuntimeError("socket exploded"),
        ):
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["status"] == "failed"
        assert "socket exploded" in suggestions["reason"]
        assert call_log is None

    def test_non_dict_payload_is_a_failure_not_a_crash(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(["not", "an", "object"]),
        ):
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["status"] == "failed"
        assert call_log is not None

    def test_empty_payload_is_a_failure(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(
                {"keyword_candidates": [], "meta_description": ""}
            ),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["status"] == "failed"

    def test_missing_api_key_skips_without_calling(self):
        with patch("ci_article_review.analysis.seo_suggest.mistral.call") as mock_call:
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys={}
            )

        mock_call.assert_not_called()
        assert suggestions["status"] == "skipped"
        assert "mistral API key" in suggestions["reason"]
        assert call_log is None

    def test_no_article_text_does_not_crash(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ):
            suggestions, _ = seo_suggest.generate(
                "", handoff=None, pub_config=None, api_keys=_KEYS
            )

        assert suggestions["status"] == "ok"


class TestDisabling:
    def test_seo_rules_flag_off_skips_without_calling(self):
        pub_config = {**_PUB_CONFIG, "seo_rules": {"suggestions": False}}
        with patch("ci_article_review.analysis.seo_suggest.mistral.call") as mock_call:
            suggestions, call_log = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=pub_config, api_keys=_KEYS
            )

        mock_call.assert_not_called()
        assert suggestions["status"] == "skipped"
        assert "disabled" in suggestions["reason"]
        assert call_log is None

    def test_on_by_default(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(_GOOD_DATA),
        ) as mock_call:
            seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config={}, api_keys=_KEYS
            )

        mock_call.assert_called_once()


class TestCandidateNormalization:
    def test_capped_at_five(self):
        many = [{"keyword": f"phrase {i}", "rationale": "why"} for i in range(9)]
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response({**_GOOD_DATA, "keyword_candidates": many}),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert len(suggestions["keyword_candidates"]) == 5

    def test_bare_strings_are_accepted_as_candidates(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(
                {**_GOOD_DATA, "keyword_candidates": ["grid interconnection", ""]}
            ),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        assert suggestions["keyword_candidates"] == [
            {"keyword": "grid interconnection", "rationale": ""}
        ]

    def test_non_list_candidates_do_not_crash(self):
        with patch(
            "ci_article_review.analysis.seo_suggest.mistral.call",
            return_value=_model_response(
                {**_GOOD_DATA, "keyword_candidates": "one big string"}
            ),
        ):
            suggestions, _ = seo_suggest.generate(
                _ARTICLE, handoff=_HANDOFF, pub_config=_PUB_CONFIG, api_keys=_KEYS
            )

        # No candidates survived, but the description did, so this is still ok.
        assert suggestions["status"] == "ok"
        assert suggestions["keyword_candidates"] == []


class TestOutline:
    def test_long_outline_is_capped(self):
        text = "\n\n".join(f"## Heading {i}" for i in range(60))
        outline = seo_suggest._outline(text)
        assert "## Heading 39" in outline
        assert "## Heading 40" not in outline
        assert "more heading(s)" in outline

    def test_no_headings_yields_empty_outline(self):
        assert seo_suggest._outline("Just prose, no headings.") == ""
