"""Model-stated confidence, and WordPress terms that fail to resolve.

Audit findings 5b(i) and 12 — both cases of the pipeline holding information it
never acted on.

Finding 5b(i): every fact-check item and every additional observation carries a
``confidence`` the models were explicitly asked for, and no code read it. Two
models flagging a passage at ``low`` produced a Section 1 consensus flag
identical to two at ``high`` — the hedging was discarded on the path into the
section that implies the most certainty.

Finding 12: WordPress category/tag slugs that could not be resolved were logged
and dropped, and ``push`` still returned success.
"""

from unittest.mock import MagicMock, patch

from ci_article_review import consolidation
from ci_article_review.adapters.cms import wordpress


def _fact_check_result(confidence):
    """Two models flagging the same passage at the same stated confidence."""

    def one(model):
        return {
            "failed": False,
            "data": {
                "outdated": [
                    {
                        "claim": "The figure is 42 percent.",
                        "current_value": "51 percent",
                        "source": "s",
                        "confidence": confidence,
                    }
                ]
            },
            "model": model,
            "tokens": {},
        }

    return {
        ("gemini", "fact_check"): one("gemini"),
        ("openai", "fact_check"): one("openai"),
    }


class TestConfidenceIsInertByDefault:
    """The change must not silently move anyone's thresholds."""

    def test_high_and_low_are_identical_with_no_configuration(self):
        high, _ = consolidation._find_consensus(_fact_check_result("high"), [], {})
        low, _ = consolidation._find_consensus(_fact_check_result("low"), [], {})
        assert high[0]["weight_sum"] == low[0]["weight_sum"]


class TestConfidenceWeightingWhenEnabled:
    _CFG = {"confidence_weights": {"high": 1.0, "medium": 0.75, "low": 0.5}}

    def test_low_confidence_reduces_the_consensus_weight(self):
        # Threshold lowered so both cases still produce a consensus entry —
        # this test is about the weight, and the threshold interaction is
        # covered separately below.
        cfg = {**self._CFG, "consensus_threshold": 1.0}
        high, _ = consolidation._find_consensus(_fact_check_result("high"), [], cfg)
        low, _ = consolidation._find_consensus(_fact_check_result("low"), [], cfg)
        assert low[0]["weight_sum"] < high[0]["weight_sum"]
        # gemini 1.5 + openai 1.0, halved by the "low" multiplier.
        assert high[0]["weight_sum"] == 2.5
        assert low[0]["weight_sum"] == 1.25

    def test_two_hedged_models_can_drop_below_the_threshold(self):
        """The behavioural point: Section 1 stops implying certainty nobody claimed."""
        cfg = {**self._CFG, "consensus_threshold": 2.0}
        consensus, single = consolidation._find_consensus(
            _fact_check_result("low"), [], cfg
        )
        assert consensus == [], "two 'low' flags still reached consensus"
        assert single, "the finding vanished instead of demoting to single-source"

    def test_agreement_at_low_confidence_can_still_reach_consensus(self):
        """Damp, never veto — several hedged models agreeing is still evidence."""
        cfg = {**self._CFG, "consensus_threshold": 1.0}
        consensus, _ = consolidation._find_consensus(_fact_check_result("low"), [], cfg)
        assert len(consensus) == 1

    def test_findings_with_no_confidence_field_are_unaffected(self):
        """Most domains never emit one; their weighting must not change."""
        results = {
            ("openai", "voice_style"): {
                "failed": False,
                "data": {"flags": [{"passage": "p", "problem": "x"}]},
                "model": "openai",
                "tokens": {},
            },
            ("claude", "voice_style"): {
                "failed": False,
                "data": {"flags": [{"passage": "p", "problem": "y"}]},
                "model": "claude",
                "tokens": {},
            },
        }
        plain, _ = consolidation._find_consensus(results, [], {})
        weighted, _ = consolidation._find_consensus(results, [], self._CFG)
        assert plain[0]["weight_sum"] == weighted[0]["weight_sum"]

    def test_an_unrecognised_confidence_value_scores_neutral(self):
        consensus, _ = consolidation._find_consensus(
            _fact_check_result("extremely sure"), [], self._CFG
        )
        plain, _ = consolidation._find_consensus(_fact_check_result("high"), [], {})
        assert consensus[0]["weight_sum"] == plain[0]["weight_sum"]

    def test_a_non_numeric_multiplier_is_ignored_with_a_warning(self, caplog):
        cfg = {"confidence_weights": {"low": "very small"}}
        consolidation._find_consensus(_fact_check_result("low"), [], cfg)
        assert "not a number" in caplog.text


class TestWordPressUnresolvedTerms:
    """Finding 12 — a post published into no category reported clean success."""

    def _wp_config(self):
        return {
            "site_url": "https://example.com",
            "username": "u",
            "application_password": "p",
        }

    def _no_such_term(self):
        mock = MagicMock()
        mock.ok = True
        mock.json.return_value = []
        mock.raise_for_status = MagicMock()
        return mock

    def _created(self):
        mock = MagicMock()
        mock.json.return_value = {"id": 1, "link": "https://example.com/p"}
        mock.raise_for_status = MagicMock()
        return mock

    def test_draft_push_succeeds_but_reports_the_missing_terms(self):
        with (
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.get",
                return_value=self._no_such_term(),
            ),
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.post",
                return_value=self._created(),
            ),
        ):
            result = wordpress.push(
                "body",
                {"title": "T", "wordpress_category": "nope", "tags": ["also-nope"]},
                self._wp_config(),
                {},
            )
        assert result["success"] is True
        assert result["unresolved_terms"] == ["also-nope", "nope"]

    def test_live_publish_refuses_rather_than_losing_the_metadata(self):
        """Fail closed: --publish-live is not reversible in practice."""
        with (
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.get",
                return_value=self._no_such_term(),
            ),
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.post"
            ) as mock_post,
        ):
            result = wordpress.push(
                "body",
                {"title": "T", "wordpress_category": "nope"},
                self._wp_config(),
                {},
                publish_live=True,
            )
        assert result["success"] is False
        assert "nope" in result["error"]
        mock_post.assert_not_called(), "the post was created despite the refusal"

    def test_live_publish_can_be_forced(self):
        with (
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.get",
                return_value=self._no_such_term(),
            ),
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.post",
                return_value=self._created(),
            ),
        ):
            result = wordpress.push(
                "body",
                {"title": "T", "wordpress_category": "nope"},
                self._wp_config(),
                {},
                publish_live=True,
                allow_missing_terms=True,
            )
        assert result["success"] is True
        assert result["unresolved_terms"] == ["nope"]

    def test_clean_publish_carries_no_warning(self):
        found = MagicMock()
        found.ok = True
        found.json.return_value = [{"id": 7}]
        found.raise_for_status = MagicMock()
        with (
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.get",
                return_value=found,
            ),
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.post",
                return_value=self._created(),
            ),
        ):
            result = wordpress.push(
                "body",
                {"title": "T", "wordpress_category": "real"},
                self._wp_config(),
                {},
                publish_live=True,
            )
        assert result["success"] is True
        assert "unresolved_terms" not in result


class TestWordPressRequiresHttps:
    """Finding 19 — an application password is not protected by base64."""

    def test_http_is_refused(self):
        import pytest

        with pytest.raises(ValueError, match="insecure"):
            wordpress.push(
                "body",
                {"title": "T"},
                {
                    "site_url": "http://example.com",
                    "username": "u",
                    "application_password": "p",
                },
                {},
            )

    def test_http_can_be_allowed_for_a_local_test_site(self):
        with (
            patch(
                "ci_article_review.adapters.cms.wordpress.requests.post"
            ) as mock_post,
        ):
            mock_post.return_value = MagicMock(
                json=lambda: {"id": 1, "link": "http://localhost/p"},
                raise_for_status=MagicMock(),
            )
            result = wordpress.push(
                "body",
                {"title": "T"},
                {
                    "site_url": "http://localhost:8080",
                    "username": "u",
                    "application_password": "p",
                    "allow_insecure_http": True,
                },
                {},
            )
        assert result["success"] is True
