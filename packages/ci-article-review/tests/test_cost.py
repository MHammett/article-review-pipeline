"""Tests for analysis.cost."""
import sys, os

from ci_article_review.analysis.cost import calculate, _price_for_model


class TestPriceForModel:
    def test_known_model_exact(self):
        in_p, out_p = _price_for_model("gpt-5.4")
        assert in_p == 2.50
        assert out_p == 15.00

    def test_prefix_match(self):
        # A versioned variant like gemini-2.5-flash-0520 should match gemini-2.5-flash
        in_p, out_p = _price_for_model("gemini-2.5-flash-0520")
        assert in_p == 0.30

    def test_unknown_model_returns_fallback(self):
        from ci_article_review.analysis.cost import _UNKNOWN_PRICE
        result = _price_for_model("some-hypothetical-model-9999")
        assert result == _UNKNOWN_PRICE

    def test_none_returns_fallback(self):
        from ci_article_review.analysis.cost import _UNKNOWN_PRICE
        assert _price_for_model(None) == _UNKNOWN_PRICE


class TestCalculate:
    def _log(self, model, prompt_tok, completion_tok, failed=False):
        return {
            "pass": f"openai:fact_check",
            "model": model,
            "failed": failed,
            "tokens": {"prompt": prompt_tok, "completion": completion_tok},
            "elapsed_seconds": 1.0,
        }

    def test_empty_log_zero_cost(self):
        result = calculate([])
        assert result["total_usd"] == 0.0
        assert result["by_pass"] == []

    def test_none_log_zero_cost(self):
        result = calculate(None)
        assert result["total_usd"] == 0.0

    def test_single_entry_cost(self):
        # gpt-5.4: $2.50/MTok in, $15.00/MTok out
        # 1000 prompt + 500 completion
        entry = self._log("gpt-5.4", 1000, 500)
        result = calculate([entry])
        expected_in = 1000 / 1_000_000 * 2.50
        expected_out = 500 / 1_000_000 * 15.00
        assert abs(result["total_usd"] - round(expected_in + expected_out, 4)) < 0.0001

    def test_failed_entry_contributes_zero(self):
        entry = self._log("gpt-5.4", 100_000, 50_000, failed=True)
        result = calculate([entry])
        assert result["total_usd"] == 0.0

    def test_pricing_known_true_for_known_models(self):
        entry = self._log("gpt-5.4", 1000, 500)
        result = calculate([entry])
        assert result["pricing_known"] is True

    def test_pricing_known_false_for_unknown_model(self):
        entry = self._log("some-unknown-model-x", 1000, 500)
        result = calculate([entry])
        assert result["pricing_known"] is False

    def test_by_pass_has_all_keys(self):
        entry = self._log("claude-opus-4-8", 2000, 800)
        result = calculate([entry])
        row = result["by_pass"][0]
        for k in ("pass", "model", "input_usd", "output_usd", "total_usd"):
            assert k in row, f"missing key: {k}"

    def test_model_annotation_stripped(self):
        # Model strings from api_call_log may include "[FALLBACK from ...]"
        entry = self._log("gpt-5.4 [FALLBACK from gpt-5.5]", 1000, 500)
        result = calculate([entry])
        assert result["pricing_known"] is True

    def test_tolerates_calibration_fields(self):
        # api_call_log entries now carry calibration fields (effort, timeout budget,
        # headroom, char_count, status). Cost calculation must ignore the extras.
        entry = self._log("gpt-5.5", 18843, 17218)
        entry.update({
            "effort": "xhigh", "timeout_budget_seconds": 819,
            "headroom_seconds": 478.9, "char_count": 73786, "status": "ok",
        })
        result = calculate([entry])
        assert result["total_usd"] > 0
        assert result["pricing_known"] is True
