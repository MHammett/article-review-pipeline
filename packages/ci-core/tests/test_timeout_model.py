"""Tests for the sliding-scale timeout model."""

import ci_core.llm.timeout_model as tm

CEILING = 1200  # task_timeout_seconds headroom; formula clamps to CEILING - 15
ANCHOR = 62000  # calibration doc size (the 1.0 size bucket)


class TestComputeTimeout:
    def test_gpt55_xhigh_matches_calibration_target(self):
        # gpt-5.5 xhigh: base 60 × model 1.3 × effort 10.5 × variance 1.25 ≈ 1024s.
        t = tm.compute_timeout(ANCHOR, "gpt-5.5", "xhigh", CEILING)
        assert 980 <= t <= 1070, t

    def test_variance_margin_applied(self):
        # The configured variance_margin must scale the result above the bare
        # base × model × effort product.
        import copy

        cfg = copy.deepcopy(tm._CONFIG)
        cfg["variance_margin"] = 1.0
        bare = tm.compute_timeout(ANCHOR, "gpt-5.5", "xhigh", CEILING, config=cfg)
        withmargin = tm.compute_timeout(ANCHOR, "gpt-5.5", "xhigh", CEILING)
        assert withmargin > bare
        assert round(withmargin / bare, 2) == tm._CONFIG["variance_margin"]

    def test_effort_is_steep(self):
        # Same model: high should be several times none (calibration showed ~5x).
        # Threshold is 2.5x rather than the tighter historical 3x because
        # floor_seconds was raised 60->90 (for Grok's thin timeout headroom),
        # which also lifts the floor-clamped "none" baseline for gpt-5.4 — the
        # ratio compresses even though the underlying "high" value (unfloored,
        # effort-driven) hasn't changed.
        none = tm.compute_timeout(ANCHOR, "gpt-5.4", "none", CEILING)
        high = tm.compute_timeout(ANCHOR, "gpt-5.4", "high", CEILING)
        assert high >= 2.5 * none, (none, high)

    def test_floor_enforced(self):
        # Grok is fast; its computed value falls below the floor and is clamped up.
        t = tm.compute_timeout(ANCHOR, "grok-4.20-0309-reasoning", None, CEILING)
        assert t == tm._CONFIG["floor_seconds"]

    def test_ceiling_enforced_on_huge_doc(self):
        t = tm.compute_timeout(500000, "gpt-5.5", "xhigh", CEILING)
        assert t == CEILING - 15

    def test_low_ceiling_clamps_xhigh(self):
        # With a tight ceiling, even gpt-5.5 xhigh is clamped down to ceiling - 15.
        t = tm.compute_timeout(ANCHOR, "gpt-5.5", "xhigh", 500)
        assert t == 500 - 15

    def test_size_scales_monotonically(self):
        sizes = [3000, 15000, 40000, 62000, 120000]
        vals = [tm.compute_timeout(c, "gpt-5.5", "xhigh", CEILING) for c in sizes]
        assert vals == sorted(vals), vals

    def test_grounded_model_high_even_without_effort(self):
        # Gemini has no reasoning_effort knob, but grounded latency is captured by
        # its model multiplier — it must exceed a fast model's timeout.
        gem = tm.compute_timeout(ANCHOR, "gemini-2.5-pro", None, CEILING)
        grok = tm.compute_timeout(ANCHOR, "grok-4.20-0309-reasoning", None, CEILING)
        assert gem > grok

    def test_model_prefix_longest_match(self):
        # "mistral-medium-3-5" must match its own entry, not a shorter prefix.
        med = tm.compute_timeout(ANCHOR, "mistral-medium-3-5", "none", CEILING)
        small = tm.compute_timeout(ANCHOR, "mistral-small-latest", "none", CEILING)
        # medium-3-5 multiplier (1.3) > small (0.8), so medium >= small
        assert med >= small

    def test_unknown_model_uses_default(self):
        t = tm.compute_timeout(ANCHOR, "some-future-model-x", "none", CEILING)
        assert t >= tm._CONFIG["floor_seconds"]

    def test_unknown_effort_falls_back_to_default(self):
        # An unrecognized effort string uses the 'default' multiplier, not a crash.
        t = tm.compute_timeout(ANCHOR, "gpt-5.4", "ultra-mega", CEILING)
        assert t >= tm._CONFIG["floor_seconds"]


class TestComputeAll:
    def test_respects_explicit_override(self):
        cfgs = {
            "openai": {
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "timeout_seconds": 999,
            }
        }
        out = tm.compute_all(ANCHOR, cfgs, CEILING)
        assert out["openai"] == 999  # explicit wins, formula skipped

    def test_computes_when_unset(self):
        cfgs = {"openai": {"model": "gpt-5.5", "reasoning_effort": "xhigh"}}
        out = tm.compute_all(ANCHOR, cfgs, CEILING)
        assert 980 <= out["openai"] <= 1070

    def test_skips_disabled_models(self):
        cfgs = {
            "openai": {"model": "gpt-5.4", "enabled": False},
            "grok": {"model": "grok-4.20-0309-reasoning"},
        }
        out = tm.compute_all(ANCHOR, cfgs, CEILING)
        assert "openai" not in out
        assert "grok" in out

    def test_reads_effort_from_either_key(self):
        # reasoning_effort (openai/mistral/grok) and effort (claude) both recognized.
        a = tm.compute_all(
            ANCHOR, {"x": {"model": "gpt-5.4", "reasoning_effort": "high"}}, CEILING
        )
        b = tm.compute_all(
            ANCHOR, {"x": {"model": "gpt-5.4", "effort": "high"}}, CEILING
        )
        assert a["x"] == b["x"]


class TestGlobalCeiling:
    """The parallel batch's outer wall-clock bound (pipeline._global_ceiling)."""

    def test_exceeds_slowest_task_with_retry_room(self):
        from ci_article_review.pipeline import _global_ceiling

        # Must sit above the slowest task + a retry_delay + slack, so a transient
        # retry or clean timeout isn't masked as a global-ceiling cancellation.
        c = _global_ceiling([819, 273, 240, 60], retry_delay=10)
        assert c == 819 + 10 + 30  # slowest + retry_delay + 30s slack

    def test_margin_above_slowest_is_more_than_old_plus_10(self):
        from ci_article_review.pipeline import _global_ceiling

        # Regression: the old +10 margin let a retry collide with the ceiling.
        c = _global_ceiling([647], retry_delay=10)
        assert c - 647 > 10

    def test_empty_list_safe(self):
        from ci_article_review.pipeline import _global_ceiling

        assert _global_ceiling([], retry_delay=10) == 40
