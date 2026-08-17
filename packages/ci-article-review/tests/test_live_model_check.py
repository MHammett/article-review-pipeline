"""The run report's answer to "did something newer than this ship?"

Two properties matter more than the rest, and both are about silence.

First, "we could not check" must never render as "you are on the newest
model". Every degraded path here — no API key, an HTTP error, a provider that
lists no date for the model you ran — produces an `unchecked` entry rather than
an empty `newer` list, because an empty `newer` list is what being current
looks like.

Second, nothing in this module may fail a run. It is advisory information
attached to a pipeline that has already spent real money on thirty model calls;
an unreadable cache file or a provider timeout has to degrade, not raise.

No network: the provider sweep is stubbed everywhere.
"""

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ci_article_review import live_model_check


def _cache(tmp_path):
    return tmp_path / "model_discovery.json"


def _entry(models, status="ok", reason="", detail=""):
    return {
        "status": status,
        "reason": reason,
        "detail": detail,
        "static": False,
        "models": models,
    }


def _write(tmp_path, providers, checked=None):
    """Write a cache directly, at a chosen age."""
    stamp = checked or datetime.datetime.now(datetime.timezone.utc)
    data = {
        "version": 1,
        "providers": {
            key: {
                "checked": stamp.isoformat(),
                "status": rec["status"],
                "reason": rec["reason"],
                "detail": rec["detail"],
                "static": rec["static"],
                "models": [
                    [mid, d.isoformat() if d else None] for mid, d in rec["models"]
                ],
            }
            for key, rec in providers.items()
        },
    }
    path = _cache(tmp_path)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


_OLD = datetime.date(2026, 1, 1)
_NEW = datetime.date(2026, 8, 1)


class TestCacheRoundTrip:
    def test_a_saved_sweep_loads_back_with_its_dates(self, tmp_path):
        collected = {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW)])}
        live_model_check.save_cache(collected, path=_cache(tmp_path))

        loaded = live_model_check.load_cache(_cache(tmp_path))
        assert loaded["openai"]["models"] == [("gpt-5.5", _OLD), ("gpt-5.6", _NEW)]
        assert loaded["openai"]["checked"].tzinfo is not None

    def test_a_model_with_no_date_survives_as_no_date(self, tmp_path):
        """Not as a very old one — that would fabricate a newer-than result."""
        live_model_check.save_cache(
            {"gemini": _entry([("gemini-9", None)])}, path=_cache(tmp_path)
        )
        loaded = live_model_check.load_cache(_cache(tmp_path))
        assert loaded["gemini"]["models"] == [("gemini-9", None)]

    def test_a_partial_sweep_does_not_clobber_the_other_providers(self, tmp_path):
        """`ci-discover --provider openai` must not erase what we knew of grok.

        A narrow manual check that reduced what the next run could say would be
        a trap: the user did more work and got less information.
        """
        path = _cache(tmp_path)
        live_model_check.save_cache(
            {
                "openai": _entry([("gpt-5.5", _OLD)]),
                "grok": _entry([("grok-4.3", _OLD)]),
            },
            path=path,
        )
        live_model_check.save_cache(
            {"openai": _entry([("gpt-5.6", _NEW)])}, providers=["openai"], path=path
        )

        loaded = live_model_check.load_cache(path)
        assert [m for m, _ in loaded["openai"]["models"]] == ["gpt-5.6"]
        assert [m for m, _ in loaded["grok"]["models"]] == ["grok-4.3"]

    @pytest.mark.parametrize(
        "content",
        ["", "not json at all", "[]", '{"version": 999, "providers": {}}'],
    )
    def test_an_unusable_cache_is_empty_not_an_exception(self, tmp_path, content):
        path = _cache(tmp_path)
        path.write_text(content, encoding="utf-8")
        assert live_model_check.load_cache(path) == {}

    def test_a_missing_cache_is_empty_not_an_exception(self, tmp_path):
        assert live_model_check.load_cache(tmp_path / "nope.json") == {}

    def test_an_unwritable_location_reports_failure_rather_than_raising(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("I am a file, not a directory", encoding="utf-8")
        got = live_model_check.save_cache(
            {"openai": _entry([])}, path=blocker / "sub" / "cache.json"
        )
        assert got is None


class TestModelsThatRan:
    def test_review_passes_map_provider_to_model(self):
        log = [
            {"pass": "openai:fact_check", "model": "gpt-5.5", "failed": False},
            {"pass": "grok:red_team", "model": "grok-4.3", "failed": False},
        ]
        assert live_model_check.models_that_ran(log) == {
            "openai": "gpt-5.5",
            "grok": "grok-4.3",
        }

    def test_a_fallback_reports_the_model_that_actually_ran(self):
        """The point of reading the call log instead of the config."""
        log = [
            {
                "pass": "openai:fact_check",
                "model": "gpt-5.4 [FALLBACK from gpt-5.5]",
                "failed": False,
            }
        ]
        assert live_model_check.models_that_ran(log) == {"openai": "gpt-5.4"}

    def test_the_grounded_annotation_is_stripped(self):
        log = [{"pass": "gemini:fact_check", "model": "gemini-2.5-pro [grounded]"}]
        assert live_model_check.models_that_ran(log) == {"gemini": "gemini-2.5-pro"}

    def test_a_failed_call_is_not_a_model_that_ran(self):
        log = [{"pass": "openai:fact_check", "model": "gpt-5.5", "failed": True}]
        assert live_model_check.models_that_ran(log) == {}

    def test_auxiliary_passes_are_skipped(self):
        """seo_suggestions and citation verification are not the review ensemble."""
        log = [{"pass": "seo_suggestions", "model": "mistral-small-latest"}]
        assert live_model_check.models_that_ran(log) == {}

    @pytest.mark.parametrize("log", [None, [], [{}]])
    def test_degenerate_logs_do_not_raise(self, log):
        assert live_model_check.models_that_ran(log) == {}


class TestCheck:
    def test_a_newer_model_is_reported_against_the_model_that_ran(self, tmp_path):
        """The gap this whole feature exists to close."""
        _write(tmp_path, {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW)])})

        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )

        assert [f["provider"] for f in got["newer"]] == ["openai"]
        assert got["newer"][0]["newer"][0]["model"] == "gpt-5.6"
        assert got["newer"][0]["newer"][0]["released"] == "2026-08-01"
        assert got["current"] == []

    def test_nothing_newer_is_reported_as_checked_and_current(self, tmp_path):
        _write(tmp_path, {"openai": _entry([("gpt-5.5", _NEW), ("gpt-4", _OLD)])})

        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )

        assert got["newer"] == []
        assert [c["provider"] for c in got["current"]] == ["openai"]
        assert got["unchecked"] == []

    def test_an_unreachable_provider_is_unchecked_not_current(self, tmp_path):
        """The distinction the report must never collapse."""
        _write(
            tmp_path,
            {"openai": _entry([], status="error", reason="HTTP 401", detail="bad key")},
        )

        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )

        assert got["newer"] == []
        assert got["current"] == []
        assert [u["provider"] for u in got["unchecked"]] == ["openai"]
        assert "401" in got["unchecked"][0]["reason"]

    def test_a_provider_never_swept_is_unchecked(self, tmp_path):
        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )
        assert got["unchecked"][0]["reason"] == "never checked"
        assert got["current"] == []

    def test_a_model_the_provider_does_not_date_is_unchecked_not_current(
        self, tmp_path
    ):
        """No date on the model you ran means no comparison is possible.

        Reporting that as "nothing newer" would be the quiet failure: the tool
        keeps working and simply stops telling you about upgrades.
        """
        _write(tmp_path, {"gemini": _entry([("gemini-2.5-pro", None)])})

        got = live_model_check.check(
            {"gemini": "gemini-2.5-pro"}, {}, cache_path=_cache(tmp_path)
        )

        assert got["current"] == []
        assert got["newer"] == []
        assert "no release date" in got["unchecked"][0]["reason"]

    def test_undated_models_are_counted_rather_than_silently_dropped(self, tmp_path):
        _write(
            tmp_path,
            {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW), ("gpt-x", None)])},
        )
        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )
        assert got["newer"][0]["undated_models"] == 1

    def test_a_model_missing_from_pricing_is_flagged_as_unpriced(self, tmp_path):
        """A brand-new model is exactly what pricing.yaml has not caught up with."""
        _write(
            tmp_path,
            {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.99-imaginary", _NEW)])},
        )
        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )
        assert got["newer"][0]["newer"][0]["price_known"] is False

    def test_a_priced_model_is_flagged_as_priced(self, tmp_path):
        _write(tmp_path, {"openai": _entry([("gpt-5.4", _OLD), ("gpt-5.5", _NEW)])})
        got = live_model_check.check(
            {"openai": "gpt-5.4"}, {}, cache_path=_cache(tmp_path)
        )
        assert got["newer"][0]["newer"][0]["price_known"] is True

    def test_newer_models_are_listed_newest_first(self, tmp_path):
        mid = datetime.date(2026, 5, 1)
        _write(
            tmp_path,
            {"openai": _entry([("gpt-5.5", _OLD), ("a", mid), ("b", _NEW)])},
        )
        got = live_model_check.check(
            {"openai": "gpt-5.5"}, {}, cache_path=_cache(tmp_path)
        )
        assert [m["model"] for m in got["newer"][0]["newer"]] == ["b", "a"]


class TestRefreshPolicy:
    def test_a_fresh_cache_makes_no_network_call(self, tmp_path):
        _write(tmp_path, {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW)])})

        with patch.object(
            live_model_check.discover, "collect_available_models"
        ) as sweep:
            got = live_model_check.check(
                {"openai": "gpt-5.5"}, {}, refresh=True, cache_path=_cache(tmp_path)
            )

        sweep.assert_not_called()
        assert got["refreshed"] is False
        assert got["newer"], "a fresh cache should still produce the finding"

    def test_a_stale_cache_is_refreshed_when_asked(self, tmp_path):
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=9)
        _write(tmp_path, {"openai": _entry([("gpt-5.5", _OLD)])}, checked=old)

        swept = {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW)])}
        with patch.object(
            live_model_check.discover, "collect_available_models", return_value=swept
        ) as sweep:
            got = live_model_check.check(
                {"openai": "gpt-5.5"}, {}, refresh=True, cache_path=_cache(tmp_path)
            )

        sweep.assert_called_once()
        assert got["refreshed"] is True
        assert got["newer"][0]["newer"][0]["model"] == "gpt-5.6"
        # …and the refreshed answer is left behind for the next run.
        assert "gpt-5.6" in _cache(tmp_path).read_text(encoding="utf-8")

    def test_a_stale_cache_is_still_used_when_not_refreshing(self, tmp_path):
        """Yesterday's answer is still an answer; its age is disclosed."""
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=9)
        _write(
            tmp_path,
            {"openai": _entry([("gpt-5.5", _OLD), ("gpt-5.6", _NEW)])},
            checked=old,
        )

        with patch.object(
            live_model_check.discover, "collect_available_models"
        ) as sweep:
            got = live_model_check.check(
                {"openai": "gpt-5.5"}, {}, refresh=False, cache_path=_cache(tmp_path)
            )

        sweep.assert_not_called()
        assert got["newer"][0]["newer"][0]["model"] == "gpt-5.6"
        assert got["oldest_check_age_hours"] > 24

    def test_a_sweep_that_raises_degrades_instead_of_propagating(self, tmp_path):
        """A provider outage must not take a paid pipeline run with it."""
        with patch.object(
            live_model_check.discover,
            "collect_available_models",
            side_effect=RuntimeError("provider on fire"),
        ):
            got = live_model_check.check(
                {"openai": "gpt-5.5"}, {}, refresh=True, cache_path=_cache(tmp_path)
            )

        assert got["refreshed"] is False
        assert [u["provider"] for u in got["unchecked"]] == ["openai"]

    def test_only_stale_providers_are_swept(self, tmp_path):
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=9)
        path = _cache(tmp_path)
        _write(tmp_path, {"openai": _entry([("gpt-5.5", _NEW)])})
        # grok's entry is old; openai's was just written.
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["providers"]["grok"] = {
            "checked": old.isoformat(),
            "status": "ok",
            "reason": "",
            "detail": "",
            "static": False,
            "models": [["grok-4.3", "2026-01-01"]],
        }
        path.write_text(json.dumps(cached), encoding="utf-8")

        with patch.object(
            live_model_check.discover, "collect_available_models", return_value={}
        ) as sweep:
            live_model_check.check(
                {"openai": "gpt-5.5", "grok": "grok-4.3"},
                {},
                refresh=True,
                cache_path=path,
            )

        assert sweep.call_args.kwargs["providers"] == ["grok"]


class TestNoRanModels:
    def test_an_empty_run_reports_unavailable_without_touching_the_network(
        self, tmp_path
    ):
        with patch.object(
            live_model_check.discover, "collect_available_models"
        ) as sweep:
            got = live_model_check.check(
                {}, {}, refresh=True, cache_path=_cache(tmp_path)
            )

        sweep.assert_not_called()
        assert got["status"] == "unavailable"
        assert got["newer"] == [] and got["current"] == [] and got["unchecked"] == []


class TestTerminalSummary:
    """What the run prints. Lives in pipeline.py; tested with the feature.

    This block prints on every single run, so its restraint when there is
    nothing to say matters as much as its clarity when there is.
    """

    @staticmethod
    def _print(live, capsys):
        from ci_article_review.pipeline import _print_live_model_check

        _print_live_model_check(live)
        return capsys.readouterr().out

    def test_a_newer_model_is_named_with_the_model_that_ran(self, capsys):
        out = self._print(
            {
                "newer": [
                    {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "newer": [
                            {
                                "model": "gpt-5.6",
                                "released": "2026-08-10",
                                "price_known": True,
                            }
                        ],
                    }
                ],
                "current": [],
                "unchecked": [],
            },
            capsys,
        )
        assert "gpt-5.5" in out and "gpt-5.6" in out
        assert "not necessarily better or cheaper" in out

    def test_an_unpriced_model_is_marked(self, capsys):
        out = self._print(
            {
                "newer": [
                    {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "newer": [
                            {
                                "model": "gpt-5.6",
                                "released": "2026-08-10",
                                "price_known": False,
                            }
                        ],
                    }
                ],
                "current": [],
                "unchecked": [],
            },
            capsys,
        )
        assert "pricing.yaml" in out

    def test_no_data_at_all_is_one_line(self, capsys):
        """The default path — off, and no ci-discover has ever run."""
        out = self._print(
            {
                "newer": [],
                "current": [],
                "unchecked": [
                    {"provider": p, "model": "m", "reason": "never checked"}
                    for p in ("openai", "gemini", "mistral")
                ],
            },
            capsys,
        )
        assert "ci-discover" in out
        assert "never checked" not in out
        assert len([line for line in out.splitlines() if line.strip()]) == 1

    def test_a_partial_check_names_what_was_missed(self, capsys):
        """Here the roll-call earns its space: some providers *were* asked."""
        out = self._print(
            {
                "newer": [],
                "current": [{"provider": "grok", "model": "grok-4.3"}],
                "unchecked": [
                    {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "reason": "the models API returned HTTP 401",
                    }
                ],
                "oldest_check_age_hours": 3.0,
            },
            capsys,
        )
        assert "openai" in out and "401" in out
        assert "grok" not in out.split("Not checked")[1]

    @pytest.mark.parametrize(
        "live", [None, {}, {"newer": [], "current": [], "unchecked": []}]
    )
    def test_nothing_to_say_prints_nothing(self, live, capsys):
        assert self._print(live, capsys) == ""


class TestCachePathDefault:
    def test_the_cache_location_is_gitignored(self):
        """A TTL'd copy of six providers' model listings is derived data.

        (``CACHE_PATH`` itself is redirected per-test by the autouse fixture in
        conftest.py, so this asserts the committed default instead.)
        """
        root = Path(__file__).resolve().parents[3]
        assert ".cache/" in (root / ".gitignore").read_text(encoding="utf-8")
