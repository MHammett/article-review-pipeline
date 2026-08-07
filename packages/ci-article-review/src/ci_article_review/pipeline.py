#!/usr/bin/env python3
"""
Article Review Pipeline — orchestration engine.

Usage:
  python pipeline.py --draft path/to/handoff.md --publication your_publication_name
  python pipeline.py --publish path/to/publication_handoff.md --publication your_publication_name [--publish-live]
"""

import sys

# Reconfigure stdout/stderr to UTF-8 on Windows (default cp1252 breaks on non-ASCII report content)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# Version guard — must run before any other imports
if sys.version_info < (3, 10):
    print(f"ERROR: Python 3.10 or higher is required. You are running {sys.version}.")
    print("Download the latest Python at https://www.python.org/downloads/")
    sys.exit(1)

# Dependency check — catch missing pip packages before any work starts
_REQUIRED_PACKAGES = {
    "requests": "requests",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}
_missing = []
for _import_name, _pkg_name in _REQUIRED_PACKAGES.items():
    try:
        __import__(_import_name)
    except ImportError:
        _missing.append(_pkg_name)
if _missing:
    print("ERROR: Required packages are not installed. Run:")
    print(f"  pip install {' '.join(_missing)}")
    print("Or install all dependencies at once:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

import argparse
import concurrent.futures
import importlib
import logging
import time

import requests
from datetime import datetime, timezone
from pathlib import Path

from .config_loader import (
    load_user_config,
    load_publication_config,
    merge_configs,
    validate_publication_name,
)
from .handoff_parser import parse_draft_submission, parse_publication_handoff
from . import history as hist
from . import consolidation
from . import redact
from .model_registry import check_model_currency
from . import timeout_model
from .analysis import readability as readability_analysis
from .analysis import seo as seo_analysis
from .analysis import cost as cost_analysis
from .analysis.webpage import build_handoff_from_url

log = logging.getLogger("pipeline")

HISTORY_ROOT = "pipeline_history"

# Module-level prompt cache — files are read once per process lifetime.
_PROMPT_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Ensemble assignment system
# ---------------------------------------------------------------------------

#: Maps domain names to their prompt file.
_DOMAIN_PROMPTS: dict[str, str] = {
    "fact_check": "fact_check.txt",
    "voice_style": "ai_speak.txt",
    "completeness": "completeness.txt",
    "argument_integrity": "argument_integrity.txt",
    "red_team": "red_team.txt",
}

#: Maps model names to their adapter module path.
_ADAPTER_MODULES: dict[str, str] = {
    "gemini": "ci_article_review.adapters.review.gemini",
    "openai": "ci_article_review.adapters.review.openai",
    "mistral": "ci_article_review.adapters.review.mistral",
    "grok": "ci_article_review.adapters.review.grok",
    "claude": "ci_article_review.adapters.review.claude",
    "perplexity": "ci_article_review.adapters.review.perplexity",
}

#: Which models run which domains at each thoroughness level.
#: Models are listed in preference order; unconfigured ones are skipped.
_THOROUGHNESS_PRESETS: dict[str, dict[str, list[str]]] = {
    "standard": {
        # One primary model per domain — current baseline behavior.
        "fact_check": ["gemini"],
        "voice_style": ["openai"],
        "completeness": ["openai"],
        "argument_integrity": ["mistral", "claude"],
        "red_team": ["mistral", "grok"],
    },
    "thorough": {
        # Two to three well-suited models per domain.
        # fact_check limited to search-grounded models for quality.
        "fact_check": ["gemini", "perplexity"],
        "voice_style": ["openai", "claude"],
        "completeness": ["openai", "mistral"],
        "argument_integrity": ["mistral", "claude", "openai"],
        "red_team": ["mistral", "grok", "claude"],
    },
    "maximum": {
        # Every configured model runs every domain.
        # Domain-specific weights in consolidation sort the signal from the noise.
        "fact_check": ["gemini", "perplexity", "openai", "mistral", "grok", "claude"],
        "voice_style": ["gemini", "perplexity", "openai", "mistral", "grok", "claude"],
        "completeness": ["gemini", "perplexity", "openai", "mistral", "grok", "claude"],
        "argument_integrity": [
            "gemini",
            "perplexity",
            "openai",
            "mistral",
            "grok",
            "claude",
        ],
        "red_team": ["gemini", "perplexity", "openai", "mistral", "grok", "claude"],
    },
}


def _model_has_credentials(model_name: str, api_keys: dict, model_cfg: dict) -> bool:
    """Return True if the model has credentials to run."""
    if model_name == "gemini":
        if model_cfg.get("provider") == "vertex_ai":
            # Vertex AI uses google-auth; project ID is the required field.
            return bool(model_cfg.get("project"))
        return bool(api_keys.get("gemini", {}).get("api_key"))
    return bool(api_keys.get(model_name, {}).get("api_key"))


def _build_assignments(
    thoroughness: str,
    model_configs: dict,
    api_keys: dict,
) -> list[tuple[str, str]]:
    """Return list of (model_name, domain) pairs to execute.

    Assignment logic:
    1. Start with the thoroughness preset.
    2. Skip models that are disabled (enabled: false).
    3. Skip models that have no credentials.
    4. Respect per-model prompt overrides (models.<name>.prompts:) — if set,
       that model only runs those domains regardless of the preset.
    5. Deduplicate (model, domain) pairs.
    """
    preset = _THOROUGHNESS_PRESETS.get(thoroughness, _THOROUGHNESS_PRESETS["standard"])
    assignments: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for domain, model_list in preset.items():
        for model_name in model_list:
            cfg = model_configs.get(model_name, {})
            if not cfg.get("enabled", True):
                continue
            # Per-model prompt override: skip this domain if not in the list
            if "prompts" in cfg and domain not in cfg["prompts"]:
                continue
            if not _model_has_credentials(model_name, api_keys, cfg):
                continue
            pair = (model_name, domain)
            if pair not in seen:
                seen.add(pair)
                assignments.append(pair)

    # Handle explicit per-model prompts for models not covered by the preset
    # (e.g. a model configured with prompts: [fact_check] that isn't in standard preset)
    for model_name, cfg in model_configs.items():
        if not cfg.get("enabled", True):
            continue
        if not _model_has_credentials(model_name, api_keys, cfg):
            continue
        for domain in cfg.get("prompts", []):
            pair = (model_name, domain)
            if pair not in seen and domain in _DOMAIN_PROMPTS:
                seen.add(pair)
                assignments.append(pair)

    return assignments


def _build_custom_assignments(pub_config, model_configs, api_keys):
    """Return (assignments, prompts_by_domain) for publication-defined custom domains.

    Each entry in pub_config.custom_domains has:
      prompt:      inline prompt string  (one of these is required)
      prompt_file: path to a .txt file
      models:      list of model names to run for this domain
      weight:      ensemble weight (optional, default 1.0)

    Returns:
      assignments       — list of (model_name, domain_name) tuples
      prompts_by_domain — dict[domain_name, prompt_str]
    """
    custom_domains = pub_config.get("custom_domains") or {}
    assignments = []
    prompts_by_domain = {}

    for domain_name, cfg in custom_domains.items():
        if not isinstance(cfg, dict):
            log.warning("custom_domains.%s must be a mapping — skipping", domain_name)
            continue

        # Resolve prompt text
        if cfg.get("prompt"):
            prompt_str = cfg["prompt"]
        elif cfg.get("prompt_file"):
            p = Path(cfg["prompt_file"])
            if not p.exists():
                log.warning(
                    "Custom domain %r: prompt_file %r not found — skipping",
                    domain_name,
                    str(p),
                )
                continue
            prompt_str = p.read_text(encoding="utf-8")
        else:
            log.warning(
                "Custom domain %r has no prompt or prompt_file — skipping", domain_name
            )
            continue

        prompts_by_domain[domain_name] = prompt_str

        for model_name in cfg.get("models") or []:
            if model_name not in _ADAPTER_MODULES:
                log.warning(
                    "Custom domain %r: unknown model %r — skipping",
                    domain_name,
                    model_name,
                )
                continue
            model_cfg = model_configs.get(model_name, {})
            if not model_cfg.get("enabled", True):
                continue
            if not _model_has_credentials(model_name, api_keys, model_cfg):
                continue
            pair = (model_name, domain_name)
            if pair not in assignments:
                assignments.append(pair)

    return assignments, prompts_by_domain


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    if name not in _PROMPT_CACHE:
        # Prompts are packaged alongside this module, so resolve relative to the
        # package — not the current working directory (which broke when the code
        # moved into packages/ci-article-review and is invoked as a module).
        path = Path(__file__).parent / "prompts" / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def _render_prompt(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value) if value else "")
    return template


def _build_user_prompt(draft: str, handoff: dict) -> str:
    parts = [f"ARTICLE TITLE: {handoff['title']}\n"]
    if handoff.get("primary_claim"):
        parts.append(f"PRIMARY CLAIM: {handoff['primary_claim']}\n")
    if handoff.get("pre_draft_analysis"):
        parts.append(f"PRE-DRAFT ANALYSIS:\n{handoff['pre_draft_analysis']}\n")
    if handoff.get("additional_context"):
        parts.append(f"ADDITIONAL CONTEXT:\n{handoff['additional_context']}\n")
    parts.append(f"\nDRAFT:\n{draft}")
    return "\n".join(parts)


def _read_handoff_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Handoff file not found: {path}\nCheck the path and try again."
        )
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Cannot read handoff file {path}: {e}") from e


# ---------------------------------------------------------------------------
# Generic domain runner
# ---------------------------------------------------------------------------


def _run_domain(
    model_name: str,
    domain: str,
    draft: str,
    handoff: dict,
    pub_config: dict,
    api_keys: dict,
    pipeline_cfg: dict,
    model_configs: dict,
    prompt_str: str | None = None,
) -> dict:
    """Call one model on one domain and return the adapter result dict.

    The result is tagged with ``_model`` and ``_domain`` so the caller can
    build the keyed results dict without re-parsing the runner name string.
    prompt_str overrides the built-in prompt file for custom publication domains.
    """
    adapter = importlib.import_module(_ADAPTER_MODULES[model_name])

    template = (
        prompt_str if prompt_str is not None else _load_prompt(_DOMAIN_PROMPTS[domain])
    )
    style = pub_config.get("style_rules", {})
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        # ci-style-profile emits `style_profile`; fall back to the legacy
        # `voice_profile` key so existing publication.yaml files keep working.
        voice_profile=pub_config.get("style_profile")
        or pub_config.get("voice_profile", ""),
        banned_words=", ".join(style.get("banned_words", [])),
        banned_phrases=", ".join(style.get("banned_phrases", [])),
        positive_rules="\n".join(f"- {r}" for r in style.get("positive_rules", [])),
        primary_claim=handoff.get("primary_claim", ""),
        pre_draft_analysis=handoff.get("pre_draft_analysis", ""),
    )
    user = _build_user_prompt(draft, handoff)
    api_key = api_keys.get(model_name, {}).get("api_key", "")

    result = adapter.call(
        system,
        user,
        api_key,
        retry=pipeline_cfg.get("retry_on_failure", True),
        retry_delay=pipeline_cfg.get("retry_delay_seconds", 10),
        provider_config=model_configs.get(model_name, {}),
    )
    result["_model"] = model_name
    result["_domain"] = domain
    return result


# ---------------------------------------------------------------------------
# Draft mode
# ---------------------------------------------------------------------------


def _global_ceiling(per_task_timeouts, retry_delay):
    """Outer wall-clock bound for the whole parallel batch.

    Must sit comfortably above the slowest task's own timeout so that (a) a task's
    own timeout fires and is collected cleanly rather than being masked as a
    global-ceiling cancellation, and (b) a transient retry (which costs an extra
    retry_delay plus a second attempt) isn't prematurely killed. We allow the
    slowest timeout + one retry_delay + 30s of scheduling/propagation slack.
    """
    if not per_task_timeouts:
        return retry_delay + 30
    return max(per_task_timeouts) + retry_delay + 30


def run_draft_pipeline(
    handoff_path,
    publication_name,
    config_dir="configs",
    cost_preset=None,
    no_timeout=False,
    only_model=None,
    only_domain=None,
    handoff=None,
):
    """Run the full draft review pipeline.

    Normally ``handoff_path`` points at a handoff document that is read and
    parsed here. URL mode (and tests) may instead pass a pre-built ``handoff``
    dict — must contain at least ``title`` and ``draft`` — in which case the
    file read/parse step is skipped and the rest of the pipeline is shared.
    """
    t_start = time.monotonic()

    log.info(f"Loading configs (publication={publication_name})")
    user_config = load_user_config(config_dir)
    pub_config_raw = load_publication_config(publication_name, config_dir)
    if cost_preset:
        # CLI override for this run only — does not modify user.yaml on disk.
        user_config.setdefault("pipeline", {})["cost_preset"] = cost_preset
        log.info(f"cost_preset overridden via --cost-preset: {cost_preset}")
    config = merge_configs(user_config, pub_config_raw)

    # Model currency check — runs against the fully resolved (post-preset) model config.
    currency = check_model_currency(config.get("models", {}))
    for w in currency["warnings"]:
        log.warning(
            f"Model currency: {w['provider']} is using {w['model']!r}, "
            f"which has been superseded by {w['replacement']!r}. "
            + (w["note"] + ". " if w["note"] else "")
            + "Update user.yaml to use the newer model."
        )
    if currency["registry_warning"]:
        log.warning(
            f"Model registry is {currency['registry_age_days']} days old "
            f"(last updated {currency['registry_date']}). "
            "Provider APIs change frequently — re-check available models and pricing."
        )
    elif currency["registry_stale"]:
        log.info(
            f"Model registry last updated {currency['registry_date']} "
            f"({currency['registry_age_days']} days ago). "
            "Consider re-checking for newer models."
        )

    api_keys = config["api_keys"]
    pipeline_cfg = config["pipeline"]
    pub_config = config["publication"]
    delta_cfg = config["delta"]
    ensemble_cfg = config.get("ensemble", {})
    model_configs = config.get("models", {})
    task_timeout = pipeline_cfg.get("task_timeout_seconds", 180)
    thoroughness = pipeline_cfg.get("thoroughness", "standard")

    if handoff is None:
        log.info(f"Parsing draft submission: {handoff_path}")
        handoff_text = _read_handoff_file(handoff_path)
        handoff = parse_draft_submission(handoff_text)
    else:
        log.info(f"Using pre-built handoff: {handoff.get('title', 'Untitled')!r}")

    if not handoff["draft"]:
        log.error(
            "No DRAFT section found in handoff document. "
            "Ensure the document contains a 'DRAFT' header followed by the article text."
        )
        sys.exit(1)

    run_start_ts = datetime.now(timezone.utc)
    run_number = handoff.get("run_number", 1)
    article_title = handoff.get("title", "Untitled")
    lt_config = pub_config.get("languagetool", {})

    # Pass 1: LanguageTool grammar correction (optional)
    grammar_enabled = pipeline_cfg.get("grammar_pass", True)
    lt_creds = api_keys.get("languagetool", {})
    lt_has_creds = bool(lt_creds.get("username") and lt_creds.get("api_key"))

    if not grammar_enabled:
        log.info("Pass 1: Grammar pass disabled (grammar_pass: false) — skipping.")
        lt_result = {
            "failed": True,
            "skipped": True,
            "change_log": [],
            "flagged_matches": [],
        }
        corrected_draft = handoff["draft"]
    elif not lt_has_creds:
        log.info(
            "Pass 1: No LanguageTool credentials configured — skipping grammar pass."
        )
        lt_result = {
            "failed": True,
            "skipped": True,
            "change_log": [],
            "flagged_matches": [],
        }
        corrected_draft = handoff["draft"]
    else:
        log.info("Pass 1: LanguageTool grammar correction")
        from .adapters.grammar import languagetool as lt

        lt_result = lt.run(
            handoff["draft"],
            lt_config,
            lt_creds["username"],
            lt_creds["api_key"],
            retry=pipeline_cfg.get("retry_on_failure", True),
            retry_delay=pipeline_cfg.get("retry_delay_seconds", 10),
        )
        if lt_result["failed"]:
            log.warning(
                f"LanguageTool failed ({lt_result.get('elapsed_seconds', '?')}s): "
                f"{lt_result.get('error')}. Proceeding with uncorrected draft."
            )
            corrected_draft = handoff["draft"]
        else:
            corrected_draft = lt_result["corrected_text"]
            log.info(
                f"LanguageTool: {len(lt_result['change_log'])} corrections "
                f"in {lt_result.get('elapsed_seconds', '?')}s."
            )

    # Pre-analysis: readability, link validation, SEO (no API calls required)
    log.info("Pre-analysis: readability, links, SEO")
    pre_analysis = {}

    pre_analysis["readability"] = readability_analysis.analyze(corrected_draft)
    log.info(
        "Readability: %s words, FK grade %.1f (%s)",
        pre_analysis["readability"]["word_count"],
        pre_analysis["readability"]["flesch_kincaid_grade"],
        pre_analysis["readability"]["reading_level"],
    )

    link_check_enabled = pipeline_cfg.get("link_validation", True)
    if link_check_enabled:
        from .analysis import links as links_analysis

        check_wayback_links = pipeline_cfg.get("wayback_link_check", True)
        # Use `is not None` rather than `or` so an explicit 0 isn't coalesced to the default.
        wayback_stale_days = pipeline_cfg.get("wayback_snapshot_stale_days")
        if wayback_stale_days is not None:
            wayback_stale_days = int(wayback_stale_days)
        log.info(
            "Link validation: scanning for URLs%s",
            " + Wayback check" if check_wayback_links else "",
        )
        pre_analysis["links"] = links_analysis.validate_links(
            corrected_draft,
            check_wayback=check_wayback_links,
            wayback_stale_days=wayback_stale_days,
        )
        broken = [r for r in pre_analysis["links"] if not r.get("ok")]
        not_archived = [
            r
            for r in pre_analysis["links"]
            if r.get("wayback", {}).get("archived") is False
        ]
        log.info(
            "Links: %d found, %d broken/error, %d not archived in Wayback",
            len(pre_analysis["links"]),
            len(broken),
            len(not_archived),
        )
    else:
        pre_analysis["links"] = []

    pre_analysis["seo"] = seo_analysis.analyze(
        corrected_draft,
        handoff,
        seo_rules=pub_config.get("seo_rules"),
    )
    seo_issues = pre_analysis["seo"]["issues"]
    if seo_issues:
        log.info(
            "SEO: %d issue(s): %s",
            len(seo_issues),
            "; ".join(i["type"] for i in seo_issues),
        )
    else:
        log.info("SEO: no issues detected")

    # Sliding-scale timeouts: size the per-model timeout from the draft length, the
    # model, and the reasoning effort. A timeout_seconds set explicitly in user.yaml
    # is treated as an override and left untouched.
    #
    # Since the review adapters stream (SSE), this computed value is the per-task
    # thread WALL-CLOCK BACKSTOP enforced by _run_with_timeout below — NOT a socket
    # timeout. Streaming does not make a long generation finish faster (a gpt-5.5
    # xhigh call still emits tokens for ~800s); it changes the socket read timeout,
    # which the adapters now hold at a small constant inter-token gap (see
    # adapters/review/streaming.py) instead of the whole-generation budget. So this
    # ceiling must still cover the genuine total generation time; what it no longer
    # has to absorb is "model buffered everything and sent nothing" — a stall is now
    # caught in ~120s by the adapter's read-gap timeout, not here. Writing into
    # model_configs means _per_model_timeout below picks it up with no further
    # wiring; the adapters ignore timeout_seconds for the socket entirely.
    char_count = len(corrected_draft)
    if no_timeout:
        # Calibration aid: remove timeout truncation so true completion times are
        # measured, never cut off. Uses a large finite value (interruptible with Ctrl-C).
        CALIBRATION_TIMEOUT = 3600
        for _prov, _cfg in model_configs.items():
            if isinstance(_cfg, dict) and _cfg.get("enabled", True):
                _cfg["timeout_seconds"] = CALIBRATION_TIMEOUT
        log.info(
            "Timeouts DISABLED for calibration (--no-timeout): all models set to %ds",
            CALIBRATION_TIMEOUT,
        )
    else:
        computed_timeouts = timeout_model.compute_all(
            char_count, model_configs, task_timeout
        )
        for _prov, _t in computed_timeouts.items():
            if isinstance(model_configs.get(_prov), dict):
                model_configs[_prov]["timeout_seconds"] = _t
        if computed_timeouts:
            log.info(
                "Timeouts (%d chars): %s",
                char_count,
                ", ".join(f"{p}={t}s" for p, t in sorted(computed_timeouts.items())),
            )

    # Pre-load all prompt files before spawning threads (warms the cache)
    for prompt_name in _DOMAIN_PROMPTS.values():
        _load_prompt(prompt_name)

    # Pass 2: Ensemble model review — built-in domains
    assignments = _build_assignments(thoroughness, model_configs, api_keys)

    # Custom publication-defined domains
    custom_assignments, custom_prompts = _build_custom_assignments(
        pub_config, model_configs, api_keys
    )
    if custom_assignments:
        log.info("Custom domains: %d assignment(s)", len(custom_assignments))
        for model_name, domain in custom_assignments:
            log.info("  Custom: %s → %s", model_name, domain)

    if not assignments:
        log.error(
            "No model assignments could be built. "
            "Check that at least one model has credentials and is enabled."
        )
        sys.exit(1)

    # Calibration filters: restrict the run to a subset so a single cell (e.g.
    # one gpt-5.5 xhigh call) can be measured without the full ensemble.
    if only_model:
        assignments = [a for a in assignments if a[0] == only_model]
        custom_assignments = [a for a in custom_assignments if a[0] == only_model]
    if only_domain:
        assignments = [a for a in assignments if a[1] == only_domain]
        custom_assignments = [a for a in custom_assignments if a[1] == only_domain]
    if (only_model or only_domain) and not (assignments or custom_assignments):
        log.error(
            "No assignments match the calibration filters (--only-model=%r --only-domain=%r). "
            "Check spelling against the configured models and domain names.",
            only_model,
            only_domain,
        )
        sys.exit(1)

    all_assignments = assignments + custom_assignments
    log.info(
        f"Pass 2: Ensemble review — {len(all_assignments)} call(s) "
        f"({len(assignments)} built-in, {len(custom_assignments)} custom), "
        f"thoroughness={thoroughness!r}, "
        f"parallel={pipeline_cfg.get('parallel_review_calls', True)}, "
        f"timeout={task_timeout}s"
    )
    for model_name, domain in assignments:
        log.info(f"  Assigned: {model_name} → {domain}")

    # Build runner list — custom domains pass their prompt string directly
    runners = [
        (
            f"{model_name}:{domain}",
            lambda m=model_name, d=domain, ps=custom_prompts.get(domain): _run_domain(
                m,
                d,
                corrected_draft,
                handoff,
                pub_config,
                api_keys,
                pipeline_cfg,
                model_configs,
                prompt_str=ps,
            ),
        )
        for model_name, domain in all_assignments
    ]

    raw_results: dict[str, dict] = {}

    if pipeline_cfg.get("parallel_review_calls", True):
        # Resolve per-model timeouts. Per-model config key: timeout_seconds.
        # Falls back to the pipeline-level task_timeout_seconds for any model
        # that doesn't set its own. The global ceiling is the maximum of all
        # individual timeouts plus a small scheduling buffer.
        def _per_model_timeout(runner_name):
            m = runner_name.split(":")[0]
            return model_configs.get(m, {}).get("timeout_seconds", task_timeout)

        def _configured_model_id(runner_name):
            m = runner_name.split(":")[0]
            return model_configs.get(m, {}).get("model", m)

        runner_timeouts = [(name, fn, _per_model_timeout(name)) for name, fn in runners]
        global_ceiling = _global_ceiling(
            [t for _, _, t in runner_timeouts],
            pipeline_cfg.get("retry_delay_seconds", 10),
        )

        def _run_with_timeout(fn, timeout, name):
            """Enforce a per-task timeout. Raises TimeoutError on expiry."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
                inner_future = inner.submit(fn)
                try:
                    return inner_future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    inner_future.cancel()
                    raise TimeoutError(f"Timed out after {timeout}s")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(runner_timeouts)
        ) as executor:
            future_to_name = {
                executor.submit(_run_with_timeout, fn, timeout, name): name
                for name, fn, timeout in runner_timeouts
            }
            done, not_done = concurrent.futures.wait(
                future_to_name.keys(),
                timeout=global_ceiling,
            )
            for future in not_done:
                name = future_to_name[future]
                future.cancel()
                log.error(
                    f"Review pass {name} exceeded global ceiling {global_ceiling}s and was cancelled."
                )
                raw_results[name] = {
                    "failed": True,
                    "error": f"Exceeded global timeout of {global_ceiling}s",
                    "model": _configured_model_id(name),
                    "tokens": {},
                    # The call was cancelled at the global ceiling, so we don't know
                    # the true elapsed time — record the ceiling as a lower bound
                    # rather than leaving it None (breaks [CALIBRATION] log parsing).
                    "elapsed_seconds": global_ceiling,
                    "_model": name.split(":")[0],
                    "_domain": name.split(":", 1)[1] if ":" in name else name,
                }
            for future in done:
                name = future_to_name[future]
                try:
                    raw_results[name] = future.result()
                except Exception as e:
                    per_timeout = _per_model_timeout(name)
                    timed_out = "timed out" in str(e).lower()
                    log.error(
                        f"Review pass {name} {'timed out after ' + str(per_timeout) + 's' if timed_out else 'raised exception: ' + str(e)}"
                    )
                    raw_results[name] = {
                        "failed": True,
                        "error": str(e),
                        "model": _configured_model_id(name),
                        "tokens": {},
                        # Best available lower bound on elapsed time — the task ran
                        # at least this long before being cancelled. Avoids "elapsed=Nones"
                        # in the [CALIBRATION] log line for timed-out passes.
                        "elapsed_seconds": per_timeout if timed_out else None,
                        "_model": name.split(":")[0],
                        "_domain": name.split(":", 1)[1] if ":" in name else name,
                    }
    else:
        for name, fn in runners:
            try:
                raw_results[name] = fn()
            except Exception as e:
                log.error(f"Review pass {name} raised exception: {e}")
                raw_results[name] = {
                    "failed": True,
                    "error": str(e),
                    "model": "unknown",
                    "tokens": {},
                    "_model": name.split(":")[0],
                    "_domain": name.split(":", 1)[1] if ":" in name else name,
                }

    # Re-key results as {(model_name, domain): result}
    results: dict[tuple[str, str], dict] = {}
    for result in raw_results.values():
        model_name = result.get("_model", "unknown")
        domain = result.get("_domain", "unknown")
        results[(model_name, domain)] = result

    # Build API call log. Each entry also carries the calibration inputs (effort,
    # the timeout budget that was actually applied, and the draft size) so the
    # report is a self-contained record for retuning configs/timeouts.yaml — no
    # cross-referencing other files needed.
    #
    # On a failed call whose adapter captured the raw response text (e.g.
    # "Malformed JSON response"), a redacted/truncated excerpt is kept in the
    # report so the failure is diagnosable after the fact. The full `raw` field
    # is never persisted — only a bounded excerpt, and only on failure.
    def _raw_excerpt(raw):
        return redact.truncate_excerpt(redact.redact_url_keys(raw))

    api_call_log = []
    for (model_name, domain), result in results.items():
        status_ok = not result.get("failed")
        model_tag = result.get("model", model_name)
        if result.get("fallback_from"):
            model_tag = f"{model_tag} [FALLBACK from {result['fallback_from']}]"
        grounding = " [grounded]" if result.get("grounding_available") else ""
        mcfg = (
            model_configs.get(model_name)
            if isinstance(model_configs.get(model_name), dict)
            else {}
        )
        effort = (
            (mcfg or {}).get("reasoning_effort") or (mcfg or {}).get("effort") or "none"
        )
        budget = (mcfg or {}).get("timeout_seconds")
        elapsed = result.get("elapsed_seconds")
        out_tokens = (result.get("tokens") or {}).get("completion")
        timed_out = "timed out" in str(result.get("error", "")).lower()
        status = "ok" if status_ok else ("timeout" if timed_out else "failed")
        headroom = (
            round(budget - elapsed, 1)
            if (budget is not None and elapsed is not None)
            else None
        )
        log_entry = {
            "pass": f"{model_name}:{domain}",
            "model": f"{model_tag}{grounding}",
            "failed": not status_ok,
            "tokens": result.get("tokens", {}),
            "elapsed_seconds": elapsed,
            "error": result.get("error") if not status_ok else None,
            # calibration fields
            "effort": effort,
            "timeout_budget_seconds": budget,
            "headroom_seconds": headroom,
            "char_count": char_count,
            "status": status,
        }
        if not status_ok and result.get("raw"):
            log_entry["raw_excerpt"] = _raw_excerpt(result["raw"])
        api_call_log.append(log_entry)
        # Machine-facing structured record — one grep-able line per call, persisted
        # to pipeline_history/pipeline_<date>.log for cross-run calibration analysis.
        log.info(
            "[CALIBRATION] model=%s domain=%s effort=%s chars=%s budget=%ss "
            "elapsed=%ss out_tokens=%s status=%s headroom=%ss",
            model_tag.split(" ")[0],
            domain,
            effort,
            char_count,
            budget,
            elapsed,
            out_tokens,
            status,
            headroom,
        )
        if status_ok:
            log.info(
                f"  {model_name}:{domain}: OK "
                f"({result.get('elapsed_seconds', '?')}s, {model_tag}{grounding})"
            )
        else:
            log.warning(
                f"  {model_name}:{domain}: FAILED — {result.get('error', 'unknown error')}"
            )
            if "raw_excerpt" in log_entry:
                log.debug(
                    f"  {model_name}:{domain}: raw response excerpt:\n"
                    f"{log_entry['raw_excerpt']}"
                )

    all_failed = all(r.get("failed") for r in results.values())
    if all_failed and pipeline_cfg.get("abort_if_all_provider_calls_fail", False):
        log.error("All review model calls failed. Aborting.")
        sys.exit(1)

    prior_report = hist.load_prior_report(HISTORY_ROOT, article_title, run_number)

    # Tag ensemble config with thoroughness for the report
    ensemble_cfg_tagged = {**ensemble_cfg, "thoroughness": thoroughness}

    log.info("Consolidating review report")
    report = consolidation.build_report(
        article_title=article_title,
        publication_name=publication_name,
        run_number=run_number,
        corrected_draft=corrected_draft,
        lt_result=lt_result,
        results=results,
        ensemble_cfg=ensemble_cfg_tagged,
        api_call_log=api_call_log,
        prior_report=prior_report,
        primary_claim=handoff.get("primary_claim", ""),
    )

    # Pass 3: Citation resolution — extract factual claims from fact-check results
    citation_sources = pub_config.get("citation_sources", [])
    if citation_sources:
        log.info(
            "Pass 3: Citation resolution (%d source adapters configured)",
            len(citation_sources),
        )
        from .adapters.citation.resolver import resolve_citations

        fact_check = report.get("section_2_fact_check") or {}
        # Pull claim text from outdated, contradicted, and unverifiable lists
        claims = []
        for key in (
            "outdated",
            "contradicted",
            "unverifiable",
            "primary_source_needed",
        ):
            for item in fact_check.get(key, []):
                claim = item.get("claim", "")
                if claim and claim not in claims:
                    claims.append(claim)
        if claims:
            citation_results = resolve_citations(claims, citation_sources)
            resolved_count = sum(1 for r in citation_results if r.get("resolved"))
            log.info(
                "Citations: %d claim(s), %d resolved to primary sources",
                len(claims),
                resolved_count,
            )
        else:
            citation_results = []
            log.info("Citations: no actionable claims to resolve")
        report["section_9_citations"] = citation_results
    else:
        report["section_9_citations"] = []

    # Cost tracking
    cost_summary = cost_analysis.calculate(api_call_log)
    report["cost_summary"] = cost_summary
    log.info(
        "Estimated cost: $%.4f (%s)",
        cost_summary["total_usd"],
        "exact"
        if cost_summary["pricing_known"]
        else "estimated — some model prices unknown",
    )

    # Attach pre-analysis
    report["pre_analysis"] = pre_analysis

    # Fallback warnings
    fallback_warnings = [
        {
            "pass": f"{model}:{domain}",
            "used": r.get("model"),
            "requested": r.get("fallback_from"),
        }
        for (model, domain), r in results.items()
        if r and r.get("fallback_from")
    ]
    if fallback_warnings:
        report["fallback_warnings"] = fallback_warnings

    # Attach currency check results so they appear in the saved report JSON
    # and can be rendered by _print_draft_summary.
    report["model_currency"] = currency

    paths = hist.save_run(
        HISTORY_ROOT,
        article_title,
        run_number,
        report,
        lt_result.get("change_log", []),
        run_ts=run_start_ts,
    )
    if paths["report_path"]:
        log.info(f"Report saved: {paths['report_path']}")

    elapsed_total = round(time.monotonic() - t_start, 1)
    _print_draft_summary(report, delta_cfg, elapsed_total)

    return report


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_draft_summary(report, delta_cfg, elapsed_total=None):
    print("\n" + "=" * 60)
    print(f"REVIEW COMPLETE: {report['article_title']}")
    print(f"Run #{report['run_number']} — {report['generated']}")
    if elapsed_total is not None:
        print(f"Total elapsed: {elapsed_total}s")

    ensemble_meta = report.get("ensemble", {})
    thoroughness = ensemble_meta.get("thoroughness", "standard")
    n_calls = len(ensemble_meta.get("assignments", []))
    print(f"Thoroughness: {thoroughness} ({n_calls} model-domain calls)")
    print("=" * 60)

    if report.get("model_failures"):
        print(
            f"\nWARNING: These model passes failed: {', '.join(report['model_failures'])}"
        )

    if report.get("fallback_warnings"):
        print(f"\n{'!' * 60}")
        print(
            "WARNING: One or more passes ran on fallback models due to capacity limits."
        )
        print(
            "Results from fallback models may be less thorough than preferred models."
        )
        for fw in report["fallback_warnings"]:
            print(
                f"  {fw['pass']}: ran on '{fw['used']}' (preferred: '{fw['requested']}')"
            )
        print("Re-run when preferred models are available to confirm findings.")
        print("!" * 60)

    currency = report.get("model_currency")
    if currency:
        if currency.get("warnings"):
            print(f"\n{'!' * 60}")
            print("MODEL CURRENCY: Outdated model(s) detected — update user.yaml")
            for w in currency["warnings"]:
                note = f" ({w['note']})" if w.get("note") else ""
                print(
                    f"  {w['provider']}: {w['model']!r} → replace with {w['replacement']!r}{note}"
                )
            print("!" * 60)
        if currency.get("notices"):
            print("\nModel upgrades available (optional):")
            for n in currency["notices"]:
                print(
                    f"  {n['provider']}: {n['model']!r} — {n.get('note') or 'newer: ' + n['newer']}"
                )
        age = currency.get("registry_age_days", 0)
        reg_date = currency.get("registry_date", "unknown")
        if currency.get("registry_warning"):
            print(
                f"\n{'!' * 60}\n"
                f"MODEL REGISTRY is {age} days old (last updated {reg_date}).\n"
                f"Provider APIs change frequently — re-check models, pricing, and\n"
                f"reasoning flags before your next run.  See docs/PROVIDERS.md.\n"
                f"{'!' * 60}"
            )
        elif currency.get("registry_stale"):
            print(
                f"\nNote: model registry last updated {reg_date} ({age} days ago). "
                "Consider re-checking for newer models."
            )
        else:
            print(
                f"\nModel registry: current (last updated {reg_date}, {age} days ago)"
            )

    if report.get("lt_skipped"):
        print(
            "\nGrammar pass: skipped (no LanguageTool credentials — run manual Grammarly pass before publishing)"
        )
    elif report.get("lt_failed"):
        print("\nWARNING: LanguageTool failed — draft not grammar-corrected")
    else:
        print(
            f"\nLanguageTool: {len(report['lt_corrections_applied'])} corrections applied"
        )

    print(f"\nSection 1 — Consensus flags: {len(report['section_1_consensus'])}")
    fact = report["section_2_fact_check"]
    fact_count = (
        sum(len(v) for v in fact.values() if isinstance(v, list)) if fact else 0
    )
    print(f"Section 2 — Fact check items: {fact_count}")
    print(f"Section 3 — Voice flags: {len(report['section_3_voice'])}")
    print(f"Section 4 — Argument flags: {len(report['section_4_argument'])}")
    print(f"Section 5 — Completeness flags: {len(report['section_5_completeness'])}")

    rt = report["section_6_red_team"]
    if not rt:
        _rt_display = "none"
    elif "most_vulnerable_claim" in rt:
        _rt_display = "3 findings (1 source)"
    else:
        n_src = len(rt)
        _rt_display = f"{n_src * 3} findings ({n_src} sources)"
    print(f"Section 6 — Red team: {_rt_display}")
    print(
        f"Section 7 — Low-confidence flags: {len(report['section_7_low_confidence'])}"
    )

    additional = report.get("section_8_additional", [])
    if additional:
        by_cat: dict[str, int] = {}
        for obs in additional:
            cat = obs.get("category", "unknown")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        cat_summary = ", ".join(f"{c}: {n}" for c, n in sorted(by_cat.items()))
        print(
            f"Section 8 — Cross-model observations: {len(additional)} ({cat_summary})"
        )
    else:
        print("Section 8 — Cross-model observations: 0")

    # Pre-analysis block
    pre = report.get("pre_analysis", {})
    if pre:
        r = pre.get("readability", {})
        if r:
            print(
                f"\nReadability: {r['word_count']} words, {r['sentence_count']} sentences, "
                f"FK grade {r['flesch_kincaid_grade']} ({r['reading_level']}), "
                f"avg sentence {r['avg_sentence_length']} words"
            )
        seo = pre.get("seo", {})
        if seo:
            seo_issues = seo.get("issues", [])
            if seo_issues:
                print(f"SEO issues ({len(seo_issues)}):")
                for iss in seo_issues:
                    print(f"  [{iss['type']}] {iss['detail']}")
            else:
                print("SEO: no issues")
        links = pre.get("links", [])
        if links:
            broken = [lk for lk in links if not lk.get("ok")]
            not_archived = [
                lk for lk in links if lk.get("wayback", {}).get("archived") is False
            ]
            stale_archive = [
                lk
                for lk in links
                if lk.get("wayback", {}).get("snapshot_stale") is True
            ]
            print(f"\nLinks: {len(links)} found", end="")
            if broken:
                print(f", {len(broken)} broken/error", end="")
            if not_archived:
                print(f", {len(not_archived)} not archived", end="")
            if stale_archive:
                print(f", {len(stale_archive)} stale archive", end="")
            print()
            for lk in links:
                status = (
                    "OK"
                    if lk.get("ok")
                    else f"BROKEN ({lk.get('status_code') or lk.get('error', '?')})"
                )
                wb = lk.get("wayback", {})
                age = wb.get("snapshot_age_days")
                if wb.get("is_archive_url"):
                    # The cited link is itself a Wayback snapshot; HTTP status (OK/BROKEN
                    # above) is the functional check. Flag staleness from its own timestamp.
                    stale = " STALE" if wb.get("snapshot_stale") else ""
                    wb_str = (
                        f"archive link{stale} ({age}d)"
                        if age is not None
                        else "archive link"
                    )
                elif wb.get("archived") is False:
                    wb_str = "not archived"
                elif wb.get("snapshot_stale"):
                    wb_str = f"stale archive ({age}d)"
                elif wb.get("archived"):
                    wb_str = f"archived ({age}d)"
                else:
                    wb_str = ""
                extras = f"  [{wb_str}]" if wb_str else ""
                redirect = (
                    f" → {lk['redirected_to']}" if lk.get("redirected_to") else ""
                )
                print(f"  {status:30s} {lk['url'][:60]}{redirect}{extras}")

    if report.get("api_call_log"):
        print("\nAPI call times  (elapsed / budget — headroom shows timeout margin):")
        for entry in report["api_call_log"]:
            status = "OK" if not entry["failed"] else "FAILED"
            elapsed = (
                f"{entry['elapsed_seconds']}s"
                if entry.get("elapsed_seconds") is not None
                else "?"
            )
            budget = entry.get("timeout_budget_seconds")
            head = entry.get("headroom_seconds")
            budget_str = f"/{budget}s" if budget is not None else ""
            head_str = f"(+{head}s)" if head is not None else ""
            effort = entry.get("effort", "")
            tokens = entry.get("tokens", {})
            tok_str = (
                f"{tokens.get('prompt', 0)}+{tokens.get('completion', 0)} tok"
                if tokens
                else ""
            )
            print(
                f"  {entry['pass']:30s} {status:6s} {elapsed:>8s}{budget_str:>7s} {head_str:>10s}  "
                f"{entry['model']}  effort={effort}  {tok_str}"
            )

    delta = report.get("delta")
    if delta:
        print("\nDelta from prior run:")
        print(f"  Word change: {delta['word_change_pct']}%")
        print(
            f"  Resolved consensus flags: {delta['resolved_consensus_count']}/{delta['prior_consensus_count']}"
        )
        print(f"  New consensus flags: {delta['new_consensus_count']}")
        if delta.get("claim_changed"):
            print("  Primary claim: CHANGED since prior run")
        if delta.get("structure_changed"):
            print("  Heading structure: CHANGED since prior run")
        if consolidation.rerun_recommended(delta, delta_cfg):
            print(
                "\n  RECOMMENDATION: Re-run after editing (significant changes detected)"
            )
        else:
            print(
                "\n  RECOMMENDATION: Draft appears stable — proceed to Grammarly pass"
            )

    # Cost summary
    cost = report.get("cost_summary")
    if cost:
        known_flag = (
            ""
            if cost["pricing_known"]
            else " (estimated — unknown model in pricing table)"
        )
        print(f"\nEstimated cost: ${cost['total_usd']:.4f}{known_flag}")
        if cost.get("by_pass"):
            for entry in cost["by_pass"]:
                if entry["total_usd"] > 0:
                    print(
                        f"  {entry['pass']:30s}  ${entry['total_usd']:.4f}  {entry['model']}"
                    )

    # Contradiction summary
    contradictions = report.get("contradictions", [])
    if contradictions:
        print(f"\n{'!' * 60}")
        print(
            f"FACT-CHECK CONTRADICTIONS: {len(contradictions)} claim(s) confirmed by one model, challenged by another"
        )
        for c in contradictions:
            print(
                f"  [{c['challenge_type'].upper()}] confirmed by {', '.join(c['confirmed_by'])}; "
                f"challenged by {', '.join(c['challenged_by'])}"
            )
            print(f"    Claim: {c['claim'][:100]}")
        print("!" * 60)

    # Citation resolution summary
    citations = report.get("section_9_citations", [])
    if citations:
        resolved = [c for c in citations if c.get("resolved")]
        not_archived = [
            c for c in resolved if c.get("wayback", {}).get("archived") is False
        ]
        stale = [c for c in resolved if c.get("wayback", {}).get("snapshot_stale")]
        print(f"\nSection 9 — Citations: {len(resolved)}/{len(citations)} resolved")
        if not_archived:
            print(
                f"  {len(not_archived)} resolved URL(s) not yet archived in Wayback Machine"
            )
        if stale:
            print(
                f"  {len(stale)} resolved URL(s) have a stale Wayback snapshot (>180 days)"
            )

    print(
        f"\nFull report: {HISTORY_ROOT}/{hist._slug(report.get('article_title', ''))}/"
    )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Publish mode
# ---------------------------------------------------------------------------


def run_publish_pipeline(
    handoff_path, publication_name, publish_live=False, config_dir="configs"
):
    log.info(f"Loading configs (publication={publication_name})")
    user_config = load_user_config(config_dir)
    pub_config_raw = load_publication_config(publication_name, config_dir)
    config = merge_configs(user_config, pub_config_raw)

    pub_config = config["publication"]
    wp_config = pub_config.get("wordpress", {})
    rank_math_config = pub_config.get("rank_math", {})

    log.info(f"Parsing publication handoff: {handoff_path}")
    handoff_text = _read_handoff_file(handoff_path)
    pub_handoff = parse_publication_handoff(handoff_text)

    if not pub_handoff["final_draft"]:
        log.error(
            "No FINAL DRAFT section found in publication handoff. "
            "Ensure the document contains a 'FINAL DRAFT' header followed by the article text."
        )
        sys.exit(1)

    from .adapters.cms import wordpress as wp

    confirmed = wp.print_checklist_and_confirm()
    if not confirmed:
        sys.exit(0)

    pub_params = {
        "title": pub_handoff.get("title", ""),
        "wordpress_category": pub_handoff["publication_parameters"].get(
            "wordpress_category"
        ),
        "tags": [
            t.strip()
            for t in pub_handoff["publication_parameters"].get("tags", "").split(",")
            if t.strip()
        ],
        "author": pub_handoff["publication_parameters"].get("author"),
        "seo": pub_handoff.get("seo", {}),
    }

    content = pub_handoff["final_draft"]

    log.info(f"Pushing to WordPress (status={'publish' if publish_live else 'draft'})")
    result = wp.push(
        content, pub_params, wp_config, rank_math_config, publish_live=publish_live
    )

    if result["success"]:
        print("\nWordPress push successful.")
        print(f"Post URL: {result['post_url']}")
        print(f"Post ID:  {result['post_id']}")
        print(
            f"Status:   {'PUBLISHED' if publish_live else 'DRAFT (pass --publish-live to publish)'}"
        )
    else:
        print(f"\nWordPress push FAILED: {result['error']}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Article Review Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --draft handoff.md --publication myblog\n"
            "  python pipeline.py --url https://example.com/post --publication myblog\n"
            "  python pipeline.py --publish pub_handoff.md --publication myblog --publish-live\n"
            "  python pipeline.py --draft handoff.md --publication myblog --verbose\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--draft",
        metavar="HANDOFF_PATH",
        help="Path to draft submission handoff document",
    )
    group.add_argument(
        "--url",
        metavar="URL",
        help="Fetch a published web page and run the review on its extracted content",
    )
    group.add_argument(
        "--publish", metavar="HANDOFF_PATH", help="Path to publication handoff document"
    )
    parser.add_argument(
        "--publication", required=True, help="Publication config name (without .yaml)"
    )
    parser.add_argument(
        "--publish-live",
        action="store_true",
        help="Publish to live (default: save as draft)",
    )
    parser.add_argument(
        "--config-dir", default="configs", help="Directory containing config files"
    )
    parser.add_argument(
        "--cost-preset",
        choices=["economy", "standard", "balanced", "thorough", "maximum"],
        help="Override cost_preset from user.yaml for this run only (useful for calibration sweeps)",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Calibration: disable timeout truncation so true completion times are measured",
    )
    parser.add_argument(
        "--only-model",
        metavar="PROVIDER",
        help="Calibration: run only this provider (e.g. openai) instead of the full ensemble",
    )
    parser.add_argument(
        "--only-domain",
        metavar="DOMAIN",
        help="Calibration: run only this domain (e.g. fact_check, voice_style, completeness, "
        "argument_integrity, red_team)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )

    args = parser.parse_args()

    if args.publish_live and args.draft:
        print(
            "WARNING: --publish-live has no effect in --draft mode and will be ignored."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Also write all log output to a persistent file so warnings aren't lost on scroll.
    # Daily rotation: one file per UTC day; same-day runs append to the same file.
    _log_dir = Path(HISTORY_ROOT)
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    _file_handler = logging.FileHandler(
        _log_dir / f"pipeline_{_log_date}.log", encoding="utf-8"
    )
    _file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_file_handler)

    try:
        validate_publication_name(args.publication)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    try:
        if args.draft:
            run_draft_pipeline(
                args.draft,
                args.publication,
                config_dir=args.config_dir,
                cost_preset=args.cost_preset,
                no_timeout=args.no_timeout,
                only_model=args.only_model,
                only_domain=args.only_domain,
            )
        elif args.url:
            handoff = build_handoff_from_url(args.url)
            run_draft_pipeline(
                None,
                args.publication,
                config_dir=args.config_dir,
                cost_preset=args.cost_preset,
                no_timeout=args.no_timeout,
                only_model=args.only_model,
                only_domain=args.only_domain,
                handoff=handoff,
            )
        elif args.publish:
            run_publish_pipeline(
                args.publish,
                args.publication,
                publish_live=args.publish_live,
                config_dir=args.config_dir,
            )
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to fetch URL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
