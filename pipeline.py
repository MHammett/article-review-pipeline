#!/usr/bin/env python3
"""
Article Review Pipeline — orchestration engine.

Usage:
  python pipeline.py --draft path/to/handoff.md --publication your_publication_name
  python pipeline.py --publish path/to/publication_handoff.md --publication your_publication_name [--publish-live]
"""
import sys

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
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from config_loader import load_user_config, load_publication_config, merge_configs, validate_publication_name
from handoff_parser import parse_draft_submission, parse_publication_handoff
import history as hist
import consolidation

log = logging.getLogger("pipeline")

HISTORY_ROOT = "pipeline_history"

# Module-level prompt cache — files are read once per process lifetime.
_PROMPT_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt(name):
    if name not in _PROMPT_CACHE:
        path = Path("prompts") / name
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                "Ensure you are running pipeline.py from the project root directory."
            )
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def _render_prompt(template, **kwargs):
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value) if value else "")
    return template


def _build_user_prompt(draft, handoff):
    parts = [f"ARTICLE TITLE: {handoff['title']}\n"]
    if handoff.get("primary_claim"):
        parts.append(f"PRIMARY CLAIM: {handoff['primary_claim']}\n")
    if handoff.get("pre_draft_analysis"):
        parts.append(f"PRE-DRAFT ANALYSIS:\n{handoff['pre_draft_analysis']}\n")
    if handoff.get("additional_context"):
        parts.append(f"ADDITIONAL CONTEXT:\n{handoff['additional_context']}\n")
    parts.append(f"\nDRAFT:\n{draft}")
    return "\n".join(parts)


def _read_handoff_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Handoff file not found: {path}\n"
            "Check the path and try again."
        )
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Cannot read handoff file {path}: {e}") from e


# ---------------------------------------------------------------------------
# Review pass runners
# ---------------------------------------------------------------------------

def _run_gemini(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import gemini
    template = _load_prompt("fact_check.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        voice_profile=pub_config.get("voice_profile", ""),
    )
    user = _build_user_prompt(draft, handoff)
    return gemini.call(
        system, user,
        api_keys["gemini"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("gemini"),
    )


def _run_openai_voice(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import openai as oai
    template = _load_prompt("ai_speak.txt")
    style = pub_config.get("style_rules", {})
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        voice_profile=pub_config.get("voice_profile", ""),
        banned_words=", ".join(style.get("banned_words", [])),
        banned_phrases=", ".join(style.get("banned_phrases", [])),
        positive_rules="\n".join(f"- {r}" for r in style.get("positive_rules", [])),
    )
    user = _build_user_prompt(draft, handoff)
    return oai.call(
        system, user,
        api_keys["openai"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("openai"),
    )


def _run_mistral_argument(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import mistral
    template = _load_prompt("argument_integrity.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        primary_claim=handoff.get("primary_claim", ""),
        pre_draft_analysis=handoff.get("pre_draft_analysis", ""),
    )
    user = _build_user_prompt(draft, handoff)
    return mistral.call(
        system, user,
        api_keys["mistral"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("mistral"),
    )


def _run_openai_completeness(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import openai as oai
    template = _load_prompt("completeness.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        primary_claim=handoff.get("primary_claim", ""),
        pre_draft_analysis=handoff.get("pre_draft_analysis", ""),
    )
    user = _build_user_prompt(draft, handoff)
    return oai.call(
        system, user,
        api_keys["openai"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("openai"),
    )


def _run_mistral_redteam(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import mistral
    template = _load_prompt("red_team.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        primary_claim=handoff.get("primary_claim", ""),
    )
    user = _build_user_prompt(draft, handoff)
    return mistral.call(
        system, user,
        api_keys["mistral"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("mistral"),
    )


def _run_grok_redteam(draft, handoff, pub_config, api_keys, pipeline_config, model_overrides):
    from adapters.review import grok
    template = _load_prompt("red_team.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        primary_claim=handoff.get("primary_claim", ""),
    )
    user = _build_user_prompt(draft, handoff)
    return grok.call(
        system, user,
        api_keys["grok"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
        model=model_overrides.get("grok"),
    )


# ---------------------------------------------------------------------------
# Draft mode
# ---------------------------------------------------------------------------

def run_draft_pipeline(handoff_path, publication_name, config_dir="configs"):
    t_start = time.monotonic()

    log.info(f"Loading configs (publication={publication_name})")
    user_config = load_user_config(config_dir)
    pub_config_raw = load_publication_config(publication_name, config_dir)
    config = merge_configs(user_config, pub_config_raw)

    api_keys = config["api_keys"]
    pipeline_cfg = config["pipeline"]
    pub_config = config["publication"]
    delta_cfg = config["delta"]
    model_overrides = config.get("models", {})
    task_timeout = pipeline_cfg.get("task_timeout_seconds", 180)

    log.info(f"Parsing draft submission: {handoff_path}")
    handoff_text = _read_handoff_file(handoff_path)
    handoff = parse_draft_submission(handoff_text)

    if not handoff["draft"]:
        log.error(
            "No DRAFT section found in handoff document. "
            "Ensure the document contains a 'DRAFT' header followed by the article text."
        )
        sys.exit(1)

    run_number = handoff.get("run_number", 1)
    article_title = handoff.get("title", "Untitled")
    lt_config = pub_config.get("languagetool", {})

    # Pass 1: LanguageTool grammar correction (optional)
    grammar_enabled = pipeline_cfg.get("grammar_pass", True)
    lt_creds = api_keys.get("languagetool", {})
    lt_has_creds = bool(lt_creds.get("username") and lt_creds.get("api_key"))

    if not grammar_enabled:
        log.info("Pass 1: Grammar pass disabled (grammar_pass: false) — skipping.")
        lt_result = {"failed": True, "skipped": True, "change_log": [], "flagged_matches": []}
        corrected_draft = handoff["draft"]
    elif not lt_has_creds:
        log.info("Pass 1: No LanguageTool credentials configured — skipping grammar pass.")
        lt_result = {"failed": True, "skipped": True, "change_log": [], "flagged_matches": []}
        corrected_draft = handoff["draft"]
    else:
        log.info("Pass 1: LanguageTool grammar correction")
        from adapters.grammar import languagetool as lt
        lt_result = lt.run(
            handoff["draft"],
            lt_config,
            lt_creds["username"],
            lt_creds["api_key"],
            retry=pipeline_cfg.get("retry_on_failure", True),
            retry_delay=pipeline_cfg.get("retry_delay_seconds", 10),
        )
        if lt_result["failed"]:
            log.warning(f"LanguageTool failed ({lt_result.get('elapsed_seconds', '?')}s): {lt_result.get('error')}. Proceeding with uncorrected draft.")
            corrected_draft = handoff["draft"]
        else:
            corrected_draft = lt_result["corrected_text"]
            log.info(f"LanguageTool: {len(lt_result['change_log'])} corrections in {lt_result.get('elapsed_seconds', '?')}s.")

    # Pre-load all prompt files before spawning threads (warms the cache)
    for prompt_name in ("fact_check.txt", "ai_speak.txt", "argument_integrity.txt", "completeness.txt", "red_team.txt"):
        _load_prompt(prompt_name)

    # Pass 2: Parallel model review
    log.info(f"Pass 2: Multi-model review (5 calls, parallel={pipeline_cfg.get('parallel_review_calls', True)}, timeout={task_timeout}s)")
    api_call_log = []

    runners = [
        ("gemini_fact_check", _run_gemini),
        ("openai_voice", _run_openai_voice),
        ("mistral_argument", _run_mistral_argument),
        ("openai_completeness", _run_openai_completeness),
        ("mistral_red_team", _run_mistral_redteam),
    ]

    # Grok is optional — added when credentials are present
    if api_keys.get("grok", {}).get("api_key"):
        runners.append(("grok_red_team", _run_grok_redteam))
        log.info("Grok API key found — adding Grok red team pass.")

    results = {}

    if pipeline_cfg.get("parallel_review_calls", True):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_name = {
                executor.submit(fn, corrected_draft, handoff, pub_config, api_keys, pipeline_cfg, model_overrides): name
                for name, fn in runners
            }
            done, not_done = concurrent.futures.wait(
                future_to_name.keys(),
                timeout=task_timeout,
            )
            for future in not_done:
                name = future_to_name[future]
                future.cancel()
                log.error(f"Review pass {name} timed out after {task_timeout}s and was cancelled.")
                results[name] = {"failed": True, "error": f"Timed out after {task_timeout}s", "pass": name, "model": "unknown", "tokens": {}}

            for future in done:
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    log.error(f"Review pass {name} raised exception: {e}")
                    results[name] = {"failed": True, "error": str(e), "pass": name, "model": "unknown", "tokens": {}}
    else:
        for name, fn in runners:
            try:
                results[name] = fn(corrected_draft, handoff, pub_config, api_keys, pipeline_cfg, model_overrides)
            except Exception as e:
                log.error(f"Review pass {name} raised exception: {e}")
                results[name] = {"failed": True, "error": str(e), "pass": name, "model": "unknown", "tokens": {}}

    # Build API call log with timing
    pass_labels = {
        "gemini_fact_check": "fact_check",
        "openai_voice": "voice",
        "mistral_argument": "argument",
        "openai_completeness": "completeness",
        "mistral_red_team": "red_team",
        "grok_red_team": "red_team_grok",
    }
    for name, result in results.items():
        api_call_log.append({
            "pass": pass_labels.get(name, name),
            "model": result.get("model", "unknown"),
            "failed": result.get("failed", False),
            "tokens": result.get("tokens", {}),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "error": result.get("error") if result.get("failed") else None,
        })
        if not result.get("failed"):
            log.info(f"  {name}: OK ({result.get('elapsed_seconds', '?')}s, {result.get('model', '?')})")
        else:
            log.warning(f"  {name}: FAILED — {result.get('error', 'unknown error')}")

    all_failed = all(r.get("failed") for r in results.values())
    if all_failed and pipeline_cfg.get("abort_if_all_provider_calls_fail", False):
        log.error("All review model calls failed. Aborting.")
        sys.exit(1)

    prior_report = hist.load_prior_report(HISTORY_ROOT, article_title, run_number)

    log.info("Consolidating review report")
    report = consolidation.build_report(
        article_title=article_title,
        publication_name=publication_name,
        run_number=run_number,
        corrected_draft=corrected_draft,
        lt_result=lt_result,
        gemini_result=results.get("gemini_fact_check", {"failed": True}),
        openai_voice_result=results.get("openai_voice", {"failed": True}),
        mistral_argument_result=results.get("mistral_argument", {"failed": True}),
        openai_completeness_result=results.get("openai_completeness", {"failed": True}),
        mistral_redteam_result=results.get("mistral_red_team", {"failed": True}),
        grok_redteam_result=results.get("grok_red_team"),
        api_call_log=api_call_log,
        prior_report=prior_report,
    )

    paths = hist.save_run(
        HISTORY_ROOT, article_title, run_number,
        report, lt_result.get("change_log", [])
    )
    if paths["report_path"]:
        log.info(f"Report saved: {paths['report_path']}")

    elapsed_total = round(time.monotonic() - t_start, 1)
    _print_draft_summary(report, delta_cfg, elapsed_total)

    return report


def _print_draft_summary(report, delta_cfg, elapsed_total=None):
    print("\n" + "=" * 60)
    print(f"REVIEW COMPLETE: {report['article_title']}")
    print(f"Run #{report['run_number']} — {report['generated']}")
    if elapsed_total is not None:
        print(f"Total elapsed: {elapsed_total}s")
    print("=" * 60)

    if report.get("model_failures"):
        print(f"\nWARNING: These model passes failed: {', '.join(report['model_failures'])}")

    if report.get("lt_skipped"):
        print("\nGrammar pass: skipped (no LanguageTool credentials — run manual Grammarly pass before publishing)")
    elif report.get("lt_failed"):
        print("\nWARNING: LanguageTool failed — draft not grammar-corrected")
    else:
        print(f"\nLanguageTool: {len(report['lt_corrections_applied'])} corrections applied")

    print(f"\nSection 1 — Consensus flags: {len(report['section_1_consensus'])}")
    fact = report['section_2_fact_check']
    fact_count = sum(len(v) for v in fact.values() if isinstance(v, list)) if fact else 0
    print(f"Section 2 — Fact check items: {fact_count}")
    print(f"Section 3 — Voice flags: {len(report['section_3_voice'])}")
    print(f"Section 4 — Argument flags: {len(report['section_4_argument'])}")
    print(f"Section 5 — Completeness flags: {len(report['section_5_completeness'])}")
    print(f"Section 6 — Red team: {'3 findings' if report['section_6_red_team'] else 'none'}")
    print(f"Section 7 — Low-confidence flags: {len(report['section_7_low_confidence'])}")

    if report.get("api_call_log"):
        print("\nAPI call times:")
        for entry in report["api_call_log"]:
            status = "OK" if not entry["failed"] else "FAILED"
            elapsed = f"{entry['elapsed_seconds']}s" if entry.get("elapsed_seconds") is not None else "?"
            tokens = entry.get("tokens", {})
            tok_str = f"{tokens.get('prompt', 0)}+{tokens.get('completion', 0)} tok" if tokens else ""
            print(f"  {entry['pass']:22s} {status:6s} {elapsed:>8s}  {entry['model']}  {tok_str}")

    delta = report.get("delta")
    if delta:
        print(f"\nDelta from prior run:")
        print(f"  Word change: {delta['word_change_pct']}%")
        print(f"  Resolved consensus flags: {delta['resolved_consensus_count']}/{delta['prior_consensus_count']}")
        print(f"  New consensus flags: {delta['new_consensus_count']}")
        if consolidation.rerun_recommended(delta, delta_cfg):
            print("\n  RECOMMENDATION: Re-run after editing (significant changes detected)")
        else:
            print("\n  RECOMMENDATION: Draft appears stable — proceed to Grammarly pass")

    slug = report.get("article_title", "")
    print(f"\nFull report: {HISTORY_ROOT}/{slug}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Publish mode
# ---------------------------------------------------------------------------

def run_publish_pipeline(handoff_path, publication_name, publish_live=False, config_dir="configs"):
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

    from adapters.cms import wordpress as wp

    confirmed = wp.print_checklist_and_confirm()
    if not confirmed:
        sys.exit(0)

    pub_params = {
        "title": pub_handoff.get("title", ""),
        "wordpress_category": pub_handoff["publication_parameters"].get("wordpress_category"),
        "tags": [t.strip() for t in pub_handoff["publication_parameters"].get("tags", "").split(",") if t.strip()],
        "author": pub_handoff["publication_parameters"].get("author"),
        "seo": pub_handoff.get("seo", {}),
    }

    content = pub_handoff["final_draft"]

    log.info(f"Pushing to WordPress (status={'publish' if publish_live else 'draft'})")
    result = wp.push(content, pub_params, wp_config, rank_math_config, publish_live=publish_live)

    if result["success"]:
        print(f"\nWordPress push successful.")
        print(f"Post URL: {result['post_url']}")
        print(f"Post ID:  {result['post_id']}")
        print(f"Status:   {'PUBLISHED' if publish_live else 'DRAFT (pass --publish-live to publish)'}")
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
            "  python pipeline.py --publish pub_handoff.md --publication myblog --publish-live\n"
            "  python pipeline.py --draft handoff.md --publication myblog --verbose\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draft", metavar="HANDOFF_PATH", help="Path to draft submission handoff document")
    group.add_argument("--publish", metavar="HANDOFF_PATH", help="Path to publication handoff document")
    parser.add_argument("--publication", required=True, help="Publication config name (without .yaml)")
    parser.add_argument("--publish-live", action="store_true", help="Publish to live (default: save as draft)")
    parser.add_argument("--config-dir", default="configs", help="Directory containing config files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        validate_publication_name(args.publication)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    try:
        if args.draft:
            run_draft_pipeline(args.draft, args.publication, config_dir=args.config_dir)
        elif args.publish:
            run_publish_pipeline(args.publish, args.publication, publish_live=args.publish_live, config_dir=args.config_dir)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
