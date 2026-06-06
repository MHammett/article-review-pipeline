#!/usr/bin/env python3
"""
Article Review Pipeline — orchestration engine.

Usage:
  python pipeline.py --draft path/to/handoff.md --publication your_publication_name
  python pipeline.py --publish path/to/publication_handoff.md --publication your_publication_name [--publish-live]
"""
import argparse
import concurrent.futures
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config_loader import load_user_config, load_publication_config, merge_configs
from handoff_parser import parse_draft_submission, parse_publication_handoff
import history as hist
import consolidation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

HISTORY_ROOT = "pipeline_history"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _load_prompt(name):
    path = Path("prompts") / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Review pass runners
# ---------------------------------------------------------------------------

def _run_gemini(draft, handoff, pub_config, api_keys, pipeline_config):
    from adapters.review import gemini
    template = _load_prompt("fact_check.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        voice_profile=pub_config.get("voice_profile", ""),
    )
    user = _build_user_prompt(draft, handoff)
    result = gemini.call(
        system, user,
        api_keys["gemini"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
    )
    result["pass"] = "fact_check"
    return result


def _run_openai_voice(draft, handoff, pub_config, api_keys, pipeline_config):
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
    result = oai.call(
        system, user,
        api_keys["openai"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
    )
    result["pass"] = "voice"
    return result


def _run_mistral_argument(draft, handoff, pub_config, api_keys, pipeline_config):
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
    result = mistral.call(
        system, user,
        api_keys["mistral"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
    )
    result["pass"] = "argument"
    return result


def _run_openai_completeness(draft, handoff, pub_config, api_keys, pipeline_config):
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
    result = oai.call(
        system, user,
        api_keys["openai"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
    )
    result["pass"] = "completeness"
    return result


def _run_mistral_redteam(draft, handoff, pub_config, api_keys, pipeline_config):
    from adapters.review import mistral
    template = _load_prompt("red_team.txt")
    system = _render_prompt(
        template,
        publication_description=pub_config.get("publication_description", ""),
        audience=str(pub_config.get("audience", {})),
        primary_claim=handoff.get("primary_claim", ""),
    )
    user = _build_user_prompt(draft, handoff)
    result = mistral.call(
        system, user,
        api_keys["mistral"]["api_key"],
        retry=pipeline_config.get("retry_on_failure", True),
        retry_delay=pipeline_config.get("retry_delay_seconds", 10),
    )
    result["pass"] = "red_team"
    return result


# ---------------------------------------------------------------------------
# Draft mode
# ---------------------------------------------------------------------------

def run_draft_pipeline(handoff_path, publication_name, config_dir="configs"):
    log.info(f"Loading configs (publication={publication_name})")
    user_config = load_user_config(config_dir)
    pub_config_raw = load_publication_config(publication_name, config_dir)
    config = merge_configs(user_config, pub_config_raw)

    api_keys = config["api_keys"]
    pipeline_cfg = config["pipeline"]
    pub_config = config["publication"]
    delta_cfg = config["delta"]

    log.info(f"Parsing draft submission: {handoff_path}")
    handoff_text = Path(handoff_path).read_text(encoding="utf-8")
    handoff = parse_draft_submission(handoff_text)

    if not handoff["draft"]:
        log.error("No DRAFT section found in handoff document.")
        sys.exit(1)

    run_number = handoff.get("run_number", 1)
    article_title = handoff.get("title", "Untitled")
    lt_config = pub_config.get("languagetool", {})

    # Pass 1: LanguageTool
    log.info("Pass 1: LanguageTool grammar correction")
    from adapters.grammar import languagetool as lt
    lt_result = lt.run(
        handoff["draft"],
        lt_config,
        api_keys["languagetool"]["username"],
        api_keys["languagetool"]["api_key"],
        retry=pipeline_cfg.get("retry_on_failure", True),
        retry_delay=pipeline_cfg.get("retry_delay_seconds", 10),
    )

    if lt_result["failed"]:
        log.warning(f"LanguageTool failed: {lt_result.get('error')}. Proceeding with uncorrected draft.")
        corrected_draft = handoff["draft"]
    else:
        corrected_draft = lt_result["corrected_text"]
        log.info(f"LanguageTool applied {len(lt_result['change_log'])} corrections.")

    # Pass 2: Parallel model review
    log.info("Pass 2: Multi-model review (5 calls)")
    api_call_log = []

    runners = [
        ("gemini_fact_check", _run_gemini),
        ("openai_voice", _run_openai_voice),
        ("mistral_argument", _run_mistral_argument),
        ("openai_completeness", _run_openai_completeness),
        ("mistral_red_team", _run_mistral_redteam),
    ]

    results = {}

    if pipeline_cfg.get("parallel_review_calls", True):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(fn, corrected_draft, handoff, pub_config, api_keys, pipeline_cfg): name
                for name, fn in runners
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    log.error(f"Review pass {name} raised exception: {e}")
                    results[name] = {"failed": True, "error": str(e), "pass": name}
    else:
        for name, fn in runners:
            try:
                results[name] = fn(corrected_draft, handoff, pub_config, api_keys, pipeline_cfg)
            except Exception as e:
                log.error(f"Review pass {name} raised exception: {e}")
                results[name] = {"failed": True, "error": str(e), "pass": name}

    # Build API call log
    for name, result in results.items():
        api_call_log.append({
            "pass": name,
            "model": result.get("model", "unknown"),
            "failed": result.get("failed", False),
            "tokens": result.get("tokens", {}),
            "error": result.get("error") if result.get("failed") else None,
        })

    # Check if all provider calls failed
    all_failed = all(r.get("failed") for r in results.values())
    if all_failed and pipeline_cfg.get("abort_if_all_provider_calls_fail", False):
        log.error("All review model calls failed. Aborting.")
        sys.exit(1)

    # Load prior report for delta
    prior_report = hist.load_prior_report(HISTORY_ROOT, article_title, run_number)

    # Consolidation
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
        api_call_log=api_call_log,
        prior_report=prior_report,
    )

    # Save to history
    paths = hist.save_run(
        HISTORY_ROOT, article_title, run_number,
        report, lt_result.get("change_log", [])
    )
    log.info(f"Report saved: {paths['report_path']}")

    # Print summary
    _print_draft_summary(report, delta_cfg)

    return report


def _print_draft_summary(report, delta_cfg):
    print("\n" + "=" * 60)
    print(f"REVIEW COMPLETE: {report['article_title']}")
    print(f"Run #{report['run_number']} — {report['generated']}")
    print("=" * 60)

    if report.get("model_failures"):
        print(f"\nWARNING: These model passes failed: {', '.join(report['model_failures'])}")

    if report.get("lt_failed"):
        print("\nWARNING: LanguageTool failed — draft not grammar-corrected")
    else:
        print(f"\nLanguageTool: {len(report['lt_corrections_applied'])} corrections applied")

    print(f"\nSection 1 — Consensus flags: {len(report['section_1_consensus'])}")
    print(f"Section 2 — Fact check items: {sum(len(v) for v in report['section_2_fact_check'].values() if isinstance(v, list))}")
    print(f"Section 3 — Voice flags: {len(report['section_3_voice'])}")
    print(f"Section 4 — Argument flags: {len(report['section_4_argument'])}")
    print(f"Section 5 — Completeness flags: {len(report['section_5_completeness'])}")
    print(f"Section 6 — Red team: {'3 findings' if report['section_6_red_team'] else 'none'}")
    print(f"Section 7 — Low-confidence flags: {len(report['section_7_low_confidence'])}")

    delta = report.get("delta")
    if delta:
        print(f"\nDelta from prior run:")
        print(f"  Word change: {delta['word_change_pct']}%")
        print(f"  Prior consensus flags: {delta['prior_consensus_count']}")
        print(f"  Resolved: {delta['resolved_consensus_count']}")
        print(f"  New: {delta['new_consensus_count']}")
        if consolidation.rerun_recommended(delta, delta_cfg):
            print("\n  RECOMMENDATION: Re-run after editing (significant changes detected)")
        else:
            print("\n  RECOMMENDATION: Draft appears stable — proceed to Grammarly pass")

    print(f"\nFull report: pipeline_history/{report['article_title']}/")
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
    handoff_text = Path(handoff_path).read_text(encoding="utf-8")
    pub_handoff = parse_publication_handoff(handoff_text)

    if not pub_handoff["final_draft"]:
        log.error("No FINAL DRAFT section found in publication handoff.")
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

    log.info(f"Pushing to WordPress (live={publish_live})")
    result = wp.push(content, pub_params, wp_config, rank_math_config, publish_live=publish_live)

    if result["success"]:
        print(f"\nWordPress push successful!")
        print(f"Post URL: {result['post_url']}")
        print(f"Post ID: {result['post_id']}")
        if not publish_live:
            print("Status: DRAFT (pass --publish-live to publish)")
        else:
            print("Status: PUBLISHED")
    else:
        print(f"\nWordPress push FAILED: {result['error']}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Article Review Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draft", metavar="HANDOFF_PATH", help="Path to draft submission handoff document")
    group.add_argument("--publish", metavar="HANDOFF_PATH", help="Path to publication handoff document")
    parser.add_argument("--publication", required=True, help="Publication config name (without .yaml)")
    parser.add_argument("--publish-live", action="store_true", help="Publish to live (default: save as draft)")
    parser.add_argument("--config-dir", default="configs", help="Directory containing config files")

    args = parser.parse_args()

    if args.draft:
        run_draft_pipeline(args.draft, args.publication, config_dir=args.config_dir)
    elif args.publish:
        run_publish_pipeline(args.publish, args.publication, publish_live=args.publish_live, config_dir=args.config_dir)


if __name__ == "__main__":
    main()
