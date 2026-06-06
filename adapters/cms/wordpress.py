import base64
import logging
import requests

log = logging.getLogger(__name__)

CHECKLIST = """
PRE-PUBLICATION CHECKLIST
==========================
CONTENT
[ ] All consensus flags addressed or explicitly dismissed with reasoning
[ ] All factual claims have primary source citations
[ ] All citations have SHA-256 checksums recorded in pipeline_history/
[ ] Pre-draft analysis counterargument dispositions reflected in final draft

TECHNICAL
[ ] All embedded components tested (charts, maps, interactive elements)
[ ] Data in visualizations matches claims in prose
[ ] Structured data schema validates (https://validator.schema.org)
[ ] All images have alt text
[ ] Internal links reviewed and functional
[ ] Canonical URL set correctly in WordPress

SEO
[ ] Focus keyword set in Rank Math
[ ] Meta description under 155 characters
[ ] OG tags set
[ ] Schema type correct for content type

PUBLICATION
[ ] WordPress category correct
[ ] Tags applied
[ ] Status is draft (default) -- confirm before switching to live
[ ] UpdraftPlus backup is current
"""


def _auth_header(username, application_password):
    token = base64.b64encode(f"{username}:{application_password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _build_post_payload(pub_params, wp_config, rank_math_config, content):
    payload = {
        "title": pub_params.get("title", ""),
        "content": content,
        "status": "draft",  # always draft unless --publish-live
        "categories": _resolve_categories(pub_params.get("wordpress_category")),
        "tags": _resolve_tags(pub_params.get("tags", [])),
    }

    author = pub_params.get("author")
    if author:
        payload["author"] = author

    # Rank Math SEO meta fields via meta
    meta = {}
    focus_keyword = pub_params.get("seo", {}).get("focus_keyword")
    meta_description = pub_params.get("seo", {}).get("meta_description")
    og_title = pub_params.get("seo", {}).get("og_title") or pub_params.get("title", "")
    og_description = pub_params.get("seo", {}).get("og_description") or meta_description
    schema_type = pub_params.get("seo", {}).get("schema_type") or rank_math_config.get("default_schema_type", "BlogPosting")

    if focus_keyword:
        meta["rank_math_focus_keyword"] = focus_keyword
    if meta_description:
        meta["rank_math_description"] = meta_description
    if og_title and rank_math_config.get("auto_set_og_tags"):
        meta["rank_math_og_title"] = og_title
    if og_description and rank_math_config.get("auto_set_og_tags"):
        meta["rank_math_og_description"] = og_description
    meta["rank_math_schema_type"] = schema_type

    if meta:
        payload["meta"] = meta

    return payload


def _resolve_categories(category_slug):
    if not category_slug:
        return []
    # WordPress REST API expects category IDs; slugs require a lookup
    # Return as-is — pipeline.py handles slug-to-ID resolution if needed
    return [category_slug] if isinstance(category_slug, int) else []


def _resolve_tags(tag_slugs):
    if not tag_slugs:
        return []
    return [t for t in tag_slugs if isinstance(t, int)]


def push(content, pub_params, wp_config, rank_math_config, publish_live=False):
    """
    Push article to WordPress. Always saves as draft unless publish_live=True.

    Returns dict with keys: success (bool), post_id, post_url, error (if failed).
    """
    site_url = wp_config["site_url"].rstrip("/")
    endpoint = wp_config.get("rest_api_endpoint", "/wp-json/wp/v2")
    api_base = f"{site_url}{endpoint}"
    username = wp_config["username"]
    app_password = wp_config["application_password"]

    headers = _auth_header(username, app_password)
    headers["Content-Type"] = "application/json"

    payload = _build_post_payload(pub_params, wp_config, rank_math_config, content)

    if publish_live:
        payload["status"] = "publish"

    try:
        resp = requests.post(f"{api_base}/posts", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        post_id = data.get("id")
        post_url = data.get("link")
        log.info(f"WordPress push successful: post_id={post_id} url={post_url}")
        return {"success": True, "post_id": post_id, "post_url": post_url}
    except requests.HTTPError as e:
        error_body = ""
        try:
            error_body = e.response.json()
        except Exception:
            error_body = str(e)
        log.error(f"WordPress push failed: {error_body}")
        return {"success": False, "error": str(error_body)}
    except Exception as e:
        log.error(f"WordPress push failed: {e}")
        return {"success": False, "error": str(e)}


def print_checklist_and_confirm():
    print(CHECKLIST)
    answer = input("Have you reviewed the checklist and confirmed all items? [yes/no]: ").strip().lower()
    if answer not in ("yes", "y"):
        print("Publication aborted. Complete the checklist and re-run.")
        return False
    return True
