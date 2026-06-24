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


def _lookup_term_ids(api_base, headers, taxonomy, items):
    """Convert category/tag slugs (strings) or IDs (ints) to WP term IDs.

    WordPress REST API requires integer IDs when creating/updating posts.
    String slugs are resolved via a GET request to the taxonomy endpoint.
    Integer IDs are passed through unchanged.

    Args:
        api_base:  Base URL for WP REST API, e.g. ``https://site.com/wp-json/wp/v2``
        headers:   Auth headers dict (already built by caller)
        taxonomy:  ``"categories"`` or ``"tags"``
        items:     List of slugs (str) or IDs (int), or a single slug string

    Returns:
        List of integer term IDs.  Slugs that cannot be resolved are logged
        as warnings and omitted.
    """
    if not items:
        return []

    # Accept a single string or a list
    if isinstance(items, (str, int)):
        items = [items]

    resolved = []
    for item in items:
        if isinstance(item, int):
            resolved.append(item)
            continue
        if not isinstance(item, str) or not item.strip():
            continue
        slug = item.strip()
        try:
            resp = requests.get(
                f"{api_base}/{taxonomy}",
                params={"slug": slug},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            terms = resp.json()
            if terms:
                resolved.append(terms[0]["id"])
                log.debug(f"Resolved {taxonomy} slug '{slug}' → ID {terms[0]['id']}")
            else:
                log.warning(
                    f"WordPress {taxonomy} slug '{slug}' not found — "
                    f"create it in WP admin first, or use its integer ID."
                )
        except Exception as e:
            log.warning(f"WordPress {taxonomy} lookup failed for '{slug}': {e}")

    return resolved


def _build_post_payload(
    pub_params, wp_config, rank_math_config, content, category_ids, tag_ids
):
    payload = {
        "title": pub_params.get("title", ""),
        "content": content,
        "status": "draft",  # always draft unless --publish-live
        "categories": category_ids,
        "tags": tag_ids,
    }

    author = pub_params.get("author")
    if author:
        payload["author"] = author

    # Rank Math SEO meta fields
    meta = {}
    seo = pub_params.get("seo", {})
    focus_keyword = seo.get("focus_keyword")
    meta_description = seo.get("meta_description")
    og_title = seo.get("og_title") or pub_params.get("title", "")
    og_description = seo.get("og_description") or meta_description
    schema_type = seo.get("schema_type") or rank_math_config.get(
        "default_schema_type", "BlogPosting"
    )

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


def push(content, pub_params, wp_config, rank_math_config, publish_live=False):
    """Push article to WordPress.  Always saves as draft unless publish_live=True.

    Resolves category and tag slugs to integer IDs via the WP REST API before
    creating the post.

    Returns dict with keys: success (bool), post_id, post_url, error (if failed).
    """
    site_url = wp_config["site_url"].rstrip("/")
    endpoint = wp_config.get("rest_api_endpoint", "/wp-json/wp/v2")
    api_base = f"{site_url}{endpoint}"
    username = wp_config["username"]
    app_password = wp_config["application_password"]

    headers = _auth_header(username, app_password)
    headers["Content-Type"] = "application/json"
    auth_headers = _auth_header(username, app_password)  # without Content-Type for GETs

    # Resolve slugs → IDs before building the post payload
    category_ids = _lookup_term_ids(
        api_base,
        auth_headers,
        "categories",
        pub_params.get("wordpress_category"),
    )
    tag_ids = _lookup_term_ids(
        api_base,
        auth_headers,
        "tags",
        pub_params.get("tags", []),
    )

    payload = _build_post_payload(
        pub_params,
        wp_config,
        rank_math_config,
        content,
        category_ids=category_ids,
        tag_ids=tag_ids,
    )

    if publish_live:
        payload["status"] = "publish"

    try:
        resp = requests.post(
            f"{api_base}/posts", headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        post_id = data.get("id")
        post_url = data.get("link")
        log.info(
            f"WordPress push successful: post_id={post_id} url={post_url} "
            f"categories={category_ids} tags={tag_ids}"
        )
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
    answer = (
        input("Have you reviewed the checklist and confirmed all items? [yes/no]: ")
        .strip()
        .lower()
    )
    if answer not in ("yes", "y"):
        print("Publication aborted. Complete the checklist and re-run.")
        return False
    return True
