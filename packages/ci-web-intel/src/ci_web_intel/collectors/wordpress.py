"""WordPress REST API collector."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from typing import Iterator

import requests

from collectors.base import Collector, CollectorError, ConfigError, Document

log = logging.getLogger(__name__)

SOURCE_NAME = "wordpress"


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return stripper.get_text()


def _strip_shortcodes(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text)


def _is_public_host(url: str) -> bool:
    try:
        from analysis.links import _is_public_host as _pipeline_check
        return _pipeline_check(url)
    except ImportError:
        pass
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or ""
    private = ("localhost", "127.", "192.168.", "10.", "172.")
    return bool(host) and not any(host == p or host.startswith(p) for p in private)


class WordPressCollector(Collector):
    SOURCE_NAME = "wordpress"

    REQUIRED_KEYS = ["site_url", "username", "application_password"]

    @classmethod
    def validate_config(cls, config: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ConfigError(cls.SOURCE_NAME, missing_keys=missing)
        site_url = config.get("site_url", "")
        if not _is_public_host(site_url):
            raise ConfigError(cls.SOURCE_NAME, message=f"site_url {site_url!r} is not a public host")

    def estimate_count(self) -> int | None:
        site_url = self.config["site_url"].rstrip("/")
        try:
            resp = requests.get(
                f"{site_url}/wp-json/wp/v2/posts",
                params={"per_page": 1, "status": "publish"},
                auth=(self.config["username"], self.config["application_password"]),
                timeout=(10, 30),
                allow_redirects=False,
            )
            if resp.status_code == 200:
                return int(resp.headers.get("X-WP-Total", 0))
        except Exception:
            pass
        return None

    def fetch(self, since: str | None = None) -> Iterator[Document]:
        site_url = self.config["site_url"].rstrip("/")
        auth = (self.config["username"], self.config["application_password"])
        base_url = f"{site_url}/wp-json/wp/v2/posts"

        params: dict = {"per_page": 100, "status": "publish"}
        if since:
            params["after"] = since

        try:
            resp = requests.get(
                base_url, params=params, auth=auth,
                timeout=(10, 60), allow_redirects=False,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise CollectorError(self.SOURCE_NAME, f"HTTP {e.response.status_code}: {e}")
        except Exception as e:
            raise CollectorError(self.SOURCE_NAME, f"Request failed: {e}")

        total = int(resp.headers.get("X-WP-Total", 0))
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        log.info("WordPress: %d posts across %d pages", total, total_pages, extra={"source": self.SOURCE_NAME})

        page1_posts = resp.json()
        yield from self._posts_to_docs(page1_posts)
        fetched = len(page1_posts)
        log.info("WordPress: fetched page 1/%d (%d posts)", total_pages, fetched)

        if total_pages > 1:
            pages = list(range(2, total_pages + 1))

            def _fetch_page(page_num):
                p = dict(params)
                p["page"] = page_num
                r = requests.get(
                    base_url, params=p, auth=auth,
                    timeout=(10, 60), allow_redirects=False,
                )
                r.raise_for_status()
                return page_num, r.json()

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(_fetch_page, p): p for p in pages}
                for fut in as_completed(futures):
                    try:
                        page_num, posts = fut.result()
                        log.info(
                            "WordPress: fetched page %d/%d (%d posts so far)",
                            page_num, total_pages, fetched + len(posts),
                            extra={"page": page_num, "total": total_pages, "count": fetched},
                        )
                        fetched += len(posts)
                        yield from self._posts_to_docs(posts)
                    except Exception as e:
                        log.warning("WordPress: page %d failed: %s", futures[fut], e)

    def _posts_to_docs(self, posts: list) -> list[Document]:
        docs = []
        for post in posts:
            raw_html = post.get("content", {}).get("rendered", "")
            text = _strip_shortcodes(_strip_html(raw_html)).strip()
            if not text:
                continue
            date = post.get("date", "")[:10]
            url = post.get("link", str(post.get("id", "")))
            metadata = {
                "post_id": post.get("id"),
                "categories": [str(c) for c in post.get("categories", [])],
                "tags": [str(t) for t in post.get("tags", [])],
            }
            docs.append(Document.from_text(
                text=text,
                source=self.SOURCE_NAME,
                register="long_form",
                date=date,
                url_or_id=url,
                metadata=metadata,
            ))
        return docs
