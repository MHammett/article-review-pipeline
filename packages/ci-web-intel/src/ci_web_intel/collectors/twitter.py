"""Twitter/X API v2 collector."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import requests

from .base import Collector, CollectorError, ConfigError, Document

log = logging.getLogger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"


class TwitterCollector(Collector):
    SOURCE_NAME = "twitter"

    REQUIRED_KEYS = ["bearer_token"]

    @classmethod
    def validate_config(cls, config: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ConfigError(cls.SOURCE_NAME, missing_keys=missing)
        if not config.get("user_id") and not config.get("username"):
            raise ConfigError(cls.SOURCE_NAME, message="Either 'user_id' or 'username' is required")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.config['bearer_token']}"}

    def _resolve_user_id(self) -> str:
        if user_id := self.config.get("user_id"):
            return str(user_id)
        username = self.config["username"]
        resp = requests.get(
            f"{TWITTER_API_BASE}/users/by/username/{username}",
            headers=self._headers(),
            timeout=(10, 30),
        )
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 2)))
            resp = requests.get(
                f"{TWITTER_API_BASE}/users/by/username/{username}",
                headers=self._headers(),
                timeout=(10, 30),
            )
        resp.raise_for_status()
        return resp.json()["data"]["id"]

    def fetch(self, since: str | None = None) -> Iterator[Document]:
        try:
            user_id = self._resolve_user_id()
        except Exception as e:
            raise CollectorError(self.SOURCE_NAME, f"Could not resolve user_id: {e}")

        exclude = []
        if self.config.get("exclude_retweets", True):
            exclude.append("retweets")
        if self.config.get("exclude_replies", True):
            exclude.append("replies")

        max_results = min(int(self.config.get("max_results_per_page", 100)), 100)
        params: dict = {
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,conversation_id",
        }
        if exclude:
            params["exclude"] = ",".join(exclude)
        if since:
            params["start_time"] = since if "T" in since else f"{since}T00:00:00Z"

        url = f"{TWITTER_API_BASE}/users/{user_id}/tweets"
        pagination_token = None
        retries = 0

        while True:
            if pagination_token:
                params["pagination_token"] = pagination_token

            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=(10, 30))
            except Exception as e:
                raise CollectorError(self.SOURCE_NAME, f"Request failed: {e}")

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** (retries + 1)))
                log.warning("Twitter rate limit; waiting %ds", wait)
                time.sleep(wait)
                retries += 1
                if retries > 3:
                    raise CollectorError(self.SOURCE_NAME, "Rate limit exhausted after 3 retries")
                continue
            elif resp.status_code >= 500:
                retries += 1
                if retries > 3:
                    raise CollectorError(self.SOURCE_NAME, f"Server error {resp.status_code} after 3 retries")
                time.sleep(2 ** retries)
                continue

            try:
                resp.raise_for_status()
            except Exception as e:
                raise CollectorError(self.SOURCE_NAME, f"HTTP error: {e}")

            retries = 0
            data = resp.json()
            tweets = data.get("data") or []
            for tweet in tweets:
                text = tweet.get("text", "").strip()
                if not text:
                    continue
                metrics = tweet.get("public_metrics", {})
                date = tweet.get("created_at", "")[:10]
                # Normalize t.co URLs
                import re
                text = re.sub(r"https://t\.co/\S+", "[link]", text)
                # Strip leading @mentions on replies
                text = re.sub(r"^(@\w+\s+)+", "", text).strip()
                doc = Document.from_text(
                    text=text,
                    source=self.SOURCE_NAME,
                    register="casual",
                    date=date,
                    url_or_id=tweet.get("id", ""),
                    metadata={
                        "tweet_id": tweet.get("id"),
                        "public_metrics": metrics,
                        "conversation_id": tweet.get("conversation_id"),
                    },
                )
                yield doc

            meta = data.get("meta", {})
            pagination_token = meta.get("next_token")
            if not pagination_token:
                break
