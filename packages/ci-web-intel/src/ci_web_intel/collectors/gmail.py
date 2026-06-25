"""Gmail collector via Google API."""

from __future__ import annotations

import base64
import logging
import os
import re
import stat
import time
from typing import Iterator

from .base import Collector, CollectorError, ConfigError, Document

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_QUOTE_PATTERNS = [
    re.compile(r"^>.*", re.MULTILINE),
    re.compile(r"^On .+wrote:.*$", re.MULTILINE | re.DOTALL),
    re.compile(r"^-{3,}.*Original Message.*-{3,}$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^From:.*$", re.MULTILINE),
]
_SIG_SEP = re.compile(r"^--\s*$", re.MULTILINE)


def _strip_quoted_blocks(text: str) -> str:
    text = _SIG_SEP.split(text)[0]
    for pat in _QUOTE_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _check_file_permissions(path: str) -> None:
    if os.name == "nt":
        log.debug("Skipping permission check on Windows for %s", path)
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode & 0o077:
            log.warning(
                "Credentials file %s has loose permissions (mode %o). "
                "Recommend: chmod 600 %s", path, mode, path
            )
    except FileNotFoundError:
        pass


def _get_text_body(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", "replace")
    for part in payload.get("parts", []):
        text = _get_text_body(part)
        if text:
            return text
    return ""


class GmailCollector(Collector):
    SOURCE_NAME = "gmail"

    REQUIRED_KEYS = ["credentials_file"]

    @classmethod
    def validate_config(cls, config: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ConfigError(cls.SOURCE_NAME, missing_keys=missing)

    def _get_service(self):
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request as GoogleRequest
            from googleapiclient.discovery import build
        except ImportError as e:
            raise CollectorError(self.SOURCE_NAME, f"Missing dependency: {e}. Install google-auth-oauthlib and google-api-python-client.")

        creds_file = os.path.expanduser(self.config["credentials_file"])
        _check_file_permissions(creds_file)

        token_file = creds_file.replace(".json", "_token.json")
        creds = None
        if os.path.exists(token_file):
            _check_file_permissions(token_file)
            try:
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(GoogleRequest())
                except Exception:
                    creds = None

            if not creds:
                flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_file, "w") as f:
                    f.write(creds.to_json())
                os.chmod(token_file, 0o600)

        return build("gmail", "v1", credentials=creds)

    def fetch(self, since: str | None = None) -> Iterator[Document]:
        try:
            service = self._get_service()
        except CollectorError:
            raise
        except Exception as e:
            raise CollectorError(self.SOURCE_NAME, f"Auth failed: {e}")

        query = self.config.get("query", "from:me")
        max_messages = int(self.config.get("max_messages", 500))

        message_ids = []
        page_token = None
        while len(message_ids) < max_messages:
            params: dict = {"userId": "me", "q": query, "maxResults": min(100, max_messages - len(message_ids))}
            if page_token:
                params["pageToken"] = page_token
            try:
                result = service.users().messages().list(**params).execute()
            except Exception as e:
                raise CollectorError(self.SOURCE_NAME, f"List messages failed: {e}")

            msgs = result.get("messages", [])
            message_ids.extend(m["id"] for m in msgs)
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        if len(message_ids) > max_messages:
            log.warning("Gmail: %d messages found, capping at %d (most recent first)", len(message_ids), max_messages)
            message_ids = message_ids[:max_messages]

        for msg_id in message_ids:
            for attempt in range(3):
                try:
                    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
                    break
                except Exception as e:
                    if attempt == 2:
                        log.warning("Gmail: failed to fetch message %s: %s", msg_id, e)
                        msg = None
                    time.sleep(2 ** attempt)

            if not msg:
                continue

            payload = msg.get("payload", {})
            text = _get_text_body(payload)
            text = _strip_quoted_blocks(text)
            if not text.strip():
                continue

            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
            date_str = headers.get("Date", "")[:10]
            labels = msg.get("labelIds", [])
            thread_id = msg.get("threadId", "")
            to_header = headers.get("To", "")
            recipient_count = len(to_header.split(",")) if to_header else 0

            doc = Document.from_text(
                text=text,
                source=self.SOURCE_NAME,
                register="correspondence",
                date=date_str,
                url_or_id=msg_id,
                metadata={
                    "message_id": msg_id,
                    "thread_id": thread_id,
                    "recipient_count": recipient_count,
                    "labels": labels,
                },
            )
            yield doc
