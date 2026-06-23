"""Microsoft Graph API / Outlook 365 collector."""

from __future__ import annotations

import logging
import os
import re
import stat
import time
from typing import Iterator

import requests

from collectors.base import Collector, CollectorError, ConfigError, Document

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPES = ["https://graph.microsoft.com/Mail.Read"]

_QUOTE_PATTERNS = [
    re.compile(r"^>.*", re.MULTILINE),
    re.compile(r"^On .+wrote:.*$", re.MULTILINE | re.DOTALL),
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
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode & 0o077:
            log.warning("Credentials file %s has loose permissions (%o). Recommend: chmod 600 %s", path, mode, path)
    except FileNotFoundError:
        pass


class Outlook365Collector(Collector):
    SOURCE_NAME = "outlook365"

    REQUIRED_KEYS = ["tenant_id", "client_id", "auth_method"]

    @classmethod
    def validate_config(cls, config: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ConfigError(cls.SOURCE_NAME, missing_keys=missing)
        if config.get("auth_method") == "client_credentials" and not config.get("client_secret"):
            raise ConfigError(cls.SOURCE_NAME, message="client_secret required for client_credentials auth_method")

    def _get_token(self) -> str:
        try:
            import msal
        except ImportError as e:
            raise CollectorError(self.SOURCE_NAME, f"Missing dependency: {e}. Install msal.")

        tenant_id = self.config["tenant_id"]
        client_id = self.config["client_id"]
        auth_method = self.config.get("auth_method", "device_code")
        creds_file = os.path.expanduser(self.config.get("credentials_file", "~/.config/voice-bootstrap/outlook_token.json"))
        _check_file_permissions(creds_file)

        authority = f"https://login.microsoftonline.com/{tenant_id}"
        token_cache = msal.SerializableTokenCache()

        if os.path.exists(creds_file):
            with open(creds_file) as f:
                token_cache.deserialize(f.read())

        if auth_method == "client_credentials":
            app = msal.ConfidentialClientApplication(
                client_id=client_id,
                client_credential=self.config["client_secret"],
                authority=authority,
                token_cache=token_cache,
            )
            result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        else:
            app = msal.PublicClientApplication(client_id=client_id, authority=authority, token_cache=token_cache)
            accounts = app.get_accounts()
            result = None
            if accounts:
                try:
                    result = app.acquire_token_silent(SCOPES, account=accounts[0])
                except Exception as e:
                    if "InteractionRequired" in type(e).__name__:
                        log.info("Outlook365 session expired — re-authentication required. Run again to get a new device code.")
                    result = None

            if not result:
                flow = app.initiate_device_flow(scopes=SCOPES)
                if "user_code" not in flow:
                    raise CollectorError(self.SOURCE_NAME, "Failed to initiate device flow")
                print(flow["message"])
                result = app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            raise CollectorError(self.SOURCE_NAME, f"Auth failed: {result.get('error_description', result)}")

        if token_cache.has_state_changed:
            os.makedirs(os.path.dirname(creds_file) or ".", exist_ok=True)
            with open(creds_file, "w") as f:
                f.write(token_cache.serialize())
            os.chmod(creds_file, 0o600)

        return result["access_token"]

    def estimate_count(self) -> int | None:
        try:
            token = self._get_token()
        except Exception:
            return None
        folder = self.config.get("folder", "SentItems")
        url = f"{GRAPH_BASE}/me/mailFolders/{folder}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.body-content-type="text"',
        }
        try:
            resp = requests.get(url, headers=headers, params={"$count": "true", "$top": 1}, timeout=(10, 30))
            resp.raise_for_status()
            return resp.json().get("@odata.count")
        except Exception:
            return None

    def fetch(self, since: str | None = None) -> Iterator[Document]:
        try:
            token = self._get_token()
        except CollectorError:
            raise
        except Exception as e:
            raise CollectorError(self.SOURCE_NAME, f"Auth failed: {e}")

        folder = self.config.get("folder", "SentItems")
        max_messages = int(self.config.get("max_messages", 500))
        odata_filter = self.config.get("query", "")
        if since and not odata_filter:
            odata_filter = f"sentDateTime ge {since}T00:00:00Z"

        headers = {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.body-content-type="text"',
        }
        url = f"{GRAPH_BASE}/me/mailFolders/{folder}/messages"
        params: dict = {
            "$select": "body,subject,sentDateTime,toRecipients,importance",
            "$top": 50,
        }
        if odata_filter:
            params["$filter"] = odata_filter

        fetched = 0
        next_url: str | None = url

        while next_url and fetched < max_messages:
            for attempt in range(3):
                try:
                    if next_url == url:
                        resp = requests.get(next_url, headers=headers, params=params, timeout=(10, 60))
                    else:
                        resp = requests.get(next_url, headers=headers, timeout=(10, 60))
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                        log.warning("Outlook365 rate limited; waiting %ds", wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except requests.HTTPError as e:
                    if attempt == 2:
                        raise CollectorError(self.SOURCE_NAME, f"HTTP error: {e}")
                    time.sleep(2 ** attempt)
            else:
                break

            data = resp.json()
            messages = data.get("value", [])
            for msg in messages:
                if fetched >= max_messages:
                    break
                body_obj = msg.get("body", {})
                raw_text = body_obj.get("content", "")
                text = _strip_quoted_blocks(raw_text)
                if not text.strip():
                    continue
                sent = msg.get("sentDateTime", "")[:10]
                recipients = msg.get("toRecipients", [])
                doc = Document.from_text(
                    text=text,
                    source=self.SOURCE_NAME,
                    register="correspondence",
                    date=sent,
                    url_or_id=msg.get("id", ""),
                    metadata={
                        "message_id": msg.get("id"),
                        "folder": folder,
                        "recipient_count": len(recipients),
                        "importance": msg.get("importance", "normal"),
                    },
                )
                yield doc
                fetched += 1

            next_url = data.get("@odata.nextLink")
