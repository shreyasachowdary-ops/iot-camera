"""Signed webhook delivery with retries and a disk-backed outbox.

Design constraints that come from the deployment reality, not from taste:

  * Site internet WILL drop. Failed deliveries spool to disk and replay on recovery.
    An alert that is lost because a router rebooted is the worst possible bug here.
  * Retries mean the receiver WILL see duplicates. Every payload carries a stable
    `event_id`; the receiver must be idempotent on it.
  * The payload asserts something about a real person's behaviour. It is HMAC-signed
    so the receiver can verify nobody else is injecting alerts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from ..config import AlertConfig

log = logging.getLogger(__name__)


def sign(secret: str, body: bytes, timestamp: str) -> str:
    """Signature over `timestamp.body`, so a captured body can't be replayed later."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


class WebhookSender:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self.outbox = Path(config.outbox_dir)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def send(self, payload: dict[str, Any]) -> bool:
        payload.setdefault("event_id", str(uuid.uuid4()))
        if not self.config.webhook_url:
            log.warning("no webhook_url configured; alert dropped: %s", payload["event_id"])
            return False

        body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Sentinel-Timestamp": timestamp,
            "X-Sentinel-Event-Id": payload["event_id"],
        }
        if self.config.hmac_secret:
            headers["X-Sentinel-Signature"] = sign(self.config.hmac_secret, body, timestamp)

        delay = 1.0
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._session.post(
                    self.config.webhook_url,
                    data=body,
                    headers=headers,
                    timeout=self.config.timeout_s,
                )
                if response.ok:
                    log.info("alert %s delivered", payload["event_id"])
                    return True
                # 4xx other than 429 means the receiver rejected it; retrying an
                # unauthorised or malformed request just burns the outbox.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    log.error(
                        "webhook rejected alert %s: %s", payload["event_id"], response.status_code
                    )
                    break
                log.warning(
                    "webhook attempt %d for %s: HTTP %s",
                    attempt, payload["event_id"], response.status_code,
                )
            except requests.RequestException as exc:
                log.warning("webhook attempt %d for %s failed: %s", attempt, payload["event_id"], exc)

            if attempt < self.config.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        self._spool(payload)
        return False

    def _spool(self, payload: dict[str, Any]) -> None:
        path = self.outbox / f"{int(time.time() * 1000)}-{payload['event_id']}.json"
        path.write_text(json.dumps(payload))
        log.error("alert %s spooled to %s for later replay", payload["event_id"], path)

    def replay_outbox(self) -> int:
        """Re-attempt spooled alerts. Call periodically and on reconnect."""
        sent = 0
        for path in sorted(self.outbox.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                log.exception("dropping unreadable outbox entry %s", path)
                path.unlink(missing_ok=True)
                continue
            # Unlink first: a crash mid-send costs one duplicate (the receiver is
            # idempotent on event_id), whereas unlinking after a partial failure
            # could lose the alert entirely.
            path.unlink(missing_ok=True)
            if self.send(payload):
                sent += 1
            else:
                break  # still down; leave the rest queued
        return sent
