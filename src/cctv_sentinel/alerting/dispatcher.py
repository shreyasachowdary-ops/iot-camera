"""Decide whether an event becomes an alert, and build the payload.

The detector answers "did something happen?". This answers "should a human be
interrupted?" -- which is a different question, and the one that determines whether
the system is still switched on in three months.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import AlertConfig
from ..detect.smoother import Event
from .webhook import WebhookSender

log = logging.getLogger(__name__)

# Severity drives the escalation ladder on the backend. A collapse in an empty
# corridor needs someone now; loitering can wait for the morning.
SEVERITY = {
    "fall": "critical",
    "fall_collapse": "critical",
    "weapon": "critical",
    "weapon_brandish": "critical",
    "fight": "high",
    "crowd_run": "high",
    "loitering": "low",
    "loitering_intrusion": "low",
}


class AlertDispatcher:
    def __init__(self, config: AlertConfig, sender: WebhookSender | None = None) -> None:
        self.config = config
        self.sender = sender or WebhookSender(config)
        self._last_sent: dict[tuple[str, str], float] = {}

    def _suppressed(self, event: Event, now: float) -> bool:
        key = (event.camera_id, event.label)
        last = self._last_sent.get(key)
        return last is not None and (now - last) < self.config.cooldown_s

    def build_payload(
        self, event: Event, clip_path: str | None = None, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = {
            "schema": "cctv-sentinel/alert@1",
            "camera_id": event.camera_id,
            "label": event.label,
            "severity": SEVERITY.get(event.label, "medium"),
            "confidence": round(event.peak_score, 4),
            "mean_confidence": round(event.mean_score, 4),
            "started_at": event.started_at,
            "ended_at": event.ended_at,
            "duration_s": round(event.duration, 2),
            "detected_at": time.time(),
            "clip_path": clip_path,
            # Explicit, because the receiving system must not treat this as ground
            # truth. A human confirms before anything escalates. See docs/01.
            "requires_human_confirmation": True,
        }
        if extra:
            payload.update(extra)
        return payload

    def dispatch(
        self, event: Event, clip_path: str | None = None, extra: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Returns the payload if it was sent, None if suppressed."""
        now = time.time()
        if self._suppressed(event, now):
            log.info("suppressed %s on %s (cooldown)", event.label, event.camera_id)
            return None

        payload = self.build_payload(event, clip_path=clip_path, extra=extra)
        self._last_sent[(event.camera_id, event.label)] = now
        self.sender.send(payload)
        return payload
