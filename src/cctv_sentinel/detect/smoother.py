"""Turn a noisy per-window score stream into discrete events.

This module is where false alarms go to die, and it matters more than the model.

Arithmetic worth internalising: one window per second per camera is 86,400 windows
per camera per day. A model with a 0.1% false-positive rate produces 86 false alerts a
day from ONE camera. Guards mute the system in week two and the project is over.

Three mechanisms stack here:

    EMA          smooths single-window spikes
    K-of-N       an event needs sustained evidence, not one lucky frame
    hysteresis   open_at > close_at, so a score hovering at the boundary
                 doesn't flap the event open and closed
    min duration sub-second events are discarded outright
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..config import ClassThreshold, DetectConfig


@dataclass
class Event:
    camera_id: str
    label: str
    started_at: float
    ended_at: float | None = None
    peak_score: float = 0.0
    scores: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


class _ClassTracker:
    def __init__(self, camera_id: str, label: str, thresholds: ClassThreshold) -> None:
        self.camera_id = camera_id
        self.label = label
        self.t = thresholds
        self._ema: float | None = None
        self._recent: deque[bool] = deque(maxlen=thresholds.open_n)
        self._last_update: float | None = None
        self.event: Event | None = None

    def _close(self, ts: float) -> Event | None:
        event = self.event
        if event is None:
            return None
        event.ended_at = ts
        self.event = None
        self._recent.clear()
        self._ema = None
        if event.duration < self.t.min_event_s:
            return None  # too short to be real
        return event

    def tick(self, ts: float) -> Event | None:
        """Close an open event that has gone stale, with no new score.

        Called every loop iteration, independently of whether a clip was scored --
        which is the point: the case this exists for is exactly the case where clips
        have stopped arriving.
        """
        if self.event is None:
            return None
        if self._last_update is not None and (ts - self._last_update) >= self.t.idle_close_s:
            return self._close(ts)
        if (ts - self.event.started_at) >= self.t.max_event_s:
            return self._close(ts)
        return None

    def update(self, score: float, ts: float) -> Event | None:
        """Feed one window score. Returns the Event when it *closes*, else None.

        Events are emitted on close rather than on open so the alert carries the full
        duration and peak score. The pre-roll comes from the ring buffer, so waiting
        costs nothing in evidence -- and `min_event_s` is short enough (1.5 s default)
        that latency stays inside the 5 s budget.
        """
        self._last_update = ts
        self._ema = score if self._ema is None else (
            self.t.ema_alpha * score + (1 - self.t.ema_alpha) * self._ema
        )
        smoothed = self._ema
        self._recent.append(smoothed >= self.t.open_at)

        if self.event is None:
            if sum(self._recent) >= self.t.open_k:
                self.event = Event(
                    camera_id=self.camera_id,
                    label=self.label,
                    started_at=ts,
                    peak_score=smoothed,
                    scores=[smoothed],
                )
            return None

        self.event.scores.append(smoothed)
        self.event.peak_score = max(self.event.peak_score, smoothed)

        if smoothed < self.t.close_at:
            return self._close(ts)
        if (ts - self.event.started_at) >= self.t.max_event_s:
            return self._close(ts)

        return None


class EventDetector:
    """Runs a tracker per class and yields closed events."""

    def __init__(self, camera_id: str, classes: list[str], config: DetectConfig) -> None:
        self.trackers = {
            label: _ClassTracker(camera_id, label, config.for_class(label))
            for label in classes
            if label != "normal"
        }

    def update(self, scores: dict[str, float], ts: float | None = None) -> list[Event]:
        ts = time.time() if ts is None else ts
        closed = []
        for label, tracker in self.trackers.items():
            event = tracker.update(scores.get(label, 0.0), ts)
            if event is not None:
                closed.append(event)
        return closed

    def tick(self, ts: float | None = None) -> list[Event]:
        """Close stale events. Must be called every loop, not only when a clip scores."""
        ts = time.time() if ts is None else ts
        closed = []
        for tracker in self.trackers.values():
            event = tracker.tick(ts)
            if event is not None:
                closed.append(event)
        return closed

    @property
    def active(self) -> list[Event]:
        return [t.event for t in self.trackers.values() if t.event is not None]
