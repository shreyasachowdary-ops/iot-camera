"""The main loop: stream -> buffer -> gate -> clip -> model -> events -> alerts."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

from ..alerting.dispatcher import AlertDispatcher
from ..config import Config
from ..detect.smoother import Event, EventDetector
from ..ingest.ring_buffer import RingBuffer
from ..ingest.stream import FrameSource
from ..models import classifier as model_factory
from ..pipeline.motion_gate import MotionGate
from ..pipeline.sampler import ClipSampler
from ..storage.clip_writer import ClipWriter

log = logging.getLogger(__name__)


@dataclass
class Stats:
    frames: int = 0
    gate_passed: int = 0
    clips_scored: int = 0
    events: int = 0
    alerts: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def summary(self) -> str:
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        pass_rate = 100.0 * self.gate_passed / self.frames if self.frames else 0.0
        return (
            f"{self.frames} frames ({self.frames / elapsed:.1f} fps), "
            f"gate pass {pass_rate:.1f}%, {self.clips_scored} clips scored, "
            f"{self.events} events, {self.alerts} alerts"
        )


@dataclass
class _PendingClip:
    """An event whose clip we are still collecting post-roll for."""

    event: Event
    write_after: float


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.source = FrameSource(config.stream)
        self.buffer = RingBuffer(
            seconds=config.ring_buffer.seconds,
            fps=config.stream.target_fps,
            jpeg_quality=config.ring_buffer.jpeg_quality,
        )
        self.gate = MotionGate(config.motion_gate)
        self.sampler = ClipSampler(config.sampler)
        self.model = model_factory.build(config.model)
        self.detector = EventDetector(config.camera_id, self.model.classes, config.detect)
        self.dispatcher = AlertDispatcher(config.alert)
        self.clips = ClipWriter(config.storage)
        self.stats = Stats()
        self._pending: list[_PendingClip] = []
        self._stop = False
        self._last_outbox_replay = 0.0
        self._warn_on_config()

    def _warn_on_config(self) -> None:
        needed = self.config.storage.pre_roll_s + self.config.storage.post_roll_s
        if needed > self.config.ring_buffer.seconds:
            log.warning(
                "ring_buffer.seconds (%.0fs) < pre_roll + post_roll (%.0fs): clips will "
                "silently lose pre-roll, which is the part a human needs to judge the "
                "event. Raise ring_buffer.seconds.",
                self.config.ring_buffer.seconds, needed,
            )

    def stop(self, *_: object) -> None:
        self._stop = True

    def run(self, max_seconds: float | None = None) -> Stats:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        last_index = -1
        last_report = time.monotonic()

        with self.source:
            log.info("runner started on camera %s", self.config.camera_id)
            while not self._stop:
                if deadline is not None and time.monotonic() > deadline:
                    break

                frame = self.source.read()
                if frame is None or frame.index == last_index:
                    if self.source.eof:
                        log.info("source exhausted")
                        break
                    # No new frame yet. Sleep well under the frame interval so we
                    # never become the reason a frame is late.
                    time.sleep(0.005)
                    self._tick_events()
                    self._flush_pending()
                    continue
                last_index = frame.index

                self._process(frame.image, frame.ts)
                self._tick_events()
                self._flush_pending()

                now = time.monotonic()
                if now - self._last_outbox_replay > 60.0:
                    self._last_outbox_replay = now
                    self.dispatcher.sender.replay_outbox()
                if now - last_report > 30.0:
                    last_report = now
                    log.info(
                        "%s | buffer %.1f MB, dropped %d",
                        self.stats.summary(),
                        self.buffer.nbytes() / 1024**2,
                        self.source.dropped,
                    )

        self._flush_pending(force=True)
        log.info("runner stopped: %s", self.stats.summary())
        return self.stats

    def _process(self, image, ts: float) -> None:
        self.stats.frames += 1

        # Always buffer, even if gated out. The pre-roll of an event includes the
        # calm seconds before it -- that is the part a human needs to judge it.
        self.buffer.append(image, ts)

        if not self.gate.check(image):
            return
        self.stats.gate_passed += 1

        clip = self.sampler.add(image, ts)
        if clip is None:
            return
        self.stats.clips_scored += 1

        scores = self.model.predict(clip)
        self._record(self.detector.update(scores, ts=time.time()))

    def _tick_events(self) -> None:
        """Close events that have gone quiet.

        Deliberately outside _process: once an incident ends the motion gate stops
        passing frames, so no clip is ever scored again and a score-driven close would
        never happen. See ClassThreshold.idle_close_s.
        """
        self._record(self.detector.tick(ts=time.time()))

    def _record(self, events: list[Event]) -> None:
        for event in events:
            self.stats.events += 1
            log.warning(
                "EVENT %s on %s: peak %.3f over %.1fs",
                event.label, event.camera_id, event.peak_score, event.duration,
            )
            self._pending.append(
                _PendingClip(event=event, write_after=time.monotonic() + self.config.storage.post_roll_s)
            )

    def _flush_pending(self, force: bool = False) -> None:
        """Write clips whose post-roll window has elapsed, then alert."""
        if not self._pending:
            return
        now = time.monotonic()
        still_pending = []
        for item in self._pending:
            if not force and now < item.write_after:
                still_pending.append(item)
                continue

            span = (
                self.config.storage.pre_roll_s
                + item.event.duration
                + self.config.storage.post_roll_s
            )
            frames = self.buffer.snapshot(seconds=span)
            name = f"{item.event.camera_id}-{item.event.label}-{int(item.event.started_at)}"
            path = self.clips.write(frames, name)

            payload = self.dispatcher.dispatch(
                item.event, clip_path=str(path) if path else None
            )
            if payload is not None:
                self.stats.alerts += 1
        self._pending = still_pending
