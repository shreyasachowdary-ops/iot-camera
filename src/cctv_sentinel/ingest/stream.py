"""RTSP / file / webcam reader.

Two non-obvious decisions, both of which matter more than they look:

1. **We drop frames, we never queue them.** If the model takes 150 ms and frames
   arrive every 100 ms, a queue grows without bound and within ten minutes you are
   alerting about something that happened ten minutes ago -- while every dashboard
   still looks healthy. Holding only the newest frame makes lateness impossible;
   `dropped` and `lag` are exported so you can see the pressure instead.

2. **We reconnect forever, with backoff.** Cameras drop. Switches reboot. A service
   that exits when a camera blinks is not a service. `camera_offline` is itself worth
   alerting on.

Live and file sources need genuinely different handling, and conflating them is a
subtle bug: a live camera delivers frames in real time, so you decimate by wall clock
and treat EOF as a disconnect. A file delivers frames as fast as the disk allows, so
wall-clock decimation throws away nearly all of them, and EOF means the stream ended.
Files are paced to real time by index instead.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import cv2
import numpy as np

from ..config import StreamConfig

log = logging.getLogger(__name__)

_LIVE_SCHEMES = {"rtsp", "rtsps", "rtmp", "http", "https", "udp", "tcp"}


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    index: int


class FrameSource:
    """Background reader exposing only the most recent frame."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self.dropped = 0
        self.connected = False
        self.eof = False
        """Set when a file source is exhausted. Live sources never set this."""

        self.is_live = self._detect_live(config.source)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FrameSource":
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- reading -----------------------------------------------------------

    def read(self) -> Frame | None:
        """Newest frame, or None if nothing has arrived yet."""
        with self._lock:
            return self._latest

    @staticmethod
    def _detect_live(source: str) -> bool:
        """A webcam index or a network url is live; an existing file path is not."""
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return True
        scheme = urlparse(str(source)).scheme.lower()
        if scheme in _LIVE_SCHEMES:
            return True
        return not os.path.exists(str(source))

    def _open(self) -> cv2.VideoCapture:
        source: str | int = self.config.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        # Ask the driver for the shallowest buffer it will give us; on RTSP this stops
        # OpenCV handing us stale frames after a stall.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - not supported by every backend
            pass
        return cap

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if not (self.config.width and self.config.height):
            return image
        h, w = image.shape[:2]
        if (w, h) == (self.config.width, self.config.height):
            return image
        return cv2.resize(
            image, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA
        )

    def _publish(self, image: np.ndarray, ts: float) -> None:
        self._index += 1
        with self._lock:
            if self._latest is not None:
                # Nobody consumed the previous frame. Expected and fine -- but count
                # it, because a rising rate means the model is falling behind real time.
                self.dropped += 1
            self._latest = Frame(image=image, ts=ts, index=self._index)

    def _run(self) -> None:
        delay = self.config.reconnect_delay_s
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                log.warning("cannot open source %s, retrying in %.1fs", self.config.source, delay)
                if not self.is_live:
                    self.eof = True
                    return
                self._stop.wait(delay)
                delay = min(delay * 2, self.config.reconnect_max_delay_s)
                continue

            self.connected = True
            delay = self.config.reconnect_delay_s
            log.info("connected to %s (%s)", self.config.source, "live" if self.is_live else "file")

            ended = self._pump_live(cap) if self.is_live else self._pump_file(cap)
            cap.release()
            self.connected = False

            if ended:  # file exhausted and not looping
                self.eof = True
                return
            self._stop.wait(delay)

    def _pump_live(self, cap: cv2.VideoCapture) -> bool:
        """Live source: frames arrive in real time, so decimate by wall clock."""
        min_interval = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0.0
        last_emit = 0.0
        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok:
                log.warning("read failed on %s, reconnecting", self.config.source)
                return False

            now = time.monotonic()
            # The camera may be sending 25 fps; the model neither needs nor benefits
            # from more than ~10.
            if min_interval and (now - last_emit) < min_interval:
                continue
            last_emit = now
            self._publish(self._resize(image), now)
        return False

    def _pump_file(self, cap: cv2.VideoCapture) -> bool:
        """File source: decimate by frame index, then pace to real time.

        Wall-clock decimation would be wrong here -- the whole file decodes in
        milliseconds, so almost every frame would fall inside the same interval and be
        discarded. Returns True when the file is exhausted and we should not loop.
        """
        native_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.target_fps
        step = max(1, int(round(native_fps / max(self.config.target_fps, 1e-6))))
        speed = max(self.config.file_speed, 1e-6)
        interval = step / native_fps / speed

        read_index = 0
        start = time.monotonic()
        emitted = 0

        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok:
                if self.config.loop_file:
                    log.info("looping file source %s", self.config.source)
                    return False
                log.info("end of file %s after %d frames", self.config.source, emitted)
                return True

            if read_index % step:
                read_index += 1
                continue
            read_index += 1

            # Pace playback so timestamps, event durations and latency measurements
            # mean the same thing they would against a live camera.
            target = start + emitted * interval
            slack = target - time.monotonic()
            if slack > 0 and self._stop.wait(slack):
                return False

            emitted += 1
            self._publish(self._resize(image), time.monotonic())
        return False
