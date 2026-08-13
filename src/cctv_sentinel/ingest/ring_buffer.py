"""Fixed-size in-memory ring buffer of recent frames.

This is the answer to "we cannot store everything". Frames go in JPEG-encoded, the
oldest is silently evicted, and memory usage is constant and known at startup. Nothing
touches disk until an alert fires and someone calls `snapshot()`.

Sizing, per camera at 640x360 / 10 fps / 30 s:

    raw BGR   640*360*3*300  = 207 MB     <- would not survive 8 cameras
    JPEG q=80 ~35 KB * 300   =  10 MB     <- fine

The JPEG encode costs ~1-2 ms of one core per frame. Worth it.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    ts: float
    """Monotonic capture timestamp."""

    jpeg: bytes

    def decode(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


class RingBuffer:
    """Thread-safe bounded buffer of the most recent `seconds` of frames."""

    def __init__(self, seconds: float, fps: float, jpeg_quality: int = 80) -> None:
        self.capacity = max(1, int(round(seconds * fps)))
        self.jpeg_quality = jpeg_quality
        self._frames: deque[BufferedFrame] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray, ts: float) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:  # pragma: no cover - opencv only fails here on a malformed frame
            return
        with self._lock:
            self._frames.append(BufferedFrame(ts=ts, jpeg=encoded.tobytes()))

    def snapshot(self, seconds: float | None = None) -> list[BufferedFrame]:
        """Copy out the most recent `seconds` of frames. Called only on an alert."""
        with self._lock:
            frames = list(self._frames)
        if seconds is None or not frames:
            return frames
        cutoff = frames[-1].ts - seconds
        return [f for f in frames if f.ts >= cutoff]

    def nbytes(self) -> int:
        with self._lock:
            return sum(len(f.jpeg) for f in self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __iter__(self) -> Iterator[BufferedFrame]:
        return iter(self.snapshot())
