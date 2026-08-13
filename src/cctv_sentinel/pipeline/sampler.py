"""Turn a frame stream into overlapping fixed-size clips for the model.

The model does not see "the video". It sees a 16-frame clip spanning ~2 seconds, once
per second. Consecutive clips overlap 50%, so an event cannot fall between two windows
and be missed.

Frames are held here by reference and dropped as they age out -- this is a bounded
working set, not a recording.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import SamplerConfig


class ClipSampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._last_emit_ts: float | None = None

    def add(self, frame: np.ndarray, ts: float) -> np.ndarray | None:
        """Add a frame. Returns a clip (T, H, W, 3) when one is ready, else None."""
        self._frames.append((ts, frame))

        # Evict anything older than the clip span; keep a little slack for jitter.
        cutoff = ts - self.config.clip_span_s * 1.5
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

        if self._last_emit_ts is not None and (ts - self._last_emit_ts) < self.config.stride_s:
            return None

        window_start = ts - self.config.clip_span_s
        window = [(t, f) for t, f in self._frames if t >= window_start]
        if len(window) < self.config.clip_frames:
            return None

        self._last_emit_ts = ts
        return self._resample(window)

    def _resample(self, window: list[tuple[float, np.ndarray]]) -> np.ndarray:
        """Pick `clip_frames` frames spread evenly across the window.

        Index-based rather than timestamp-based: frames arriving at an uneven rate
        (which they will, over RTSP) still yield a clip of exactly the right length.
        """
        n = self.config.clip_frames
        idx = np.linspace(0, len(window) - 1, num=n).round().astype(int)
        return np.stack([window[i][1] for i in idx])

    def reset(self) -> None:
        self._frames.clear()
        self._last_emit_ts = None
