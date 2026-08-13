"""Cheapest possible filter: has anything moved?

~95% of CCTV footage is an empty corridor. This gate costs ~0.1 ms per frame and
removes the overwhelming majority of the work before anything expensive runs. It is
the single reason one small box can cover eight cameras.

Deliberately a simple absolute difference rather than MOG2 background subtraction:
it has no state to corrupt, no warm-up period, and no surprises when the scene
changes. MOG2 handles swaying trees and flickering tube lights better and is the
upgrade path once you're tuning against a real site.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import MotionGateConfig


class MotionGate:
    def __init__(self, config: MotionGateConfig) -> None:
        self.config = config
        self._prev: np.ndarray | None = None
        self._open_for = 0
        self.last_score = 0.0

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, self.config.downscale_to, interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(small, (5, 5), 0)

    def check(self, frame: np.ndarray) -> bool:
        """True if this frame should be passed downstream."""
        if not self.config.enabled:
            return True

        current = self._prepare(frame)
        if self._prev is None:
            self._prev = current
            return False

        delta = cv2.absdiff(current, self._prev)
        self._prev = current

        changed = int(np.count_nonzero(delta > self.config.pixel_delta))
        self.last_score = changed / float(delta.size)

        if self.last_score >= self.config.min_changed_fraction:
            # Hold the gate open briefly. Real events contain still instants -- a
            # person mid-fall is momentarily motionless -- and we don't want the clip
            # chopped in half.
            self._open_for = self.config.cooldown_frames
            return True

        if self._open_for > 0:
            self._open_for -= 1
            return True

        return False
