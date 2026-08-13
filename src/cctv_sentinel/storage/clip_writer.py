"""Write evidence clips -- the ONLY thing this system ever puts on disk.

Pre-roll comes free from the ring buffer (those frames were already in RAM), post-roll
is collected for a few seconds after the event closes. Result: ~40 s per incident
instead of 24 h per camera.

The directory is hard-capped with LRU eviction. Inference must not be able to fill a
disk -- a full disk on an edge box means the whole service stops, which is a much
worse outcome than losing the oldest clip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ..config import StorageConfig
from ..ingest.ring_buffer import BufferedFrame

log = logging.getLogger(__name__)


class ClipWriter:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.dir = Path(config.clip_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, frames: list[BufferedFrame], name: str) -> Path | None:
        if not frames:
            log.warning("no frames to write for clip %s", name)
            return None

        path = self.dir / f"{name}.mp4"
        first = frames[0].decode()
        if first is None:
            log.error("could not decode first frame of clip %s", name)
            return None
        height, width = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.config.fps, (width, height))
        if not writer.isOpened():
            log.error("cannot open video writer for %s", path)
            return None

        try:
            for buffered in frames:
                image = buffered.decode()
                if image is None:
                    continue
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                writer.write(image)
        finally:
            writer.release()

        log.info("wrote clip %s (%d frames, %.1f KB)", path, len(frames), path.stat().st_size / 1024)
        self.enforce_quota()
        return path

    def enforce_quota(self) -> None:
        clips = sorted(self.dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in clips)
        while total > self.config.max_dir_bytes and clips:
            oldest = clips.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            log.warning("clip quota exceeded, evicted %s", oldest.name)
