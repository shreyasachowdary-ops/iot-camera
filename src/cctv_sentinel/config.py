"""Configuration objects, loaded from YAML.

Deliberately plain dataclasses rather than pydantic: this has to install cleanly on a
Jetson, and every dependency is a dependency you have to cross-compile one day.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamConfig:
    source: str = "0"
    """RTSP url, file path, or a webcam index as a string."""

    target_fps: float = 10.0
    """Frames per second handed downstream. The camera may send more; we drop."""

    width: int = 640
    height: int = 360
    reconnect_delay_s: float = 2.0
    reconnect_max_delay_s: float = 30.0

    loop_file: bool = False
    """File sources only: restart at EOF instead of ending the stream. Useful for
    soak-testing against a recorded clip."""

    file_speed: float = 1.0
    """File sources only. 1.0 plays at real time, which is what you want when
    validating latency and event timing. Raise it to replay hours of footage quickly
    when sweeping thresholds -- but note the ring buffer and event durations are then
    in file-time, not wall-time."""


@dataclass
class RingBufferConfig:
    seconds: float = 30.0
    jpeg_quality: int = 80
    """Frames are JPEG-encoded on the way in. Raw BGR at 640x360x10fps is 207 MB per
    30 s per camera; JPEG is ~10 MB. See docs/02."""


@dataclass
class MotionGateConfig:
    enabled: bool = True
    downscale_to: tuple[int, int] = (160, 90)
    pixel_delta: int = 25
    """Per-pixel intensity change counted as 'moved'."""

    min_changed_fraction: float = 0.004
    """Fraction of pixels that must move for the frame to pass the gate."""

    cooldown_frames: int = 20
    """Once motion is seen, keep the gate open this many frames. Stops a clip being
    chopped in half by a momentary still instant mid-event."""


@dataclass
class SamplerConfig:
    clip_frames: int = 16
    """Frames per clip handed to the model."""

    clip_span_s: float = 2.0
    """Wall-clock duration the clip covers. Frames are sampled evenly across it."""

    stride_s: float = 1.0
    """How often a new clip is scored. < clip_span_s means windows overlap, so an
    event cannot fall between two windows."""


@dataclass
class ModelConfig:
    backend: str = "stub"
    """stub | torch | onnx"""

    weights: str | None = None
    classes: list[str] = field(default_factory=lambda: ["normal", "fight", "fall"])
    input_size: int = 224
    device: str = "auto"

    stub_class: str | None = None
    """stub backend only: which class the heuristic reports on. Defaults to the first
    non-normal class. The stub cannot tell classes apart, so it must commit its whole
    score to one of them -- splitting it across N classes would cap the score at 1/N
    and the event thresholds could never be reached."""


@dataclass
class ClassThreshold:
    open_at: float = 0.75
    """Score needed to OPEN an event."""

    close_at: float = 0.45
    """Score below which an open event closes. Lower than open_at on purpose --
    hysteresis, so a score hovering at the boundary doesn't flap."""

    open_k: int = 3
    """Need this many of the last `open_n` windows above open_at."""

    open_n: int = 5
    ema_alpha: float = 0.5
    min_event_s: float = 1.5
    """Events shorter than this are discarded as noise."""

    idle_close_s: float = 3.0
    """Close an open event if no clip has been scored for this long.

    Load-bearing, not a safety net. An event closes on a LOW score, but after a real
    incident ends the scene goes still, the motion gate stops passing frames, and no
    further clips are ever scored -- so without this the event stays open forever and
    the alert is never dispatched. The calmer the aftermath, the worse the failure."""

    max_event_s: float = 120.0
    """Hard cap. Forces a close on a camera where the scene never settles (a busy
    road, a factory line) so one event cannot swallow an entire shift."""


@dataclass
class DetectConfig:
    default: ClassThreshold = field(default_factory=ClassThreshold)
    per_class: dict[str, ClassThreshold] = field(default_factory=dict)

    def for_class(self, name: str) -> ClassThreshold:
        return self.per_class.get(name, self.default)


@dataclass
class AlertConfig:
    webhook_url: str | None = None
    hmac_secret: str | None = None
    timeout_s: float = 10.0
    max_retries: int = 5
    cooldown_s: float = 120.0
    """Per (camera, class): suppress repeat alerts within this window."""

    outbox_dir: str = "var/outbox"
    """Failed webhooks are spooled here and replayed. Survives an internet outage."""


@dataclass
class StorageConfig:
    clip_dir: str = "var/clips"
    pre_roll_s: float = 20.0
    post_roll_s: float = 10.0
    fps: float = 10.0
    max_dir_bytes: int = 5 * 1024**3
    """Hard cap on the clip directory. Oldest clips are evicted. Inference must never
    be able to fill a disk."""


@dataclass
class Config:
    camera_id: str = "cam-01"
    stream: StreamConfig = field(default_factory=StreamConfig)
    ring_buffer: RingBufferConfig = field(default_factory=RingBufferConfig)
    motion_gate: MotionGateConfig = field(default_factory=MotionGateConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown config key: {key!r}")
            current = getattr(cfg, key)
            if dataclasses.is_dataclass(current) and isinstance(value, dict):
                setattr(cfg, key, _merge(current, value))
            else:
                setattr(cfg, key, value)
        return cfg


def _merge(obj: Any, patch: dict[str, Any]) -> Any:
    """Recursively overlay a dict onto a dataclass instance."""
    fields = {f.name: f for f in dataclasses.fields(obj)}
    updates: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in fields:
            raise ValueError(f"unknown config key: {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if key == "per_class" and isinstance(value, dict):
            updates[key] = {
                cls_name: _merge(ClassThreshold(), sub) for cls_name, sub in value.items()
            }
        elif dataclasses.is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return dataclasses.replace(obj, **updates)
