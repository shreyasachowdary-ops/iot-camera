"""Tests for the parts that must not break: buffering, gating, event logic, signing."""

from __future__ import annotations

import numpy as np
import pytest

from cctv_sentinel.alerting.dispatcher import AlertDispatcher
from cctv_sentinel.alerting.webhook import sign
from cctv_sentinel.config import AlertConfig, ClassThreshold, Config, DetectConfig
from cctv_sentinel.detect.smoother import EventDetector
from cctv_sentinel.ingest.ring_buffer import RingBuffer
from cctv_sentinel.models.classifier import StubClassifier
from cctv_sentinel.config import ModelConfig, MotionGateConfig, SamplerConfig
from cctv_sentinel.pipeline.motion_gate import MotionGate
from cctv_sentinel.pipeline.sampler import ClipSampler


def frame(value: int = 0, size: tuple[int, int] = (90, 160)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


# -- ring buffer -----------------------------------------------------------


def test_ring_buffer_is_bounded():
    """The whole storage argument rests on this: it must never grow."""
    buf = RingBuffer(seconds=2.0, fps=10)
    for i in range(500):
        buf.append(frame(i % 256), ts=float(i))
    assert len(buf) == 20
    assert buf.nbytes() < 5 * 1024**2


def test_ring_buffer_keeps_newest_and_decodes():
    buf = RingBuffer(seconds=1.0, fps=10)
    for i in range(50):
        buf.append(frame(i), ts=float(i))
    frames = buf.snapshot()
    assert frames[-1].ts == 49.0
    assert frames[0].ts == 40.0
    assert frames[-1].decode().shape == (90, 160, 3)


def test_ring_buffer_snapshot_window():
    buf = RingBuffer(seconds=10.0, fps=10)
    for i in range(100):
        buf.append(frame(i % 256), ts=i * 0.1)
    recent = buf.snapshot(seconds=2.0)
    assert 19 <= len(recent) <= 21


# -- motion gate -----------------------------------------------------------


def test_motion_gate_blocks_static_scene():
    gate = MotionGate(MotionGateConfig())
    gate.check(frame(50))  # priming frame
    assert not any(gate.check(frame(50)) for _ in range(10))


def test_motion_gate_passes_on_change():
    gate = MotionGate(MotionGateConfig())
    gate.check(frame(0))
    assert gate.check(frame(255))


def test_motion_gate_cooldown_survives_a_still_instant():
    """A person mid-fall is momentarily motionless; the clip must not be cut in half."""
    gate = MotionGate(MotionGateConfig(cooldown_frames=5))
    gate.check(frame(0))
    assert gate.check(frame(255))
    assert all(gate.check(frame(255)) for _ in range(5))


# -- sampler ---------------------------------------------------------------


def test_sampler_emits_correct_shape_and_respects_stride():
    sampler = ClipSampler(SamplerConfig(clip_frames=16, clip_span_s=2.0, stride_s=1.0))
    clips = []
    for i in range(60):  # 6 s at 10 fps
        clip = sampler.add(frame(i % 256), ts=i * 0.1)
        if clip is not None:
            clips.append(clip)
    assert clips, "sampler produced no clips"
    assert clips[0].shape == (16, 90, 160, 3)
    # 6 s of frames, 2 s warm-up, 1 s stride -> ~4 clips
    assert 3 <= len(clips) <= 5


def test_sampler_waits_for_a_full_window():
    sampler = ClipSampler(SamplerConfig(clip_frames=16, clip_span_s=2.0, stride_s=1.0))
    assert all(sampler.add(frame(i), ts=i * 0.1) is None for i in range(10))


# -- event detection -------------------------------------------------------


def _detector(**kwargs) -> EventDetector:
    thresholds = ClassThreshold(
        open_at=0.7, close_at=0.4, open_k=3, open_n=5, ema_alpha=1.0, min_event_s=0.0
    )
    for key, value in kwargs.items():
        setattr(thresholds, key, value)
    return EventDetector("cam-01", ["normal", "fight"], DetectConfig(default=thresholds))


def test_single_spike_does_not_open_an_event():
    """The property that keeps 86,400 windows/day from producing 86 false alerts."""
    detector = _detector()
    for i, score in enumerate([0.1, 0.1, 0.99, 0.1, 0.1, 0.1]):
        assert detector.update({"fight": score}, ts=float(i)) == []
    assert detector.active == []


def test_sustained_scores_open_then_close_an_event():
    detector = _detector()
    for i, score in enumerate([0.9] * 4):
        assert detector.update({"fight": score}, ts=float(i)) == []
    assert len(detector.active) == 1

    closed = detector.update({"fight": 0.1}, ts=10.0)
    assert len(closed) == 1
    assert closed[0].label == "fight"
    assert closed[0].peak_score == pytest.approx(0.9)
    assert closed[0].duration == pytest.approx(8.0)


def test_hysteresis_holds_an_event_open_in_the_dead_band():
    """Scores between close_at and open_at must not flap the event."""
    detector = _detector()
    for i in range(4):
        detector.update({"fight": 0.9}, ts=float(i))
    assert len(detector.active) == 1
    for i in range(4, 8):
        assert detector.update({"fight": 0.55}, ts=float(i)) == []
    assert len(detector.active) == 1


def test_idle_close_fires_when_clips_stop_arriving():
    """Regression: an event used to hang open forever once the scene went quiet.

    Events close on a LOW score, but after a real incident ends the motion gate stops
    passing frames, so no clip is ever scored again. Without a time-based close the
    alert -- dispatched on close -- was never sent. The calmer the aftermath, the
    worse the failure.
    """
    detector = _detector(idle_close_s=3.0)
    for i in range(4):
        detector.update({"fight": 0.9}, ts=float(i))
    assert len(detector.active) == 1

    assert detector.tick(ts=5.0) == []      # not stale yet
    closed = detector.tick(ts=6.5)          # 3s since the last score at t=3
    assert len(closed) == 1
    assert closed[0].label == "fight"
    assert detector.active == []


def test_tick_is_a_noop_with_no_open_event():
    detector = _detector()
    assert detector.tick(ts=1000.0) == []


def test_max_event_duration_forces_a_close():
    """A camera on a busy road never settles; one event must not swallow a shift."""
    detector = _detector(max_event_s=10.0, close_at=0.0)
    for i in range(4):
        detector.update({"fight": 0.9}, ts=float(i))
    assert len(detector.active) == 1
    closed = detector.update({"fight": 0.9}, ts=20.0)
    assert len(closed) == 1
    assert detector.active == []


def test_short_events_are_discarded():
    detector = _detector(min_event_s=5.0)
    for i in range(4):
        detector.update({"fight": 0.9}, ts=i * 0.1)
    assert detector.update({"fight": 0.05}, ts=0.5) == []


def test_normal_class_never_produces_events():
    detector = _detector()
    for i in range(10):
        assert detector.update({"normal": 1.0, "fight": 0.0}, ts=float(i)) == []


# -- alerting --------------------------------------------------------------


def test_cooldown_suppresses_repeat_alerts(tmp_path):
    """One brawl is one alert, not forty."""
    config = AlertConfig(webhook_url=None, cooldown_s=60.0, outbox_dir=str(tmp_path))
    dispatcher = AlertDispatcher(config)
    detector = _detector()
    for i in range(4):
        detector.update({"fight": 0.9}, ts=float(i))
    event = detector.update({"fight": 0.1}, ts=10.0)[0]

    assert dispatcher.dispatch(event) is not None
    assert dispatcher.dispatch(event) is None


def test_payload_shape():
    config = AlertConfig(webhook_url=None)
    dispatcher = AlertDispatcher(config)
    detector = _detector()
    for i in range(4):
        detector.update({"fight": 0.9}, ts=float(i))
    event = detector.update({"fight": 0.1}, ts=10.0)[0]

    payload = dispatcher.build_payload(event, clip_path="var/clips/x.mp4")
    assert payload["label"] == "fight"
    assert payload["severity"] == "high"
    assert payload["camera_id"] == "cam-01"
    assert payload["requires_human_confirmation"] is True
    assert payload["clip_path"] == "var/clips/x.mp4"


def test_failed_send_is_spooled_to_outbox(tmp_path):
    """An alert lost to a router reboot is the worst possible bug here."""
    from cctv_sentinel.alerting.webhook import WebhookSender

    config = AlertConfig(
        webhook_url="http://127.0.0.1:9/nope", max_retries=1, timeout_s=0.2,
        outbox_dir=str(tmp_path / "outbox"),
    )
    sender = WebhookSender(config)
    assert sender.send({"label": "fight"}) is False
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1


def test_hmac_signature_binds_the_timestamp():
    body = b'{"a":1}'
    assert sign("secret", body, "100") != sign("secret", body, "200")
    assert sign("secret", body, "100") == sign("secret", body, "100")


# -- model + config --------------------------------------------------------


def test_stub_classifier_returns_a_distribution():
    model = StubClassifier(ModelConfig(classes=["normal", "fight", "fall"]))
    clip = np.random.randint(0, 255, (16, 90, 160, 3), dtype=np.uint8)
    scores = model.predict(clip)
    assert set(scores) == {"normal", "fight", "fall"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_stub_scores_motion_above_stillness():
    model = StubClassifier(ModelConfig(classes=["normal", "fight"]))
    still = np.stack([frame(100)] * 16)
    moving = np.stack([frame((i * 60) % 256) for i in range(16)])
    assert model.predict(moving)["fight"] > model.predict(still)["fight"]


def test_stub_can_actually_cross_the_shipped_threshold():
    """Regression: the stub used to divide its score across the non-normal classes,
    capping it at 1/N. With 3 classes it maxed out at 0.5 and could never reach the
    0.75 open_at in configs/default.yaml -- so the whole 'prove the pipeline without a
    model' workflow silently did nothing."""
    config = Config.load("configs/default.yaml")
    model = StubClassifier(config.model)
    violent = np.stack([frame((i * 90) % 256) for i in range(16)])
    score = model.predict(violent)[model.stub_class]
    assert score > config.detect.for_class(model.stub_class).open_at


def test_stub_rejects_a_class_list_it_cannot_report_on():
    with pytest.raises(ValueError):
        StubClassifier(ModelConfig(classes=["normal"]))
    with pytest.raises(ValueError):
        StubClassifier(ModelConfig(classes=["normal", "fight"], stub_class="nope"))


def test_config_roundtrip_and_nested_override():
    config = Config.from_dict(
        {
            "camera_id": "gate-cam",
            "stream": {"source": "rtsp://x/1", "target_fps": 5.0},
            "detect": {"per_class": {"fall": {"open_at": 0.6}}},
        }
    )
    assert config.camera_id == "gate-cam"
    assert config.stream.target_fps == 5.0
    assert config.stream.width == 640  # untouched default
    assert config.detect.for_class("fall").open_at == 0.6
    assert config.detect.for_class("fight").open_at == 0.75  # falls back to default


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"stream": {"typo_here": 1}})


def test_shipped_config_is_valid():
    config = Config.load("configs/default.yaml")
    assert "normal" in config.model.classes
    assert config.detect.for_class("fall").open_at < config.detect.for_class("fight").open_at
    # hysteresis must actually be hysteresis
    for label in config.model.classes:
        t = config.detect.for_class(label)
        assert t.close_at < t.open_at
    # a clip must fit in the ring buffer, or pre-roll is silently truncated
    assert config.storage.pre_roll_s + config.storage.post_roll_s <= config.ring_buffer.seconds


# -- source classification -------------------------------------------------


@pytest.mark.parametrize(
    "source,is_live",
    [
        ("0", True),
        ("rtsp://user:pass@10.0.0.5:554/Streaming/Channels/102", True),
        ("http://cam.local/stream.mjpg", True),
        ("/no/such/file.mp4", True),   # unknown path -> assume live, keep retrying
    ],
)
def test_live_source_detection(source, is_live):
    from cctv_sentinel.ingest.stream import FrameSource

    assert FrameSource._detect_live(source) is is_live


def test_existing_file_is_not_live(tmp_path):
    """Regression: files were decimated by wall clock, which discarded nearly every
    frame, and EOF was treated as a disconnect so playback looped forever."""
    from cctv_sentinel.ingest.stream import FrameSource

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    assert FrameSource._detect_live(str(path)) is False
