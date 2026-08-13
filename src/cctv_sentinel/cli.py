"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import Config
from .detect.smoother import Event


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    from .service.runner import Runner

    config = Config.load(args.config)
    if args.source:
        config.stream.source = args.source
    if args.camera_id:
        config.camera_id = args.camera_id
    if args.webhook:
        config.alert.webhook_url = args.webhook

    runner = Runner(config)
    runner.run(max_seconds=args.duration)
    return 0


def cmd_demo_alert(args: argparse.Namespace) -> int:
    """Print the webhook payload without sending it, so a backend can be built first."""
    from .alerting.dispatcher import AlertDispatcher

    config = Config.load(args.config) if args.config else Config()
    now = time.time()
    event = Event(
        camera_id=config.camera_id,
        label="fight",
        started_at=now - 4.2,
        ended_at=now,
        peak_score=0.91,
        scores=[0.78, 0.85, 0.91, 0.88, 0.62],
    )
    dispatcher = AlertDispatcher(config.alert)
    payload = dispatcher.build_payload(event, clip_path="var/clips/cam-01-fight-1723526400.mp4")
    payload["event_id"] = "018f3c7e-0000-7000-8000-demo00000001"
    print(json.dumps(payload, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Check a camera url is reachable and report what it actually delivers."""
    import cv2

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"FAILED to open {args.source}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    n, start = 0, time.monotonic()
    while n < 60 and (time.monotonic() - start) < 10:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    elapsed = time.monotonic() - start
    cap.release()

    print(f"source        {args.source}")
    print(f"resolution    {width}x{height}")
    print(f"reported fps  {reported_fps:.1f}")
    print(f"measured fps  {n / elapsed:.1f} ({n} frames in {elapsed:.1f}s)")
    if width > 1280:
        print("\nNOTE: that looks like a main stream. Use the camera's substream for")
        print("analytics (Hikvision /Streaming/Channels/102, Dahua subtype=1) -- the")
        print("model runs at 224x224 anyway. See docs/01-architecture.md.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cctv-sentinel")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the detection service")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--source", help="override stream source (rtsp url, file, or webcam index)")
    run.add_argument("--camera-id", help="override camera id")
    run.add_argument("--webhook", help="override webhook url")
    run.add_argument("--duration", type=float, help="stop after N seconds (for testing)")
    run.set_defaults(func=cmd_run)

    demo = sub.add_parser("demo-alert", help="print a sample webhook payload")
    demo.add_argument("--config", default=None)
    demo.set_defaults(func=cmd_demo_alert)

    probe = sub.add_parser("probe", help="test a camera url and report what it delivers")
    probe.add_argument("source")
    probe.set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
