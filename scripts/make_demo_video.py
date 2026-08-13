"""Render a single continuous video with a calm stretch, an incident, then calm again.

Used by demo.sh to show live inference: the detector should stay quiet, fire during the
incident, and settle afterwards. Unlike the clips from make_synthetic_dataset.py this is
one unbroken stream, so it exercises the sliding-window and event logic rather than
classification alone.

**The geometry here deliberately matches make_synthetic_dataset.py** -- same 320x240
frame, same 34x70 people, same floor line. That is not incidental. Rendering this at
640x360 instead makes people occupy ~19% of frame height instead of ~29%, and the
model's score on the incident drops from 0.95 to 0.36 -- below threshold, so nothing
fires at all. Same model, same event, same motion; only the apparent scale changed.

That is the benchmark-to-site domain gap in miniature, and it is the single most common
reason a model that scored well in training does nothing useful on real cameras. On real
footage you cannot fix it by regenerating the video: you fix it by training on clips
from the actual cameras, at the actual mounting heights. See docs/04-tradeoffs.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a demo incident video")
    parser.add_argument("--out", type=Path, default=Path("data/demo_incident.mp4"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--incident-start", type=float, default=12.0)
    parser.add_argument("--incident-end", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=10)
    # 320x240 to match make_synthetic_dataset.py exactly -- see the module docstring.
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    rng = np.random.default_rng(0)
    floor_y = args.height - 40
    people = [{"x": 80.0, "vx": 0.8}, {"x": 200.0, "vx": -0.6}]

    for i in range(int(args.seconds * args.fps)):
        t = i / args.fps
        frame = np.full((args.height, args.width, 3), 40, np.uint8)
        frame = (frame + rng.normal(0, 2, frame.shape)).clip(0, 255).astype(np.uint8)
        cv2.rectangle(frame, (0, floor_y), (args.width, args.height), (68, 68, 68), -1)

        incident = args.incident_start < t < args.incident_end
        for person, colour in zip(people, [(200, 180, 160), (160, 200, 180)]):
            if incident:
                person["x"] += rng.normal(0, 9)
                dy = rng.normal(0, 7)
            else:
                person["vx"] += rng.normal(0, 0.1)
                person["vx"] = float(np.clip(person["vx"], -1.6, 1.6))
                person["x"] += person["vx"]
                dy = np.sin(t * 3) * 1.5
            person["x"] = float(np.clip(person["x"], 10, args.width - 50))
            y = floor_y - 70 + dy
            cv2.rectangle(frame, (int(person["x"]), int(y)),
                          (int(person["x"]) + 34, int(y) + 70), colour, -1)

        label = "INCIDENT" if incident else "normal"
        cv2.putText(frame, f"t={t:5.1f}s  {label}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"wrote {args.out} ({args.seconds:.0f}s, incident at "
          f"{args.incident_start:.0f}-{args.incident_end:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
