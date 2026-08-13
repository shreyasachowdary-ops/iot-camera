"""Generate a small synthetic video dataset so the pipeline can be run end to end.

This exists so you can demo the whole thing with no data, no GPU and no downloads. It
is NOT a research dataset -- "fight" is fast erratic motion and "normal" is slow steady
motion, which is a far easier problem than the real one. High scores here mean the
plumbing works, nothing more.

Filenames follow `cam<NN>_<seq>.mp4` so group-aware splitting has something to group by,
exactly as your real footage should be named.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_person(frame: np.ndarray, x: int, y: int, w: int, h: int, colour) -> None:
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), colour, -1)


def make_clip(path: Path, kind: str, seed: int, width: int = 320, height: int = 240,
              fps: int = 15, seconds: float = 3.0, camera: int = 0) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Each camera gets its own background and lighting so that group-aware splitting is
    # actually doing something -- a random split would let the model learn the camera.
    base = 30 + camera * 12
    floor_y = height - 40 - camera * 5
    colours = [(200, 180, 160), (160, 200, 180)]

    n_frames = int(fps * seconds)
    people = [
        {"x": rng.uniform(40, width - 90), "y": floor_y - 70, "vx": rng.uniform(-1, 1)}
        for _ in range(2)
    ]

    for i in range(n_frames):
        frame = np.full((height, width, 3), base, np.uint8)
        frame = (frame + rng.normal(0, 3, frame.shape)).clip(0, 255).astype(np.uint8)
        cv2.rectangle(frame, (0, floor_y), (width, height), (base + 25,) * 3, -1)

        for p, colour in zip(people, colours):
            if kind == "fight":
                # Fast, erratic, high-frequency motion.
                p["x"] += rng.normal(0, 9)
                jitter_y = rng.normal(0, 7)
            else:
                # Slow, steady walking.
                p["vx"] += rng.normal(0, 0.15)
                p["vx"] = float(np.clip(p["vx"], -1.6, 1.6))
                p["x"] += p["vx"]
                jitter_y = np.sin(i * 0.3) * 1.5

            p["x"] = float(np.clip(p["x"], 10, width - 50))
            draw_person(frame, p["x"], p["y"] + jitter_y, 34, 70, colour)

        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--per-class-per-camera", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in ["normal", "fight"]:
        (args.out_dir / name).mkdir(parents=True, exist_ok=True)

    seed = args.seed
    total = 0
    for camera in range(args.cameras):
        for kind in ["normal", "fight"]:
            for j in range(args.per_class_per_camera):
                path = args.out_dir / kind / f"cam{camera:02d}_{j:03d}.mp4"
                make_clip(path, kind, seed=seed, camera=camera)
                seed += 1
                total += 1

    print(f"wrote {total} clips to {args.out_dir}")
    print(f"  {args.cameras} cameras x {args.per_class_per_camera} clips x 2 classes")
    print("\nNOTE: synthetic data. 'fight' is fast erratic motion, 'normal' is slow "
          "walking.\nThis is a much easier problem than real footage -- high scores here "
          "prove the\npipeline runs, not that the approach works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
