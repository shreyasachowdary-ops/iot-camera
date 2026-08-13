"""Turn a folder of raw videos into a training-ready cache.

Two jobs, and the second one is the one that protects your results:

1. **Decode once.** Decoding video is ~95% of epoch time. Every clip is decoded exactly
   once into a single uint8 memmap; training then reads that. Epoch time drops from
   tens of minutes to tens of seconds, and it is resumable if interrupted.

2. **Split by GROUP, not by clip.** If clips from the same camera or the same recording
   session land in both train and val, the model recognises the *scene* rather than the
   *action* and validation scores are inflated -- routinely by 20+ points. You report
   95%, deploy, and get 65%. See docs/00-primer.md section 7.

Expected input layout:

    raw/
      fight/   cam01_2024-05-03_0007.mp4  ...
      normal/  cam01_2024-05-03_0012.mp4  ...

The group is parsed from the filename (everything before the first `_` by default), so
`cam01_*` all lands on one side of the split.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class ClipRecord:
    path: Path
    label: int
    label_name: str
    group: str
    split: str = ""


@dataclass
class PrepareStats:
    total: int = 0
    decoded: int = 0
    failed: list[str] = field(default_factory=list)
    per_split: dict[str, Counter] = field(default_factory=dict)


def parse_group(path: Path, pattern: str | None) -> str:
    """Group key for leakage-safe splitting.

    Default: text before the first underscore, so `cam01_2024-05-03_0007.mp4` -> `cam01`.
    Pass a regex with one capture group to override.
    """
    if pattern:
        match = re.search(pattern, path.name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        log.warning("group pattern did not match %s; using the filename as its own group", path.name)
        return path.stem
    return path.stem.split("_")[0] if "_" in path.stem else path.stem


def scan(raw_dir: Path, classes: list[str], group_pattern: str | None) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for index, name in enumerate(classes):
        class_dir = raw_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class directory: {class_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no video files under {class_dir}")
        for path in files:
            records.append(
                ClipRecord(path=path, label=index, label_name=name,
                           group=parse_group(path, group_pattern))
            )
    return records


def assign_splits(
    records: list[ClipRecord], ratios: tuple[float, float, float], seed: int
) -> None:
    """Group-aware split. Every clip from a group goes to exactly one split.

    Groups are shuffled then greedily assigned to whichever split is furthest below its
    target share. Greedy rather than proportional because group sizes are uneven -- one
    camera may have 10x the clips of another, and a naive split would badly miss the
    ratios.
    """
    by_group: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)

    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    names = ["train", "val", "test"]
    targets = {name: ratio for name, ratio in zip(names, ratios)}
    counts = {name: 0 for name in names}
    total = len(records)

    # Largest groups first: placing the big ones early stops a late giant group from
    # blowing past a target it can no longer compensate for.
    for group in sorted(groups, key=lambda g: -len(by_group[g])):
        deficits = {
            name: targets[name] - (counts[name] / total if total else 0.0) for name in names
        }
        chosen = max(names, key=lambda name: (deficits[name], targets[name]))
        for record in by_group[group]:
            record.split = chosen
        counts[chosen] += len(by_group[group])

    n_groups = len(groups)
    if n_groups < 6:
        log.warning(
            "only %d distinct groups (%s). Group-aware splitting cannot make a "
            "meaningful validation set from this -- results will be dominated by which "
            "group landed where. Collect footage from more cameras/sessions, or expect "
            "wide error bars.",
            n_groups, ", ".join(groups[:8]),
        )


def decode_clip(path: Path, num_frames: int, size: int) -> np.ndarray | None:
    """Decode a video to exactly `num_frames` evenly spaced RGB frames at size x size."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        return None

    # Evenly spaced across the whole clip. If the video is shorter than num_frames the
    # indices repeat, which pads by duplication rather than failing.
    idx = np.linspace(0, len(frames) - 1, num=num_frames).round().astype(int)
    out = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for i, j in enumerate(idx):
        resized = cv2.resize(frames[j], (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return out


def prepare(
    raw_dir: Path,
    out_dir: Path,
    classes: list[str],
    num_frames: int = 16,
    size: int = 224,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    group_pattern: str | None = None,
) -> PrepareStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    records = scan(raw_dir, classes, group_pattern)
    assign_splits(records, ratios, seed)
    records.sort(key=lambda r: (r.split, r.label, str(r.path)))

    stats = PrepareStats(total=len(records))
    shape = (len(records), num_frames, size, size, 3)
    x_path = out_dir / "clips.u8"

    done = 0
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("shape") == list(shape) and previous.get("signature") == _signature(records):
            done = previous.get("done", 0)
            log.info("resuming decode at %d/%d", done, len(records))
        else:
            log.info("cache is stale (dataset changed); rebuilding from scratch")

    mode = "r+" if x_path.exists() and done else "w+"
    x = np.memmap(x_path, dtype=np.uint8, mode=mode, shape=shape)

    for i in range(done, len(records)):
        record = records[i]
        clip = decode_clip(record.path, num_frames, size)
        if clip is None:
            stats.failed.append(str(record.path))
            log.warning("could not decode %s (writing zeros)", record.path)
            x[i] = 0
        else:
            x[i] = clip
            stats.decoded += 1

        if (i + 1) % 25 == 0 or i + 1 == len(records):
            x.flush()
            meta_path.write_text(json.dumps({
                "done": i + 1, "shape": list(shape), "signature": _signature(records),
            }))
            log.info("decoded %d/%d", i + 1, len(records))

    x.flush()

    labels = np.array([r.label for r in records], dtype=np.int64)
    np.save(out_dir / "labels.npy", labels)
    (out_dir / "manifest.csv").write_text(
        "index,path,label,label_name,group,split\n"
        + "".join(
            f"{i},{r.path},{r.label},{r.label_name},{r.group},{r.split}\n"
            for i, r in enumerate(records)
        )
    )

    for split in ["train", "val", "test"]:
        stats.per_split[split] = Counter(r.label_name for r in records if r.split == split)

    meta_path.write_text(json.dumps({
        "done": len(records),
        "shape": list(shape),
        "signature": _signature(records),
        "classes": classes,
        "num_frames": num_frames,
        "size": size,
        "groups": sorted({r.group for r in records}),
        "splits": {k: dict(v) for k, v in stats.per_split.items()},
    }, indent=2))

    return stats


def _signature(records: list[ClipRecord]) -> str:
    """Cheap fingerprint so a changed dataset invalidates the cache."""
    import hashlib

    h = hashlib.sha256()
    for record in records:
        h.update(f"{record.path}|{record.label}|{record.split}\n".encode())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode raw videos into a training cache")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", default=["normal", "fight"],
                        help="subdirectory names; ORDER DEFINES LABEL INDICES")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-pattern", default=None,
                        help="regex with one capture group for the group key; "
                             "default is the filename prefix before the first '_'")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    stats = prepare(
        raw_dir=args.raw_dir, out_dir=args.out_dir, classes=args.classes,
        num_frames=args.num_frames, size=args.size, ratios=tuple(args.ratios),
        seed=args.seed, group_pattern=args.group_pattern,
    )

    print(f"\nprepared {stats.total} clips ({stats.decoded} decoded, {len(stats.failed)} failed)")
    for split, counter in stats.per_split.items():
        total = sum(counter.values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        print(f"  {split:<6} {total:>5} clips  ({detail})")
    if stats.failed:
        print(f"\nfailed to decode {len(stats.failed)} files, first few:")
        for path in stats.failed[:5]:
            print("   ", path)
    print("\nCheck the class balance above before trusting any accuracy number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
