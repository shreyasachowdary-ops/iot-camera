"""Dataset over the prepared cache, with deployment-shaped augmentation.

Augmentation exists to stop the model memorising (docs/00-primer.md section 7). The
choice of augmentations matters as much as having them: augment toward the conditions
you will actually face. For CCTV, night/IR and low-bitrate simulation matter far more
than the textbook flips -- half your real incidents happen after dark, and IR footage
looks nothing like daylight footage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet/Kinetics statistics. MUST match whatever the pretrained backbone used, and
# must match preprocess() in models/classifier.py at inference. Mismatch costs a few
# points silently -- no error, just worse.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def read_manifest(cache_dir: Path) -> list[dict]:
    lines = (cache_dir / "manifest.csv").read_text().strip().split("\n")
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        # A path could contain commas; everything after the first field and before the
        # last four is the path.
        row = dict(zip(header, [parts[0], ",".join(parts[1:-4]), *parts[-4:]]))
        rows.append(row)
    return rows


class ClipDataset(Dataset):
    """Reads decoded clips from the memmap produced by prepare.py."""

    def __init__(self, cache_dir: Path, split: str, train: bool | None = None,
                 augment: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.classes: list[str] = meta["classes"]
        self.num_frames: int = meta["num_frames"]
        self.size: int = meta["size"]
        self.shape = tuple(meta["shape"])

        rows = read_manifest(self.cache_dir)
        self.indices = [int(r["index"]) for r in rows if r["split"] == split]
        if not self.indices:
            raise ValueError(f"no clips in split {split!r} under {cache_dir}")

        all_labels = np.load(self.cache_dir / "labels.npy")
        self.labels = all_labels[self.indices]
        self.groups = [rows[i]["group"] for i in self.indices]
        self.split = split
        self.train = (split == "train") if train is None else train
        self.augment = augment and self.train
        self._x: np.memmap | None = None  # opened lazily, see below

    @property
    def x(self) -> np.memmap:
        # Opened lazily and per-process: a memmap cannot be safely inherited across a
        # DataLoader worker fork, and doing so gives silently corrupted reads.
        if self._x is None:
            self._x = np.memmap(
                self.cache_dir / "clips.u8", dtype=np.uint8, mode="r", shape=self.shape
            )
        return self._x

    def class_counts(self) -> dict[str, int]:
        return {name: int((self.labels == i).sum()) for i, name in enumerate(self.classes)}

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, for imbalanced data.

        Without this, a set that is 95% normal trains a model that always says normal
        and looks 95% accurate. See docs/00-primer.md section 10.
        """
        counts = np.array([(self.labels == i).sum() for i in range(len(self.classes))])
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        clip = torch.from_numpy(np.array(self.x[self.indices[i]]))  # (T,H,W,C) uint8
        clip = clip.permute(3, 0, 1, 2).float().div_(255.0)         # (C,T,H,W) 0..1

        if self.augment:
            clip = self._augment(clip)

        return (clip - MEAN) / STD, int(self.labels[i])

    def _augment(self, clip: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if random.random() < 0.5:                      # horizontal flip
            clip = clip.flip(-1)

        if random.random() < 0.3:                      # night / infrared
            grey = clip.mean(0, keepdim=True).expand_as(clip)
            clip = (grey * random.uniform(0.4, 0.85)).clamp(0, 1)
            clip = (clip + torch.randn_like(clip) * 0.03).clamp(0, 1)

        if random.random() < 0.3:                      # low-bitrate substream
            small = random.choice([64, 96, 128])
            t = clip.transpose(0, 1)                   # (T,C,H,W) for interpolate
            t = F.interpolate(t, size=(small, small), mode="bilinear", align_corners=False)
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
            clip = t.transpose(0, 1)

        if random.random() < 0.5:                      # brightness / contrast
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(-0.1, 0.1)
            clip = ((clip - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)

        if random.random() < 0.3:                      # random resized crop
            scale = random.uniform(0.75, 1.0)
            crop = int(self.size * scale)
            top = random.randint(0, self.size - crop)
            left = random.randint(0, self.size - crop)
            cropped = clip[:, :, top:top + crop, left:left + crop]
            clip = F.interpolate(
                cropped.transpose(0, 1), size=(self.size, self.size),
                mode="bilinear", align_corners=False,
            ).transpose(0, 1)

        return clip
