"""Evaluate a checkpoint and produce the numbers you would put in a write-up.

Deliberately opinionated about method:

  * The decision threshold is chosen on VALIDATION and then applied unchanged to test.
    Choosing it on test means reporting the best of many tries as a single honest
    estimate, which it is not.
  * PR-AUC leads, accuracy is shown with a warning when classes are imbalanced.
  * Per-group results are printed, because a model that works on 4 cameras and fails on
    the 5th has a problem an average will hide.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import engine, metrics, models
from .dataset import ClipDataset

log = logging.getLogger(__name__)


def predict(model: nn.Module, dataset: ClipDataset, device: torch.device,
            batch_size: int = 8, num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for clips, labels in loader:
            logits = model(clips.to(device))
            probs.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def per_group_report(y: np.ndarray, probs: np.ndarray, groups: list[str],
                     threshold: float) -> str:
    """Break results down by camera/session. Averages hide per-camera failure."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        buckets[group].append(i)

    lines = [f"{'group':<20} {'n':>5} {'recall':>8} {'precision':>10} {'FP':>5}"]
    for group in sorted(buckets):
        idx = np.array(buckets[group])
        m = metrics.compute(y[idx], probs[idx], threshold=threshold)
        recall = "  n/a" if (m.tp + m.fn) == 0 else f"{m.recall:>8.3f}"
        precision = "     n/a" if (m.tp + m.fp) == 0 else f"{m.precision:>10.3f}"
        lines.append(f"{group:<20} {len(idx):>5} {recall} {precision} {m.fp:>5}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = ck.get("classes", ["normal", "fight"])
    arch = ck.get("config", {}).get("arch", args.arch)

    model = models.build(arch, num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    log.info("loaded %s (%s) from %s", arch, ",".join(classes), args.checkpoint)

    # 1. Choose the threshold on validation.
    val_set = ClipDataset(args.cache_dir, "val", train=False)
    val_probs, val_y = predict(model, val_set, device, args.batch_size, args.num_workers)
    threshold = (
        args.threshold if args.threshold is not None
        else metrics.best_threshold(val_y, val_probs, metric=args.optimise)
    )
    val_metrics = metrics.compute(val_y, val_probs, threshold, classes)

    print("\n" + "=" * 72)
    print(f"VALIDATION  (threshold chosen here: {threshold:.3f}, maximising {args.optimise})")
    print("=" * 72)
    print(val_metrics.report(classes))
    print("\nthreshold sweep:")
    print(metrics.threshold_sweep(val_y, val_probs))

    result = {"threshold": threshold, "val": vars(val_metrics).copy()}

    # 2. Apply it, unchanged, to test.
    try:
        test_set = ClipDataset(args.cache_dir, "test", train=False)
    except ValueError:
        log.warning("no test split found; reporting validation only")
        return result

    test_probs, test_y = predict(model, test_set, device, args.batch_size, args.num_workers)
    test_metrics = metrics.compute(test_y, test_probs, threshold, classes)

    print("\n" + "=" * 72)
    print(f"TEST  (threshold {threshold:.3f} carried over from validation, not retuned)")
    print("=" * 72)
    print(test_metrics.report(classes))

    print("\nper-group breakdown (a model that fails on one camera is a real problem):")
    print(per_group_report(test_y, test_probs, test_set.groups, threshold))

    gap = val_metrics.pr_auc - test_metrics.pr_auc
    if gap > 0.1:
        print(f"\nNOTE: val PR-AUC exceeds test by {gap:.3f}. Some of that is normal "
              "(the threshold was picked on val), but a large gap suggests the val set "
              "is not representative -- check the group split.")

    result["test"] = vars(test_metrics).copy()
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--arch", default="tiny3dcnn", choices=models.ARCHITECTURES,
                        help="only used if the checkpoint does not record it")
    parser.add_argument("--threshold", type=float, default=None,
                        help="fix the threshold instead of choosing it on validation")
    parser.add_argument("--optimise", default="f1", choices=["f1", "precision", "recall"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
