"""Train several architectures on the same data and print one comparison table.

**This is the most useful thing in the repo for a research conversation.** A single
accuracy number from one model tells you almost nothing. The comparison tells you what
is actually happening in your data:

    motion_baseline   not learned. Pure motion energy.
    frame_average     sees every frame but NOT their order -- deliberately motion-blind.
    tiny3dcnn         a real, if small, spatio-temporal model.

How to read the result:

  * Your model barely beats `motion_baseline`  -> the task is being solved by "how much
    movement is there", and you do not need a neural network for that. Look at your data.
  * `frame_average` is close to your model      -> the signal is in APPEARANCE, not
    motion. Suspect leakage: the model may be recognising the scene, not the action.
    Check your group split.
  * Big gap over both                           -> temporal modelling is genuinely
    earning its complexity. This is what you want to show.

Bring this table to the meeting rather than a single number.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from cctv_sentinel.training import metrics, models
from cctv_sentinel.training.dataset import ClipDataset
from cctv_sentinel.training.evaluate import predict
from cctv_sentinel.training.train import main as train_main

log = logging.getLogger(__name__)


def run_one(arch: str, cache_dir: Path, out_root: Path, epochs: int, batch_size: int,
            lr: float, num_workers: int) -> dict:
    out_dir = out_root / arch
    started = time.time()

    argv = [
        "--cache-dir", str(cache_dir), "--out-dir", str(out_dir), "--arch", arch,
        "--epochs", str(epochs), "--batch-size", str(batch_size),
        "--head-lr", str(lr), "--backbone-lr", str(lr),
        "--num-workers", str(num_workers), "--monitor", "pr_auc",
    ]
    if arch in ("tiny3dcnn", "frame_average", "motion_baseline"):
        argv.append("--no-pretrained")

    train_main(argv)
    elapsed = time.time() - started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = models.build(arch, num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    val_set = ClipDataset(cache_dir, "val", train=False)
    val_probs, val_y = predict(model, val_set, device, batch_size, num_workers)
    threshold = metrics.best_threshold(val_y, val_probs, metric="f1")

    test_set = ClipDataset(cache_dir, "test", train=False)
    test_probs, test_y = predict(model, test_set, device, batch_size, num_workers)
    m = metrics.compute(test_y, test_probs, threshold, classes)

    return {
        "arch": arch,
        "params_m": models.count_parameters(model) / 1e6,
        "train_seconds": elapsed,
        "threshold": threshold,
        "test_pr_auc": m.pr_auc,
        "test_roc_auc": m.roc_auc,
        "test_f1": m.f1,
        "test_recall": m.recall,
        "test_precision": m.precision,
        "test_accuracy": m.accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare architectures on the same data")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs/compare"))
    parser.add_argument("--archs", nargs="+",
                        default=["motion_baseline", "frame_average", "tiny3dcnn"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")

    results = []
    for arch in args.archs:
        print(f"\n{'=' * 78}\ntraining {arch}\n{'=' * 78}")
        try:
            results.append(run_one(arch, args.cache_dir, args.out_root, args.epochs,
                                   args.batch_size, args.lr, args.num_workers))
        except Exception as exc:
            print(f"  {arch} failed: {exc}")

    print(f"\n\n{'=' * 78}\nCOMPARISON (test split, threshold chosen on validation)\n{'=' * 78}")
    header = (f"{'model':<18}{'params':>9}{'train':>8}{'PR-AUC':>9}{'ROC-AUC':>9}"
              f"{'F1':>8}{'recall':>8}{'prec':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['arch']:<18}{r['params_m']:>8.2f}M{r['train_seconds']:>7.0f}s"
              f"{r['test_pr_auc']:>9.3f}{r['test_roc_auc']:>9.3f}{r['test_f1']:>8.3f}"
              f"{r['test_recall']:>8.3f}{r['test_precision']:>7.3f}")

    by_arch = {r["arch"]: r for r in results}
    print()
    if "motion_baseline" in by_arch and len(results) > 1:
        base = by_arch["motion_baseline"]["test_pr_auc"]
        best = max((r for r in results if r["arch"] != "motion_baseline"),
                   key=lambda r: r["test_pr_auc"])
        gain = best["test_pr_auc"] - base
        if gain < 0.05:
            print(f"WARNING: {best['arch']} beats plain motion energy by only {gain:+.3f} PR-AUC.")
            print("The task may be solvable by movement quantity alone -- inspect the data")
            print("before investing in a bigger model.")
        else:
            print(f"{best['arch']} beats motion energy by {gain:+.3f} PR-AUC -- learning is "
                  "adding real signal.")

    if "frame_average" in by_arch:
        fa = by_arch["frame_average"]["test_pr_auc"]
        temporal = [r for r in results if r["arch"] not in ("frame_average", "motion_baseline")]
        if temporal:
            best_t = max(temporal, key=lambda r: r["test_pr_auc"])
            gap = best_t["test_pr_auc"] - fa
            print(f"temporal gain over the motion-blind frame_average: {gap:+.3f} PR-AUC")
            if gap < 0.03:
                print("  -> frame order barely matters here. Either the task is solvable from")
                print("     appearance alone, or the split is leaking scene identity.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_root / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
