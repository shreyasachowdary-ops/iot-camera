"""Training entry point.

Single GPU / CPU:
    python -m cctv_sentinel.training.train --cache-dir data/cache --arch tiny3dcnn

All 8 GPUs on a DGX:
    torchrun --nproc_per_node=8 -m cctv_sentinel.training.train \\
        --cache-dir data/cache --arch videomae --batch-size 8

Resume after an interruption (same command plus --resume) -- checkpoints are written
every epoch.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from . import engine, metrics, models
from .dataset import ClipDataset
from .engine import EpochResult, TrainConfig

log = logging.getLogger(__name__)


def build_loaders(cache_dir: Path, config: TrainConfig) -> tuple[DataLoader, DataLoader, ClipDataset]:
    train_set = ClipDataset(cache_dir, "train")
    val_set = ClipDataset(cache_dir, "val", train=False)

    train_sampler = DistributedSampler(train_set, shuffle=True) if engine.is_distributed() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if engine.is_distributed() else None

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, train_set


def run(args: argparse.Namespace) -> dict:
    rank, world_size, device = engine.setup_distributed()
    config = TrainConfig(
        arch=args.arch, epochs=args.epochs, batch_size=args.batch_size,
        accum_steps=args.accum_steps, head_lr=args.head_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        amp=not args.no_amp, amp_dtype=args.amp_dtype, num_workers=args.num_workers,
        seed=args.seed, class_weights=not args.no_class_weights,
        early_stop_patience=args.early_stop_patience, monitor=args.monitor,
    )
    engine.set_seed(config.seed + rank)

    train_loader, val_loader, train_set = build_loaders(args.cache_dir, config)
    classes = train_set.classes

    if engine.is_main():
        counts = train_set.class_counts()
        total = sum(counts.values())
        log.info("classes: %s", classes)
        log.info("train %d clips | %s", total,
                 ", ".join(f"{k}={v} ({v / max(total,1):.1%})" for k, v in counts.items()))
        log.info("val   %d clips", len(val_loader.dataset))
        # Class balance decides whether accuracy means anything at all.
        share = max(counts.values()) / max(total, 1)
        if share > 0.7:
            log.warning(
                "the majority class is %.0f%% of the training set -- accuracy will be "
                "misleading. Judge this run on PR-AUC and recall, not accuracy.", share * 100
            )

    model = models.build(
        config.arch, num_classes=len(classes), pretrained=not args.no_pretrained,
        checkpoint=args.hf_checkpoint,
    ).to(device)
    if engine.is_main():
        log.info("%s: %.2fM trainable parameters", config.arch,
                 models.count_parameters(model) / 1e6)

    if engine.is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    weights = train_set.class_weights().to(device) if config.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        models.param_groups(raw_model, config.head_lr, config.backbone_lr),
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // config.accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * steps_per_epoch)
    )

    amp_enabled, amp_dtype = engine.resolve_amp(config, device)
    scaler = torch.amp.GradScaler(device.type) if (amp_enabled and amp_dtype == torch.float16) else None
    if engine.is_main() and amp_enabled:
        log.info("mixed precision: %s", amp_dtype)

    out_dir = args.out_dir
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    start_epoch, best_score, history_raw = 0, 0.0, []
    if args.resume and last_path.exists():
        start_epoch, best_score, history_raw = engine.load_checkpoint(
            last_path, model, optimizer, scheduler, device
        )
        if engine.is_main():
            log.info("resumed at epoch %d (best %s %.4f)", start_epoch, config.monitor, best_score)

    history = [EpochResult(**h) for h in history_raw]
    best_val_loss = min((h.val_loss for h in history), default=float("inf"))
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)  # or every epoch sees the same shuffle

        started = time.time()
        train_loss = engine.train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config,
            scheduler, amp_enabled, amp_dtype,
        )
        val_loss, probs, ys = engine.evaluate(
            model, val_loader, criterion, device, amp_enabled, amp_dtype
        )

        m = metrics.compute(ys, probs, threshold=0.5, classes=classes)
        result = EpochResult(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            metrics={"accuracy": m.accuracy, "balanced_accuracy": m.balanced_accuracy,
                     "precision": m.precision, "recall": m.recall, "f1": m.f1,
                     "roc_auc": m.roc_auc, "pr_auc": m.pr_auc},
            seconds=time.time() - started, lr=optimizer.param_groups[0]["lr"],
        )
        history.append(result)

        if engine.is_main():
            log.info(format_or_nan(result, config.monitor))
            engine.save_checkpoint(last_path, model, optimizer, scheduler, epoch,
                                   best_score, config, classes, history)
            engine.write_history(out_dir / "history.json", history)

            score = result.metrics.get(config.monitor, 0.0)
            score = 0.0 if score != score else score  # NaN -> 0
            # Tie-break on val loss. PR-AUC and ROC-AUC saturate at 1.0 on easy data,
            # and with a strict > comparison the first epoch to hit 1.0 would be kept
            # forever -- even though it may be far from calibrated. Ranking is perfect
            # at that point; val loss is what still distinguishes the checkpoints.
            improved = score > best_score or (score == best_score and val_loss < best_val_loss)
            if improved:
                best_score = score
                best_val_loss = val_loss
                epochs_without_improvement = 0
                engine.save_checkpoint(best_path, model, optimizer, scheduler, epoch,
                                       best_score, config, classes, history)
                log.info("  new best %s: %.4f -> %s", config.monitor, best_score, best_path)
            else:
                epochs_without_improvement += 1

            warning = engine.overfitting_warning(history)
            if warning:
                log.warning("  %s", warning)

        if config.early_stop_patience and epochs_without_improvement >= config.early_stop_patience:
            if engine.is_main():
                log.info("early stopping: no improvement in %d epochs", config.early_stop_patience)
            break

    summary: dict = {}
    if engine.is_main():
        log.info("training complete. best %s = %.4f", config.monitor, best_score)
        log.info("best checkpoint: %s", best_path)
        summary = {"best_score": best_score, "monitor": config.monitor,
                   "epochs": len(history), "classes": classes,
                   "best_checkpoint": str(best_path)}

    engine.cleanup_distributed()
    return summary


def format_or_nan(result: EpochResult, monitor: str) -> str:
    return engine.format_epoch(result, monitor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a video classifier")
    parser.add_argument("--cache-dir", type=Path, required=True, help="output of prepare.py")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--arch", default="tiny3dcnn", choices=models.ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="gradient accumulation: effective batch = batch_size * this")
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--hf-checkpoint", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--monitor", default="pr_auc",
                        choices=["pr_auc", "roc_auc", "f1", "accuracy", "balanced_accuracy"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
