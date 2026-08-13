"""Training and evaluation loops.

Handles the mechanics that are easy to get subtly wrong: mixed precision, gradient
accumulation, resumable checkpointing, and multi-GPU via DDP. Nothing here is specific
to video -- it is the standard supervised-classification loop from
docs/00-primer.md section 5.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# -- distributed helpers ---------------------------------------------------

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise DDP from torchrun's environment variables.

    Set up by `torchrun --nproc_per_node=8`, which is how you use all 8 GPUs on a DGX.
    Returns (rank, world_size, device); falls back to single-device cleanly.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    log.info("distributed: rank %d/%d on %s (%s)", rank, world_size, device, backend)
    return rank, world_size, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def gather_arrays(local: np.ndarray, device: torch.device) -> np.ndarray:
    """Concatenate a per-rank array across all ranks.

    Needed because DistributedSampler gives each rank a different slice of the val set;
    metrics computed on one rank's slice alone are wrong.
    """
    if not is_distributed():
        return local

    tensor = torch.from_numpy(np.ascontiguousarray(local)).to(device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(get_world_size())]
    dist.all_gather(sizes, torch.tensor([tensor.shape[0]], dtype=torch.long, device=device))
    max_size = int(max(int(s.item()) for s in sizes))

    padded = torch.zeros(max_size, *tensor.shape[1:], dtype=tensor.dtype, device=device)
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded)

    return np.concatenate(
        [g[: int(s.item())].cpu().numpy() for g, s in zip(gathered, sizes)], axis=0
    )


# -- config and state ------------------------------------------------------

@dataclass
class TrainConfig:
    arch: str = "tiny3dcnn"
    epochs: int = 10
    batch_size: int = 8
    accum_steps: int = 1
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_steps: int = 0
    amp: bool = True
    amp_dtype: str = "bf16"
    num_workers: int = 4
    seed: int = 0
    class_weights: bool = True
    grad_clip: float = 1.0
    early_stop_patience: int = 0
    monitor: str = "pr_auc"


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0
    lr: float = 0.0


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(config: TrainConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    """bf16 where supported, else fp16, else off.

    bf16 has fp32's exponent range so it needs no loss scaling -- prefer it on A100/H100.
    """
    if not config.amp or device.type != "cuda":
        return False, torch.float32
    if config.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


# -- loops -----------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    config: TrainConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(clips)
            loss = criterion(logits, labels) / config.accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)   # unscale before clipping, or the clip
                                             # threshold is applied to scaled gradients
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * config.accum_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (mean loss, positive-class probabilities, labels)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    probs, ys = [], []

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(clips)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        probs.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
        ys.append(labels.cpu().numpy())

    local_probs = np.concatenate(probs) if probs else np.array([])
    local_ys = np.concatenate(ys) if ys else np.array([])
    return (
        total_loss / max(n_batches, 1),
        gather_arrays(local_probs, device),
        gather_arrays(local_ys, device),
    )


# -- checkpointing ---------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int,
                    best: float, config: TrainConfig, classes: list[str],
                    history: list[EpochResult]) -> None:
    if not is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best": best,
        "config": asdict(config),
        "classes": classes,
        "history": [asdict(h) for h in history],
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None,
                    device: torch.device | None = None) -> tuple[int, float, list]:
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    return ck.get("epoch", -1) + 1, ck.get("best", 0.0), ck.get("history", [])


def write_history(path: Path, history: list[EpochResult]) -> None:
    if is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(h) for h in history], indent=2))


def format_epoch(result: EpochResult, monitor: str) -> str:
    watched = result.metrics.get(monitor, float("nan"))
    return (
        f"epoch {result.epoch:>3} | {result.seconds:>5.1f}s | lr {result.lr:.2e} | "
        f"train {result.train_loss:.4f} | val {result.val_loss:.4f} | "
        f"acc {result.metrics.get('accuracy', float('nan')):.4f} | "
        f"{monitor} {watched:.4f}"
    )


def overfitting_warning(history: list[EpochResult], patience: int = 3) -> str | None:
    """Train loss falling while val loss rises is the classic signature.

    Worth surfacing explicitly: it is the failure that hides, and it is easy to keep
    celebrating a falling training loss while the model quietly gets worse.
    """
    if len(history) < patience + 1:
        return None
    recent = history[-(patience + 1):]
    val_rising = all(b.val_loss > a.val_loss for a, b in zip(recent, recent[1:]))
    train_falling = all(b.train_loss < a.train_loss for a, b in zip(recent, recent[1:]))
    if val_rising and train_falling:
        return (
            f"val loss has risen for {patience} straight epochs while train loss fell "
            "-- this is overfitting. Use the best checkpoint, not the last one, and "
            "consider more augmentation or fewer epochs."
        )
    return None
