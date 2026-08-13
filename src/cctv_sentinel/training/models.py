"""Model zoo, smallest first.

All models take (B, C, T, H, W) and return raw logits (B, num_classes). Keeping one
convention across every backbone means the training loop, export and inference never
need to know which one is loaded.

    motion_baseline   ~0 params    a non-learned motion-energy score. The dumb baseline.
    frame_average     ~1M          2D CNN per frame, averaged. MOTION-BLIND on purpose.
    tiny3dcnn         ~800k        small 3D CNN, trains from scratch on CPU. For demos.
    r3d_18            33M          torchvision, Kinetics-pretrained. Solid baseline.
    videomae          87M          HuggingFace, self-supervised. Best on small data.

Always run the two baselines. If an 87M-parameter transformer barely beats motion
energy, that is the single most important thing to learn about your dataset, and you
want to learn it in week one rather than in the write-up. `frame_average` is the sharper
of the two: it sees every frame but cannot see ORDER, so the gap between it and a real
video model is exactly the value of temporal modelling on your data.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class MotionBaseline(nn.Module):
    """Non-learned reference: mean absolute difference between consecutive frames.

    Has one trainable scalar pair purely so it can be calibrated into logits; it cannot
    learn to recognise anything. If your real model does not clearly beat this, the
    problem is the data, not the architecture.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=(1, 2, 3, 4))
        score = energy * self.scale + self.bias
        logits = torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=score.dtype)
        logits[:, -1] = score
        return logits


class FrameAverage(nn.Module):
    """2D CNN applied per frame, then averaged over time.

    Deliberately motion-blind: averaging destroys frame order, so it cannot distinguish
    someone falling from someone standing up. That is the point -- it isolates how much
    of your signal is temporal versus merely appearance.
    """

    def __init__(self, num_classes: int = 2, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * t, c, h, w)
        f = self.features(x).flatten(1).reshape(b, t, -1).mean(dim=1)
        return self.head(f)


class Tiny3DCNN(nn.Module):
    """Small 3D CNN that trains from scratch on a CPU in minutes.

    Not competitive with a pretrained backbone -- it exists so the whole pipeline can be
    run, demoed and debugged without a GPU or a model download.
    """

    def __init__(self, num_classes: int = 2, width: int = 16, dropout: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int, stride: tuple[int, int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, width, (1, 2, 2)),
            block(width, width * 2, (2, 2, 2)),
            block(width * 2, width * 4, (2, 2, 2)),
            block(width * 4, width * 8, (2, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x).flatten(1)))


class VideoMAEWrapper(nn.Module):
    """HuggingFace VideoMAE. Takes (B,C,T,H,W); VideoMAE wants (B,T,C,H,W)."""

    def __init__(self, num_classes: int, checkpoint: str, dropout: float = 0.0) -> None:
        super().__init__()
        from transformers import VideoMAEForVideoClassification

        self.net = VideoMAEForVideoClassification.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True,
            hidden_dropout_prob=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=x.permute(0, 2, 1, 3, 4)).logits


class R3D18Wrapper(nn.Module):
    """torchvision r3d_18, Kinetics-400 pretrained. Native (B,C,T,H,W)."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r3d_18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ARCHITECTURES = ["motion_baseline", "frame_average", "tiny3dcnn", "r3d_18", "videomae"]


def build(
    arch: str,
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    dropout: float = 0.3,
) -> nn.Module:
    if arch == "motion_baseline":
        return MotionBaseline(num_classes)
    if arch == "frame_average":
        return FrameAverage(num_classes)
    if arch == "tiny3dcnn":
        return Tiny3DCNN(num_classes, dropout=dropout)
    if arch == "r3d_18":
        return R3D18Wrapper(num_classes, pretrained=pretrained)
    if arch == "videomae":
        return VideoMAEWrapper(num_classes, checkpoint=checkpoint)
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHITECTURES}")


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list[dict]:
    """Discriminative learning rates: fast for the new head, gentle for the backbone.

    The head is randomly initialised and knows nothing, so it should move fast. The
    backbone already encodes what motion looks like, and large updates would destroy
    that -- catastrophic forgetting. See docs/00-primer.md section 8.
    """
    head_keys = ("head", "classifier", "fc")
    head, backbone = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if any(k in name for k in head_keys) else backbone).append(param)

    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
