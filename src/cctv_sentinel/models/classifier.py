"""Clip classifier backends.

Three backends behind one interface:

    stub   motion-energy heuristic. No GPU, no weights, no training. Lets you build
           and test the entire pipeline before a model exists -- which is the whole
           point of Phase 1 in docs/05. It is NOT a detector; it will fire on anyone
           walking briskly. Do not ship it.
    torch  a fine-tuned video backbone (VideoMAE / X3D). See notebooks/train_colab.ipynb.
    onnx   the same model exported for edge inference.

Everything returns the same thing: {class_name: probability}.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from ..config import ModelConfig

log = logging.getLogger(__name__)

# Kinetics normalisation -- matches how the backbones in notebooks/train_colab.ipynb
# were pretrained. Change here and in the notebook together, or your accuracy silently
# drops by a few points.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Classifier(Protocol):
    classes: list[str]

    def predict(self, clip: np.ndarray) -> dict[str, float]:
        """clip: (T, H, W, 3) uint8 BGR -> {class: probability}"""
        ...


def build(config: ModelConfig) -> Classifier:
    if config.backend == "stub":
        return StubClassifier(config)
    if config.backend == "torch":
        return TorchClassifier(config)
    if config.backend == "onnx":
        return OnnxClassifier(config)
    raise ValueError(f"unknown model backend: {config.backend!r}")


def preprocess(clip: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W, 3) uint8 BGR -> (1, 3, T, size, size) float32 normalised RGB."""
    frames = [
        cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in clip
    ]
    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = arr[..., ::-1]  # BGR -> RGB
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(3, 0, 1, 2)  # (T,H,W,C) -> (C,T,H,W)
    return np.ascontiguousarray(arr[None])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class StubClassifier:
    """Placeholder scoring on frame-to-frame motion energy.

    Rationale: it responds to the same input signal a real violence model responds
    to (sudden, large, high-frequency movement), so the downstream smoothing,
    hysteresis and cooldown behave realistically while you develop them. It has no
    idea what a fight is.

    Because it cannot discriminate between classes, it reports its whole score on one
    class (`model.stub_class`). Spreading the score across N classes would cap each at
    1/N, and the event thresholds in configs/default.yaml could never be crossed.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.classes = config.classes
        self._size = 64
        candidates = [c for c in self.classes if c != "normal"]
        if not candidates:
            raise ValueError("model.classes needs at least one non-'normal' class")
        self.stub_class = config.stub_class or candidates[0]
        if self.stub_class not in self.classes:
            raise ValueError(f"model.stub_class {self.stub_class!r} is not in model.classes")

    def predict(self, clip: np.ndarray) -> dict[str, float]:
        small = np.stack(
            [
                cv2.cvtColor(
                    cv2.resize(f, (self._size, self._size), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
                for f in clip
            ]
        ).astype(np.float32)

        diffs = np.abs(np.diff(small, axis=0))
        energy = float(diffs.mean()) / 255.0
        # Jerk: a fight is not just movement, it is movement that changes abruptly.
        jerk = float(np.abs(np.diff(diffs.mean(axis=(1, 2)))).mean()) / 255.0

        score = min(1.0, energy * 6.0 + jerk * 20.0)
        scores = {name: 0.0 for name in self.classes}
        scores[self.stub_class] = score
        if "normal" in scores:
            scores["normal"] = 1.0 - score
        return scores


class TorchClassifier:
    """TorchScript module produced by notebooks/train_colab.ipynb."""

    def __init__(self, config: ModelConfig) -> None:
        import torch  # imported lazily: the stub path must work without torch

        self.classes = config.classes
        self.input_size = config.input_size
        self._torch = torch

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not config.weights:
            raise ValueError("model.weights is required for the torch backend")
        self.model = torch.jit.load(config.weights, map_location=self.device)
        self.model.eval()
        log.info("loaded torchscript model %s on %s", config.weights, self.device)

    def predict(self, clip: np.ndarray) -> dict[str, float]:
        batch = preprocess(clip, self.input_size)
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(batch).to(self.device)
            logits = self.model(tensor).float().cpu().numpy()[0]
        return dict(zip(self.classes, _softmax(logits).tolist()))


class OnnxClassifier:
    """ONNX Runtime backend -- the edge deployment path (CUDA EP or CPU)."""

    def __init__(self, config: ModelConfig) -> None:
        import onnxruntime as ort

        self.classes = config.classes
        self.input_size = config.input_size
        if not config.weights:
            raise ValueError("model.weights is required for the onnx backend")

        providers = ["CPUExecutionProvider"]
        if config.device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(config.weights, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        log.info("loaded onnx model %s with %s", config.weights, providers)

    def predict(self, clip: np.ndarray) -> dict[str, float]:
        batch = preprocess(clip, self.input_size)
        logits = self.session.run(None, {self._input_name: batch})[0][0]
        return dict(zip(self.classes, _softmax(np.asarray(logits)).tolist()))
