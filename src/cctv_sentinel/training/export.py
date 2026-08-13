"""Export a trained checkpoint to TorchScript and/or ONNX.

A `.pt` checkpoint is a Python object graph -- it needs your source code to load. An
exported model is self-contained, which is what you want for anything that is not this
repo.

Both exports take `(B, C, T, H, W)`, matching `preprocess()` in
`models/classifier.py`, so the inference runner can load either without transposes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from . import models

log = logging.getLogger(__name__)


def load_for_export(checkpoint: Path, arch: str | None = None) -> tuple[torch.nn.Module, list[str], dict]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = ck.get("classes", ["normal", "fight"])
    config = ck.get("config", {})
    resolved = arch or config.get("arch")
    if not resolved:
        raise ValueError("checkpoint has no architecture recorded; pass --arch")

    model = models.build(resolved, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, classes, config


def export(checkpoint: Path, out_dir: Path, arch: str | None = None,
           num_frames: int = 16, size: int = 224,
           torchscript: bool = True, onnx: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, classes, config = load_for_export(checkpoint, arch)
    example = torch.randn(1, 3, num_frames, size, size)

    with torch.no_grad():
        reference = model(example)
    log.info("model: %s -> %s, classes %s", tuple(example.shape), tuple(reference.shape), classes)

    written: dict[str, Path] = {}

    if torchscript:
        path = out_dir / "model.torchscript.pt"
        traced = torch.jit.trace(model, example)
        torch.jit.save(traced, path)
        # Verify the trace rather than assuming it: tracing silently bakes in any
        # data-dependent control flow it happened to take.
        with torch.no_grad():
            error = (torch.jit.load(path)(example) - reference).abs().max().item()
        if error > 1e-4:
            raise RuntimeError(f"torchscript output diverges from source model by {error:.2e}")
        log.info("torchscript -> %s (max diff %.2e)", path, error)
        written["torchscript"] = path

    if onnx:
        path = out_dir / "model.onnx"
        try:
            torch.onnx.export(
                model, example, str(path),
                input_names=["clip"], output_names=["logits"],
                dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            log.info("onnx -> %s", path)
            written["onnx"] = path
        except Exception as exc:
            log.warning("onnx export failed (%s); torchscript is still usable", exc)

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n")
    log.info("classes -> %s (ORDER MATTERS -- must match model.classes in the config)",
             out_dir / "classes.txt")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a checkpoint for inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-torchscript", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    written = export(
        args.checkpoint, args.out_dir, args.arch, args.num_frames, args.size,
        torchscript=not args.no_torchscript, onnx=not args.no_onnx,
    )

    print("\nexported:")
    for kind, path in written.items():
        print(f"  {kind:<12} {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("""
To use it for inference, in configs/default.yaml:

  model:
    backend: torch                       # or onnx
    weights: weights/model.torchscript.pt
    classes: [normal, fight]             # MUST match classes.txt, in the same order
    input_size: %d
""" % args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
