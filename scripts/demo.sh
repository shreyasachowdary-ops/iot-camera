#!/usr/bin/env bash
# End-to-end demo: synthetic data -> prepare -> compare -> train -> evaluate -> export -> live inference.
# Runs on a CPU in ~3 minutes. No GPU, no downloads, no real data required.
#
# The dataset is SYNTHETIC: "fight" is fast erratic motion, "normal" is slow walking.
# That is a far easier problem than real footage. This proves the pipeline runs; it
# does not prove the approach works. Say so when demoing it.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

SIZE=112          # small, so it trains on a CPU. Use 224 with a real GPU.
EPOCHS=${EPOCHS:-6}

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "1/6  generating synthetic footage (8 cameras x 24 clips)"
python scripts/make_synthetic_dataset.py --out-dir data/raw --cameras 8 --per-class-per-camera 12

step "2/6  preparing: decode once into a memmap, split by camera (not randomly)"
python -m cctv_sentinel.training.prepare \
    --raw-dir data/raw --out-dir data/cache \
    --classes normal fight --num-frames 16 --size "$SIZE"

step "3/6  baselines: is a neural network even earning its keep?"
python scripts/compare_baselines.py \
    --cache-dir data/cache --out-root runs/compare --epochs "$EPOCHS" --num-workers 2

step "4/6  training tiny3dcnn"
python -m cctv_sentinel.training.train \
    --cache-dir data/cache --out-dir runs/demo --arch tiny3dcnn \
    --epochs "$EPOCHS" --batch-size 8 --head-lr 3e-3 --backbone-lr 3e-3 \
    --num-workers 2 --monitor pr_auc

step "5/6  evaluating: threshold picked on val, applied unchanged to test"
python -m cctv_sentinel.training.evaluate \
    --checkpoint runs/demo/best.pt --cache-dir data/cache --out runs/demo/eval.json

step "6/6  exporting to TorchScript and running live inference"
python -m cctv_sentinel.training.export \
    --checkpoint runs/demo/best.pt --out-dir weights --size "$SIZE" --no-onnx

python scripts/make_demo_video.py --out data/demo_incident.mp4
python -m cctv_sentinel.cli run --config configs/trained.yaml \
    --source data/demo_incident.mp4 --duration 60

printf '\n\033[1m=== done ===\033[0m\n'
echo "  comparison table   runs/compare/comparison.json"
echo "  metrics            runs/demo/eval.json"
echo "  exported model     weights/model.torchscript.pt"
echo "  alert clip         var/clips/"
echo
echo "Reminder: synthetic data. The pipeline is real; the numbers are not."
