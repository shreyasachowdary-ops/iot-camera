# CCTV Sentinel

Research pipeline for training video models to detect anomalies in CCTV footage —
fights, falls, collapses — plus a working inference service to demo them on live video.

**Start here if you're new to ML:** [docs/00-primer.md](docs/00-primer.md) explains
every concept from zero — what a model is, how training works, frames and FPS, epochs
and batches, overfitting, metrics. Written to be read before a meeting with an ML
engineer.

---

## The five-minute demo

No GPU, no data, no downloads. ~3 minutes on a laptop.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
export PYTHONPATH=src

bash scripts/demo.sh
```

Generates synthetic footage → prepares it → trains three models and compares them →
evaluates → exports → runs live detection on a video stream and fires an alert.

The data is synthetic and deliberately easy. **It proves the pipeline works, not that
the approach works.** Say that when you demo it.

---

## Documentation

| Doc | What it covers |
|---|---|
| **[00-primer.md](docs/00-primer.md)** | **Every concept from zero.** Models, tensors, FPS, epochs, overfitting, transfer learning, architectures, metrics, glossary |
| [07-training-guide.md](docs/07-training-guide.md) | Hands-on: train on your own footage, on a DGX or one cloud GPU |
| [08-meeting-cheatsheet.md](docs/08-meeting-cheatsheet.md) | One page: terms, numbers, questions to ask |
| [01-architecture.md](docs/01-architecture.md) | Where inference runs (not inside the camera) |
| [02-streaming-and-storage.md](docs/02-streaming-and-storage.md) | Processing live video without storing it |
| [03-data-and-training.md](docs/03-data-and-training.md) | Public datasets, model choices, Colab |
| [04-tradeoffs.md](docs/04-tradeoffs.md) | Pros/cons, false-alarm economics, privacy/DPDP |
| [05-roadmap.md](docs/05-roadmap.md) | Phased plan |

---

## Training on your own footage

```
data/raw/
  normal/  cam01_2024-05-03_0001.mp4  ...
  fight/   cam01_2024-05-03_0007.mp4  ...
```

Name files `<camera>_<anything>.mp4`. The prefix is the **group**, and all clips from
one group land on the same side of the train/val split — otherwise the model learns to
recognise the corridor instead of the action and your scores are fiction.

```bash
# 1. decode once into a memmap, split by camera (not randomly)
python -m cctv_sentinel.training.prepare \
    --raw-dir data/raw --out-dir data/cache --classes normal fight

# 2a. all 8 GPUs on a DGX
torchrun --nproc_per_node=8 -m cctv_sentinel.training.train \
    --cache-dir data/cache --out-dir runs/videomae --arch videomae \
    --epochs 15 --batch-size 8 --amp-dtype bf16

# 2b. or one cloud GPU
python -m cctv_sentinel.training.train \
    --cache-dir data/cache --out-dir runs/videomae --arch videomae \
    --epochs 15 --batch-size 8 --accum-steps 4

# 3. threshold picked on val, applied unchanged to test
python -m cctv_sentinel.training.evaluate \
    --checkpoint runs/videomae/best.pt --cache-dir data/cache

# 4. export for inference
python -m cctv_sentinel.training.export \
    --checkpoint runs/videomae/best.pt --out-dir weights
```

Full walkthrough with troubleshooting: [docs/07-training-guide.md](docs/07-training-guide.md).

### Always run the baselines

```bash
python scripts/compare_baselines.py --cache-dir data/cache
```

**Bring this table to a meeting instead of a single accuracy number.** Real output from
the synthetic demo:

```
model                params   train   PR-AUC  ROC-AUC      F1  recall   prec
----------------------------------------------------------------------------
motion_baseline       0.00M      9s    1.000    1.000   1.000   1.000  1.000
frame_average         0.09M     31s    0.397    0.292   0.667   0.917  0.524
tiny3dcnn             0.29M     15s    1.000    1.000   1.000   1.000  1.000

WARNING: tiny3dcnn beats plain motion energy by only +0.000 PR-AUC.
```

That warning is the tool working: on this synthetic data "fight" really is just fast
motion, so a neural network adds nothing — and it says so rather than letting you report
1.000 as a success.

| Result on real data | Meaning |
|---|---|
| Model ≈ `motion_baseline` | Task is "how much movement". You don't need a network. |
| Model ≈ `frame_average` | Signal is appearance, not motion. **Suspect leakage.** |
| Clear gap over both | Temporal modelling is earning its complexity. |

---

## Live inference

The trained model can drive a real-time service on an RTSP stream, with no footage
stored: frames live in a fixed-size RAM ring buffer and only ~40 s around an alert is
written to disk.

```bash
python -m cctv_sentinel.cli run --config configs/trained.yaml \
    --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

See [docs/01](docs/01-architecture.md) and [docs/02](docs/02-streaming-and-storage.md).

---

## Layout

```
src/cctv_sentinel/
  training/    prepare -> train -> evaluate -> export  (the research pipeline)
    prepare.py    decode-once cache + group-aware splitting
    dataset.py    augmentation shaped toward night/IR and low-bitrate footage
    models.py     motion_baseline | frame_average | tiny3dcnn | r3d_18 | videomae
    engine.py     AMP, gradient accumulation, DDP, resumable checkpoints
    metrics.py    ROC-AUC, PR-AUC, threshold sweep -- pure numpy, readable
  ingest/      RTSP reader, ring buffer          } inference
  pipeline/    motion gate, clip sampler         } service
  models/      classifier backends               }
  detect/      event smoothing and hysteresis    }
  alerting/    signed webhook, dedupe, outbox    }
scripts/       demo.sh, compare_baselines.py, synthetic data generators
notebooks/     Colab training notebook
```

## Tests

```bash
export PYTHONPATH=src && python -m pytest tests/ -q     # 61 tests
```

Covers metrics against hand-computed values, group-split correctness (no group in two
splits), model shape contracts, and the inference event logic.
