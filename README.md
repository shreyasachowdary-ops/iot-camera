# CCTV Sentinel

Real-time anomaly detection on live CCTV streams (fights, falls/collapse, weapons,
crowd panic) that fires a webhook to a control room within a few seconds — **without
storing the video**.

This repo is both a working skeleton and the plan. If you are reading it for the
first time, read the docs in order:

| Doc | Answers |
|---|---|
| [docs/01-architecture.md](docs/01-architecture.md) | Where the software actually runs. (Spoiler: not inside the camera.) |
| [docs/02-streaming-and-storage.md](docs/02-streaming-and-storage.md) | **"We can't store everything and feed it to AI"** — the ring-buffer answer. |
| [docs/03-data-and-training.md](docs/03-data-and-training.md) | Datasets, labels, model choices, the exact Colab recipe. |
| [docs/04-tradeoffs.md](docs/04-tradeoffs.md) | Pros/cons, cost, false-alarm reality, privacy/DPDP. |
| [docs/05-roadmap.md](docs/05-roadmap.md) | What to build in what order, week by week. |

## The short version

```
IP camera ──RTSP──> edge box ──> motion gate ──> person gate ──> clip model ──> smoother ──> webhook
                       │                                                            │
                   ring buffer (last 30s, RAM)  ────── on alert: dump clip ──────────┘
```

Four ideas carry the whole design:

1. **Nothing is stored by default.** Frames live in a fixed-size RAM ring buffer and
   are overwritten. Only when an alert fires do you write the ~40 s around the event.
   A 20-camera site goes from ~860 GB/day to ~2 GB/day.
2. **Cheap gates before the expensive model.** ~95% of CCTV footage is an empty
   corridor. Motion diff (µs) → person detector (ms) → video classifier (100 ms).
   You only pay for the last stage on the few percent of time that matters.
3. **You classify short clips, not "the video".** A sliding 2-second window every
   1 second. Statelessly scored, then temporally smoothed so one noisy frame can't
   page a security guard.
4. **Colab is for training only.** It cannot host a 24/7 RTSP consumer. Train there,
   export weights, run inference on an edge box or a GPU VM.

## Quick start

```bash
pip install -r requirements.txt

# Run the whole loop against a webcam / file / RTSP url with a stub model
# (no GPU, no weights needed — proves the plumbing end to end)
python -m cctv_sentinel.cli run --config configs/default.yaml --source 0

# Point at a real camera
python -m cctv_sentinel.cli run --config configs/default.yaml \
    --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"

# See what the webhook payload looks like without sending it
python -m cctv_sentinel.cli demo-alert
```

The default config uses `model.backend: stub`, a motion-energy heuristic that stands
in for the trained model. Everything else — gating, buffering, smoothing, clip
export, webhook retry — is real. Swap `backend: torch` once you have weights from
[the Colab notebook](notebooks/train_colab.ipynb).

## Layout

```
src/cctv_sentinel/
  ingest/      RTSP reader (drops frames instead of queueing), ring buffer
  pipeline/    motion gate, person gate, clip sampler
  models/      classifier backends (stub / torch / onnx)
  detect/      score smoothing, hysteresis, event state machine
  alerting/    dedupe + cooldown dispatcher, signed webhook with retries
  storage/     writes clips only for alerts
  service/     the runner that wires it together
notebooks/     Colab training notebook
configs/       YAML config
```

## Status

Skeleton + plan. The gates, buffering, event logic, and alerting are implemented and
tested. The trained model is not — that's [Phase 2](docs/05-roadmap.md).
