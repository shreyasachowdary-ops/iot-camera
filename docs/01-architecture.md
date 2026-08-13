# 01 — Architecture: where does this software actually run?

You said "we need to embed the software into the CCTV. I don't know." That instinct
is the single biggest fork in the project, so let's settle it first.

## You almost certainly cannot put your model inside the camera

A typical IP camera (Hikvision, Dahua, CP Plus, and the OEM rebrands that make up
most of the Indian market) is a locked-down ARM SoC running a vendor firmware. There
is no package manager, no Python, ~64–256 MB RAM, and no way to sideload code without
voiding warranty or bricking it. "Embedding software into the CCTV" is not a thing
you can do to cameras that are already installed.

There are three real deployment shapes:

### A. Edge box per site — **recommended**

One small computer per building, pulling RTSP from every camera on the LAN.

- NVIDIA Jetson Orin Nano 8 GB (~₹25–35k) — 4–8 cameras of real analytics, hardware
  H.264 decode via NVDEC, runs PyTorch/TensorRT directly.
- Or an Intel N100/N305 mini PC (~₹18–25k) with OpenVINO + a Coral TPU. Cheaper,
  weaker, fine for pose-based fall detection but strained by video transformers.

**Why this wins:** video never leaves the site (huge for the privacy conversation and
for Indian bandwidth), latency is ~1–2 s, and it survives an internet outage — it can
queue alerts and flush when the link returns. You install one box, not 40 cameras.

### B. Central GPU server / cloud VM

All streams pushed to one machine with a T4/L4/A10.

- Good when cameras are already on a VMS (Milestone, Genetec, CP Plus IVMS) with an
  RTSP re-stream, or when sites are too small to justify a box each.
- Bad because you now ship every camera's video over WAN 24/7. One 1080p stream at
  4 Mbit/s × 20 cameras = 80 Mbit/s sustained upload. Most Indian sites do not have
  that. **Mitigation:** pull the camera's *substream* (usually 640×360 @ ~512 kbit/s)
  — which you want anyway, see below.

### C. Genuinely on-camera (Axis ACAP, Hanwha Wisenet, Ambarella CVx)

Real, but a different project: you need the vendor SDK, and the model must be
quantized to INT8 and shrunk to a few hundred MB/s of compute. Only worth it if you
are selling a *product* to be installed with new cameras, not retrofitting an
existing site. Park this until Phase 4.

**Decision: build for (A), keep the code so (B) is a config change.** The code in this
repo does exactly that — the runner doesn't know or care whether it's on a Jetson or
a cloud VM.

## The pipeline

```
                    ┌──────────────────────────────────────────────┐
   RTSP substream   │  ingest/stream.py                            │
   640×360 @ 10 fps │  reader thread, DROPS frames when busy       │
   ────────────────>│  (never queues — a backlog means you are     │
                    │   alerting about the past)                   │
                    └───────────────┬──────────────────────────────┘
                                    │ every frame
                    ┌───────────────▼──────────────────────────────┐
                    │  ingest/ring_buffer.py                        │
                    │  last 30 s, JPEG-encoded, fixed RAM, no disk  │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
   ~0.1 ms/frame    │  pipeline/motion_gate.py                      │
                    │  downscale to 160×90, abs-diff, % pixels      │
                    │  changed > threshold?                         │
                    └───────────────┬──────────────────────────────┘
                                    │ only if motion  (~5% of the time)
                    ┌───────────────▼──────────────────────────────┐
   ~8 ms/frame      │  pipeline/person_gate.py                      │
                    │  YOLO person detection, every Nth frame       │
                    │  >= 1 person? (>= 2 for fight classes)        │
                    └───────────────┬──────────────────────────────┘
                                    │ only if people present
                    ┌───────────────▼──────────────────────────────┐
   ~60-150 ms/clip  │  models/classifier.py                         │
   every 1 s        │  16 frames spanning 2 s -> class scores       │
                    │  {normal, fight, fall, weapon, panic}         │
                    └───────────────┬──────────────────────────────┘
                                    │ raw scores
                    ┌───────────────▼──────────────────────────────┐
                    │  detect/smoother.py                           │
                    │  EMA + hysteresis: needs K of last N windows  │
                    │  above threshold to OPEN an event, and a      │
                    │  lower threshold to keep it open              │
                    └───────────────┬──────────────────────────────┘
                                    │ Event(open/close)
                    ┌───────────────▼──────────────────────────────┐
                    │  alerting/dispatcher.py                       │
                    │  dedupe key (camera, class), cooldown 120 s,  │
                    │  severity, escalation                         │
                    └──────┬────────────────────────┬──────────────┘
                           │                        │
              ┌────────────▼──────────┐  ┌──────────▼───────────────┐
              │ storage/clip_writer   │  │ alerting/webhook.py       │
              │ pre-roll from ring    │  │ HMAC-signed POST,         │
              │ buffer + 10 s post    │  │ exponential retry,        │
              │ -> ONE mp4            │  │ disk-backed outbox        │
              └───────────────────────┘  └──────────────────────────┘
```

## Two details that matter more than they look

**Pull the substream, not the main stream.** Nearly every IP camera exposes a second,
low-res H.264 channel (`/Streaming/Channels/102` on Hikvision, `subtype=1` on Dahua).
It is 640×360 or 704×576 at ~512 kbit/s. Your model runs at 224×224 anyway, so the
main 1080p/4K stream is pure waste — 8× the decode cost for zero accuracy. Record the
main stream to the NVR as usual; *analyse* the substream.

**Drop frames, never queue them.** If the model takes 150 ms and frames arrive every
100 ms, a queue grows without bound and after ten minutes you are alerting about
something that happened ten minutes ago. `ingest/stream.py` keeps only the newest
frame and lets the reader thread overwrite it. A dropped frame is invisible; a
growing queue is a silent failure that only shows up in production.

## What "notify higher authorities" should mean

Do not wire the model straight to an SMS blast. The path is:

1. **Webhook** → your backend (signed, retried, idempotent via `event_id`).
2. Backend fans out per site policy: control-room dashboard first, WhatsApp/Telegram
   to the supervisor with the 40-second clip, SMS/call only on escalation.
3. **Human confirms or dismisses in the app.** That click is your labelled data — it
   is how the system gets better, and it is the only honest way to measure the
   false-alarm rate.
4. Escalate to police/emergency contact only after confirmation, or after N minutes
   of no acknowledgement on a high-severity class.

If you skip step 3 you will ship a system that cries wolf, the guards will mute it in
week two, and the project is dead. See [04-tradeoffs.md](04-tradeoffs.md).

> Compliance note for India: an automated SMS/OTP path needs DLT registration under
> TRAI rules, and auto-dialling emergency services has its own liability. Start with
> app + WhatsApp Business API, add SMS once the false-alarm rate is measured.
