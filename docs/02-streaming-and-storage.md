# 02 — "We can't store everything and feed it to AI"

This was your sharpest question, and the answer reframes the problem: **you were
never supposed to store it.**

## First, the number that makes the point

One 1080p H.264 camera at 4 Mbit/s:

| | |
|---|---|
| per hour | 1.8 GB |
| per day | 43 GB |
| 20 cameras / day | 860 GB |
| 20 cameras / month | ~26 TB |

Nobody trains on that, and nobody should pay to keep it. Meanwhile the *interesting*
content of a month of footage at a normal site is maybe 20 minutes. The whole design
is about not paying for the other 43,180 minutes.

## The mental shift: it's a stream, not a dataset

Think of it like a heart monitor, not a video library. The model looks at a moving
2-second window and answers one question — "is this normal?" — then forgets. There is
no growing store, no "feed the video to the AI".

Concretely, three separate things, and people conflate them:

| | What it is | Storage |
|---|---|---|
| **Training data** | Public datasets + a small consented on-site collection | ~50 GB, one time, on Colab/Drive |
| **Live inference** | A 2 s sliding window in RAM | ~0 (fixed, overwritten) |
| **Evidence** | Clips around confirmed alerts | ~40 s per incident |

You only ever "store everything" if you try to make live inference and training the
same pipeline. Keep them separate and the storage problem evaporates.

## The ring buffer — the core trick

You need *pre-roll*: when the model says "fight" at 14:32:07, the alert is useless
without the ten seconds *before* it. But you can't keep the last ten seconds unless
you were already recording... which is what you were trying to avoid.

Resolution: a **fixed-size circular buffer in RAM.** Keep the last 30 seconds of
frames. Frame N+1 overwrites frame N−900. Memory usage is constant and known at
startup — it never grows, so it can never fill a disk.

```python
# ingest/ring_buffer.py — collections.deque(maxlen=N) does the hard part
buf = RingBuffer(seconds=30, fps=10)   # 300 slots, oldest silently evicted
buf.append(frame, ts)
...
clip = buf.snapshot()                  # only ever called when an alert fires
```

**Encode the frames as JPEG when they go in.** This is the difference between
practical and not:

| Storage of 30 s @ 10 fps, 640×360 | Size |
|---|---|
| raw BGR numpy (`640*360*3*300`) | **207 MB** ← per camera, unacceptable |
| JPEG q=80 (~35 KB/frame) | **~10 MB** ← fine, 8 cameras = 80 MB |

The JPEG encode costs ~1–2 ms per frame on one core. Worth it 100×. (The exotic
option is to tee the H.264 GOPs straight from the stream and never decode for
buffering at all — ~2 MB for 30 s. That's a Phase-3 optimisation; start with JPEG.)

Then, and only then:

```
alert fires at T
  ├─ pull frames [T-20s, T] from ring buffer   (they were already in RAM)
  ├─ keep recording to a temp file until T+10s
  └─ mux into one 30 s mp4, upload, put the URL in the webhook
```

**Result:** you store ~40 s per incident. At 30 incidents/day across 20 cameras
that's ~600 MB/day of clips instead of 860 GB/day of footage — a 1400× reduction —
and every byte you kept is a byte someone will actually look at.

## The gates — why you don't even run the model most of the time

Running a video transformer on every frame of every camera is what makes people think
they need a datacentre. You don't, because CCTV is overwhelmingly boring.

**Gate 1 — motion.** Downscale to 160×90 greyscale, absolute-difference against the
previous frame, count pixels over a threshold. ~0.1 ms. An empty corridor fails this
gate 95%+ of the time. Use a background subtractor (MOG2) rather than naive diff so
that a swaying tree or a flickering tube light doesn't trip it.

**Gate 2 — people.** A YOLO person detector every 5th frame, ~8 ms at 640×360. No
person means no fight and no collapse, full stop. For fight classes specifically,
require ≥2 people with overlapping/nearby boxes — that one rule kills a huge share of
false positives before the model ever sees the clip.

**Gate 3 — the model.** 16 frames spanning 2 s, every 1 s (so windows overlap 50% and
an event can't fall between them). ~60–150 ms.

Compounded, the heavy model runs on roughly **2–5% of wall-clock time**. That is what
lets one Jetson handle 8 cameras instead of one.

```
100 fps of raw frames arriving (10 cams × 10 fps)
      ↓ motion gate      → ~5 fps survive
      ↓ person gate      → ~2 fps survive
      ↓ clip windows     → ~2 model calls/sec total
```

## Do you need audio?

Screaming and glass-breaking are strong signals and audio is ~1000× cheaper than
video to process. XD-Violence (see [03](03-data-and-training.md)) is explicitly
audio-visual for this reason. **But**: recording audio raises the legal bar sharply
(in India, interception/recording of conversation has separate exposure from video
surveillance, and consent signage that covers video may not cover audio). Recommend:
skip audio for v1, revisit with legal sign-off.

## What about "video transcription per second"?

You mentioned transcribing the video per second. There is a version of that which
works and a version that doesn't:

- ❌ **Caption every second with a VLM and store the text.** ~1 caption/s/camera ×
  20 cameras = 1.7 M captions/day. At even ₹0.02 each that's ₹34k/day, latency is
  1–3 s, and you've built a very expensive way to say "a person walks down a
  corridor" 1.7 million times.
- ✅ **Use a VLM as a second-stage confirmer.** When the fast model flags something,
  send *that one clip* to a VLM (Gemini Flash / Qwen2-VL) and ask "is someone being
  assaulted here? answer yes/no + one sentence." Now you're making ~30 calls a day,
  you get a human-readable alert description for free, and it acts as a false-positive
  filter. This is a genuinely good idea — it's Phase 3 in the roadmap.
- ✅ **Use a VLM to auto-label your training data.** Offline, no latency budget. Much
  cheaper than hand-labelling.

So: the transcription instinct is right, but it belongs on the ~0.1% of clips that
already looked suspicious, not on the firehose.

## Failure modes to design for now

| Failure | Handling |
|---|---|
| Camera drops / network blip | Reader thread reconnects with backoff; `camera_offline` is itself an alert |
| Internet down at site | Disk-backed outbox, replay webhooks on reconnect, idempotent `event_id` |
| Model slower than real time | Drop frames (never queue); export a `lag_seconds` metric and alarm on it |
| Disk fills with clips | Clip dir has a hard cap + LRU eviction; inference never writes anywhere else |
| Camera moved / refocused | Scene-change detector → flag for re-tuning, thresholds are per-camera |
| Night / IR mode | IR footage looks nothing like daytime — needs its own eval slice, likely its own threshold |
