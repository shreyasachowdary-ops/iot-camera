# 05 — Roadmap: what to build, in what order

The ordering principle: **prove the pipeline before you prove the model.** A perfect
model with no working stream/alert path is worth nothing; a stub model in a working
pipeline is a demo you can show someone on Friday.

---

## Phase 0 — Decide (this week, no code)

Answer these four. Everything downstream depends on them.

1. **Which events?** Recommend starting with `fight` + `fall_collapse` only.
2. **Which site?** You need one real pilot location with real cameras and a
   cooperative owner. Without it you are guessing.
3. **Who receives the alert, on what device, and what do they do in the next 60
   seconds?** If you can't answer this, the product doesn't exist yet.
4. **Edge box or central server?** ([01-architecture.md](01-architecture.md) — recommend edge.)

Also start the paperwork now, because it has lead time: signage, the contract clause
letting you train on site footage, DLT registration if you'll ever send SMS.

---

## Phase 1 — The pipeline, with a stub model (2 weeks)

**Goal: end-to-end alert from a live camera, with a model that doesn't work yet.**

- [x] RTSP ingest with frame-dropping and reconnect — `ingest/stream.py`
- [x] Ring buffer, JPEG-encoded, fixed memory — `ingest/ring_buffer.py`
- [x] Motion gate — `pipeline/motion_gate.py`
- [x] Clip sampler (16 frames / 2 s, sliding) — `pipeline/sampler.py`
- [x] Score smoothing + hysteresis event state machine — `detect/smoother.py`
- [x] Dedupe / cooldown dispatcher — `alerting/dispatcher.py`
- [x] Signed webhook with retry + disk outbox — `alerting/webhook.py`
- [x] Clip writer (pre-roll + post-roll → mp4) — `storage/clip_writer.py`
- [x] Stub motion-energy "model" so the whole loop runs with no GPU
- [ ] **Pose-based fall detection** (YOLO11-pose + rules). This is a *real* detector
      that needs no training — it gives you a genuine working demo in Phase 1.
- [ ] Person gate using the same YOLO model
- [ ] Prometheus metrics: fps, lag_seconds, gate pass rates, alerts/hour

**Exit criterion:** you stand in front of a camera, lie down on the floor, and a
webhook with a clip lands in your phone within 5 seconds.

---

## Phase 2 — Train the real model (3–4 weeks)

**Goal: a fight classifier that beats the stub, trained on Colab.**

- [ ] Download RWF-2000 + Real Life Violence; decode-once to a numpy memmap on Drive
- [ ] Fine-tune VideoMAE-Base, 2-class. Target > 88% on the RWF test split
- [ ] Augment for night/IR and low-bitrate substreams
- [ ] Export TorchScript + ONNX; wire into `models/classifier.py` `backend: torch`
- [ ] Benchmark on the target hardware; if too slow, distil to X3D-M and INT8-quantize
- [ ] **Collect 2 weeks of normal footage from the pilot site** — sweep the threshold
      against it until false alerts < 1/camera/day, then report the recall you're left with

Run the UCF-Crime MIL track in parallel if you want the wider class list — it's cheap
once features are extracted (see [03](03-data-and-training.md#route-3)).

**Exit criterion:** a threshold that satisfies the false-alarm budget on *your site's*
footage, with a documented recall number on staged events.

---

## Phase 3 — Make it survivable in production (3 weeks)

- [ ] **Staged collection day** at the pilot site: consented volunteers, real angles,
      day and night. ~200 clips. Fine-tune on them. This is the biggest single
      accuracy jump you will get.
- [ ] **VLM second-stage confirmer** — flagged clips go to Gemini Flash / Qwen2-VL for
      a yes/no + one-line description. Cuts false positives and writes your alert text.
- [ ] Confirm/dismiss UI. Every click is a label. Feed it back monthly.
- [ ] Per-camera thresholds and per-camera zone masks (ignore the road outside the gate)
- [ ] Multi-camera scheduling on one box; graceful degradation when overloaded
- [ ] Offline queue + replay; camera-offline alerting; watchdog/systemd unit
- [ ] Load test: 8 cameras, 24 h, measure drift in lag and memory

---

## Phase 4 — Only if the pilot works

- On-camera deployment (Axis ACAP / Ambarella) for new installs
- Weapon detection (separate object-detection model)
- Cross-camera tracking (person re-ID) — note this *does* raise the privacy bar
- Open-set anomaly detection as a secondary, lower-severity signal
- Multi-tenant backend, per-site dashboards

---

## The one-week version

If you want something to show someone quickly, skip everything above and do this:

```
Day 1-2  Get a camera streaming into the runner in this repo (stub model). Works today.
Day 3-4  Add YOLO11-pose. Fall detection from rules: aspect ratio flip + hip velocity
         + stays down > 5 s.
Day 5    Point the webhook at a Telegram bot. Attach the clip.
Day 6-7  Record 20 staged falls at the real camera angle. Tune the rules against them.
```

That's a genuine, demonstrable fall-detection system in a week, with zero training —
and it is a far better thing to show an investor or a customer than a notebook with a
92% accuracy number in it.
