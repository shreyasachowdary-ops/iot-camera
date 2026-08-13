# 03 — Data, models, and how you actually train this on Colab

## Step 0: stop saying "anomaly". Pick classes.

"Detect abnormality" is not trainable. Every hour you spend narrowing this saves a
month later. The honest split:

**Closed-set classification (what you should build first).** You enumerate the events
you care about and train a supervised classifier. Works, is measurable, ships.

```
normal | fight | fall_collapse | weapon_brandish | crowd_run | loitering_intrusion
```

**Open-set anomaly detection ("anything unusual").** You train only on normal footage
and flag deviation. This is what the research papers mean by anomaly detection, it is
much harder, it produces a score you can't explain to a security guard, and in
practice it fires on a delivery van parked oddly. It is *not* a good v1.

**Recommendation: start with two classes that matter most and are most separable —
`fight` and `fall_collapse`.** Add weapons after, because weapon detection is an
object-detection problem, not a video-classification one, and wants its own model.

## Datasets (all publicly available, all free for research)

| Dataset | Size | What it gives you | Notes |
|---|---|---|---|
| **RWF-2000** | 2,000 clips, 5 s, real surveillance | fight / no-fight | **Start here.** Actual CCTV footage, not movies. SOTA ~90%. |
| **Real Life Violence Situations** | 2,000 clips | violence / non-violence | Mixed sources, good augmentation of RWF |
| **UCF-Crime** | 1,900 untrimmed videos, 128 h | 13 classes: abuse, arrest, arson, assault, burglary, explosion, fighting, road accident, robbery, shooting, shoplifting, stealing, vandalism | *Weakly* labelled — video-level only, no timestamps. Needs MIL (below). The standard benchmark. |
| **XD-Violence** | 4,754 videos, 217 h | violence, audio+visual | Largest; has audio track |
| **Hockey Fight / Movies Fight** | 1,000 / 200 | fight | Small, easy, not CCTV-like. Sanity checks only. |
| **UR Fall Detection / Le2i FDD** | 70 / 191 | falls | Small and staged, but falls are rare in the wild so you need them |
| **NTU RGB+D 120** | 114k clips | 120 actions incl. falling down, punching, kicking | Skeleton data — pairs perfectly with a pose model |
| **ShanghaiTech / CUHK Avenue / UCSD Ped** | — | unsupervised anomaly | For the open-set route only |

**The one thing you must add: your own site footage.** Public datasets are shot in
different places, angles, and lighting. A model at 90% on RWF-2000 routinely lands at
60% on a real camera at a real height on a real Indian street. Budget for:

- 2–4 weeks of *normal* footage from the actual cameras (this is your negative class
  and your false-positive test set — the single highest-value data you will collect).
- A staged collection day: get consenting people to stage falls, shoves, scuffles,
  running, at the real camera angles, day and night. 200 staged clips from your own
  cameras will move the needle more than another public dataset.

Set up consent + signage before you record a single frame. See
[04-tradeoffs.md](04-tradeoffs.md).

## Model choices

### Route 1 — Clip classifier (recommended v1)

Fine-tune a pretrained video backbone on RWF-2000-style clips.

| Model | Params | Latency (T4, 16 frames) | Notes |
|---|---|---|---|
| **X3D-M** | 3.8 M | ~25 ms | Best accuracy-per-FLOP; great on Jetson. **Pick this for edge.** |
| **VideoMAE-Base** | 87 M | ~110 ms | Strongest accuracy, self-supervised pretrain, fine-tunes well on small data |
| MoViNet-A2 (stream) | 4.8 M | ~15 ms | Designed for streaming — carries state across frames instead of re-reading a window |
| R(2+1)D-18 | 33 M | ~45 ms | Old but reliable baseline |

Start VideoMAE-Base on Colab to find out what accuracy is achievable, then distil to
X3D-M for the edge box if you're compute-bound.

### Route 2 — Pose / skeleton (do this *in addition*, especially for falls)

`YOLO11-pose` (or RTMPose) → keypoints per person per frame → either hand-written
rules or a small ST-GCN.

For **fall/collapse** this is better than a video model and it's not close:
- bounding-box aspect ratio flips from tall to wide,
- vertical velocity of the hip keypoint spikes,
- then the person stays near-horizontal for >5 s (this last condition is what
  separates a fall from sitting down or bending over).

It's ~30 lines of rules, needs no training data, runs on a CPU, and is *explainable* —
you can tell the customer exactly why it fired. For fights, pose helps too (velocity
of wrist keypoints, inter-person distance) but is weaker than a video model.

**Ship pose-based falls in Phase 1.** It gets you a working demo in a week.

### Route 3 — MIL on frozen features (this is what makes UCF-Crime tractable on Colab)

UCF-Crime has no timestamps — a 30-minute video is labelled "robbery" with no idea
when. Multiple Instance Learning handles this: treat a video as a bag of 32 segments,
train so that the *max-scoring* segment of an anomalous video scores higher than the
max of any normal video. (Sultani et al., CVPR 2018; then RTFM, MGFN, UR-DMU.)

The practical magic:

```
Stage 1 (once, slow):  video -> I3D or CLIP features -> save .npy   [~2 GB for all of UCF-Crime]
Stage 2 (fast, repeat): tiny 3-layer MLP on those features          [trains in 5 MINUTES on a T4]
```

Because stage 2 is a 2 MB MLP over precomputed features, you can iterate 50 times in
an afternoon on free Colab. **This is the single best fit between your compute budget
and the problem.** Do the feature extraction once, checkpoint the `.npy` files to
Drive, and never touch a video decoder again.

### Route 4 — VLM, zero training

Send clips to Gemini Flash / Qwen2-VL with a prompt. No training, works day one, gives
you a sentence for the alert. Too slow and too expensive for the firehose; excellent
as a **second-stage confirmer** and as an **auto-labeller**. See
[02](02-streaming-and-storage.md#what-about-video-transcription-per-second).

## The Colab reality check

You asked what to expect. Honestly:

**Pros**
- Free T4 (16 GB) is genuinely enough for everything above, if you use precomputed features.
- Zero environment setup; `torch`, `transformers`, `decord`, `pytorchvideo` install in a minute.
- Drive mounting makes checkpointing easy.
- Colab Pro (~₹1,000/mo) gets you L4/A100 and **background execution**, which is the
  feature that actually matters.

**Cons — plan around every one of these**
- **Idle disconnect ~90 min, hard cap ~12 h (free tier often less, and it's a lottery).**
  Any run longer than an hour *must* checkpoint to Drive every epoch and be resumable.
  Write the resume logic on day one, not after you lose a 6-hour run.
- **The filesystem is ephemeral.** Anything not in Drive or GCS is gone.
- **Drive I/O is terrible for many small files.** Reading 100k individual JPEGs from
  Drive will be slower than the GPU by 10×. **Pack your dataset into `.tar` shards or
  a single `.npy`/`webdataset` and copy it to local `/content` disk first.** This one
  mistake is the most common reason people think "Colab is too slow to train on".
- **GPU quota**: free tier throttles you after sustained use, often for a day.
- **You cannot host the real-time service on Colab.** No inbound RTSP, no persistent
  daemon, no static IP. Colab trains; something else serves. Do not design around it.

**Verdict:** Colab is the right tool for training here, provided you use the
precomputed-features approach. If you find yourself decoding video inside the training
loop on Colab, you've taken a wrong turn.

## The training recipe

`notebooks/train_colab.ipynb` implements this. Summary:

1. **Download RWF-2000** to `/content` (not Drive).
2. **Decode once, cache forever.** Every clip → 16 frames at 224×224 → one `uint8`
   numpy memmap. 2,000 clips ≈ 4.8 GB. Save the memmap to Drive. All subsequent runs
   skip decoding entirely — epoch time drops from ~25 min to ~40 s.
3. **Fine-tune** `MCG-NJU/videomae-base-finetuned-kinetics`, replacing the head with
   2 classes. AdamW, lr 1e-4 head / 1e-5 backbone, cosine schedule, 10 epochs,
   batch 8 with grad-accum 4, mixed precision.
4. **Augment** like the deployment: random horizontal flip, random resized crop,
   colour jitter, **and a heavy grayscale/low-light augmentation** to simulate IR
   night footage. Also downscale-then-upscale to mimic a low-bitrate substream.
5. **Evaluate on a held-out set, then on your own site footage separately.** Report
   both. The gap between them is the number that tells you whether you have a product.
6. **Export** to TorchScript and ONNX. Quantize INT8 for the edge box.
7. **Pick the threshold from a PR curve**, targeting a false-alarm *rate*, not an
   accuracy. See below.

## Measure the right thing

Accuracy on a balanced benchmark is meaningless here. Real footage is ~99.99% normal,
so a model that always says "normal" scores 99.99%.

Track these instead:

| Metric | Target for a shippable v1 |
|---|---|
| **False alerts per camera per day** | **< 1.** This is the product-defining metric. |
| Recall on real events | > 80% (a missed fight is bad, but guards muting the system is worse) |
| Time-to-alert (event start → webhook sent) | < 5 s |
| AUC / AP on held-out site data | report it, but never tune on it alone |

Set the decision threshold by sweeping it against **your own site's normal footage**
until you hit the false-alarm budget, *then* measure what recall you have left. That's
the honest ordering, and it's the opposite of what most tutorials do.
