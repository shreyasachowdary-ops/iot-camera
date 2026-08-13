# 08 — Meeting cheat-sheet

One page. Skim it on the way in.

---

## If you say nothing else, say these five things

1. **"I want to check the class balance before trusting any accuracy number."**
   Real CCTV is ~99.9% normal, so a model that always says "normal" scores 99.9%.
   Signals immediately that you won't be fooled by a headline number.

2. **"We should split by camera, not randomly, or we'll leak scene identity."**
   The single highest-value sentence on this page. Random splits put the same corridor
   in train and val, so the model recognises the *place* rather than the *action* and
   validation is inflated by 20+ points.

3. **"What's our baseline?"**
   Always have a dumb one. If an 87M-parameter transformer barely beats motion energy,
   that's the most important thing to learn — and better learned in week one.

4. **"Can we agree the test set stays sealed until the end?"**
   Every peek at test makes it less trustworthy. Standard research discipline.

5. **"How many actual incidents are in the footage?"**
   Not hours — *incidents*. 500 hours with 12 fights is a 12-example problem, and that
   changes the entire approach.

---

## The 30-second version of the project

> We're training a video classifier to spot fights and collapses in CCTV. The model
> looks at ~2-second clips — 16 frames sampled across the window — and outputs a
> probability per class. We fine-tune a Kinetics-pretrained backbone rather than
> training from scratch, because we have hundreds of clips, not millions. The main risks
> are class imbalance and data leakage across the train/test split, so we're splitting
> by camera and judging on PR-AUC rather than accuracy.

If you can deliver that, you're fine.

---

## Terms most likely to come up

| They say | It means |
|---|---|
| "What's your **batch size**?" | Clips processed at once. Limited by GPU memory. |
| "Are you using **AMP**?" | 16-bit math. ~2× faster, half the memory. Yes, bf16. |
| "Is it **overfitting**?" | Memorising instead of generalising. Watch val loss rise while train falls. |
| "What's the **AUC**?" | Threshold-independent quality. **PR-AUC** when imbalanced. |
| "Did you **freeze** the backbone?" | Hold pretrained layers fixed. We don't; we use a 10× lower LR on them. |
| "What's your **LR schedule**?" | How learning rate decays. Cosine. |
| "**Epochs**?" | Full passes over the training data. |
| "Are you doing **DDP**?" | Multi-GPU training. Yes — `torchrun --nproc_per_node=8`. |
| "What's the **class balance**?" | Ratio of fight to normal. Ask for this early. |
| "How did you **split**?" | Train/val/test. **Say: by camera group, not randomly.** |
| "**Transfer learning** from what?" | Kinetics-400 pretrained VideoMAE. |
| "What's the **inference latency**?" | Time per clip. ~110 ms for VideoMAE-Base on a T4. |

---

## Numbers worth having memorised

| | |
|---|---|
| Clip length fed to the model | 16 frames spanning ~2 s |
| Input resolution | 224×224 |
| Tensor shape | `(B, C, T, H, W)` = `(8, 3, 16, 224, 224)` |
| VideoMAE-Base size | 87M parameters |
| Pretraining dataset | Kinetics-400, 650k clips, 400 actions |
| RWF-2000 | 2,000 clips of real surveillance, fight/no-fight |
| Typical batch size, A100 | 16–32 for VideoMAE |
| Windows per camera per day | 86,400 at 1/second |
| 1080p camera storage | ~43 GB/day |

That windows-per-day figure is the one that makes the false-alarm argument concrete: a
0.1% false-positive rate is 86 false alerts a day from a single camera.

---

## Questions to ask

**Data (ask these first — they matter most)**
- How many hours, and how many actual incidents?
- Who labels it, and do we mark start/end times or tag whole clips?
- How many distinct cameras and locations? Night footage as well as day?
- Can we split by camera and recording day?
- Are we cleared to train on this footage — consent, DPDP, retention?

**Scope**
- Specific named events, or open-set "anything unusual"? (Very different problems.)
- Is the deliverable a paper, a benchmark number, or a demo?
- What counts as success — a target PR-AUC, or beating a specific baseline?

**Method**
- Kinetics-pretrained VideoMAE, or a domain-specific checkpoint you'd prefer?
- Do we want a pose-based branch for falls alongside the video classifier?
- Given the dataset size, should we do k-fold rather than a single split?

**Infrastructure**
- How many DGX GPUs can we actually use, and is it shared or scheduled?
- Is the data on local NVMe or network storage? (Decides whether the dataloader or the
  GPU is the bottleneck — usually the dataloader.)
- Cloud fallback? Are we allowed to move footage off-premises? (Often no, which settles it.)

---

## What you can demo, right now

```bash
bash scripts/demo.sh
```

Three minutes on a laptop CPU: synthetic footage → prepare → three models compared →
train → evaluate → export → live detection firing on a video stream.

The most interesting moment is the baseline table, which on the synthetic data prints:

```
WARNING: tiny3dcnn beats plain motion energy by only +0.000 PR-AUC.
```

That's the tooling catching a "your model isn't earning its complexity" result
automatically. Showing that you built the check — rather than a slide with 99% on it —
is the thing that will land.

---

## If you get asked something you don't know

Say so, then say what you'd do to find out. "I don't know — I'd check X" is a good
answer and everyone in the room has given it. Guessing is the only bad option, and
senior engineers detect it instantly.

Two specific traps:

- **Don't defend a number you can't explain.** If you don't know how a metric was
  computed, say you'll verify it.
- **Don't agree that accuracy looks great.** Ask what the class balance was.
