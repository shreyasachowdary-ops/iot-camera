# 07 — Training guide

Hands-on. Read [00-primer.md](00-primer.md) first for what the terms mean.

Four commands, in order:

```
prepare  ->  train  ->  evaluate  ->  export
```

---

## 0. The five-minute demo (no data, no GPU)

Run this before the meeting to confirm everything works, and to have something live to
show. It takes about 3 minutes on a laptop CPU.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only is fine
export PYTHONPATH=src

bash scripts/demo.sh
```

That generates a synthetic dataset, prepares it, trains three models, compares them,
exports the best, and runs live inference. Every number below in this doc came from
actually running it.

**Be upfront that the data is synthetic.** "Fight" is fast erratic motion and "normal"
is slow walking, which is far easier than the real problem. It proves the pipeline, not
the approach.

---

## 1. Organise your footage

```
data/raw/
  normal/
    cam01_2024-05-03_0001.mp4
    cam01_2024-05-03_0002.mp4
    cam02_2024-05-04_0001.mp4
  fight/
    cam01_2024-05-03_0007.mp4
    ...
```

Two rules:

**One directory per class.** The directory name is the label. The `--classes` order
defines the label indices, and that order must stay identical through training, export,
and inference. Swap them and every fight scores as normal, with no error, ever.

**Name files `<group>_<whatever>.mp4`.** The group is everything before the first
underscore, and it is what stops your results being fiction. All clips from `cam01`
land on the same side of the train/val split, so the model can't recognise the corridor
instead of the action. Group by camera at minimum; camera + day is better.

> If your footage is long recordings rather than clips, cut it into 3–5 second clips
> around each labelled event first, and take "normal" clips from the same cameras at
> other times. Sampling normals from *different* cameras than your fights is a classic
> way to accidentally build a camera classifier.

---

## 2. Prepare

```bash
python -m cctv_sentinel.training.prepare \
    --raw-dir data/raw \
    --out-dir data/cache \
    --classes normal fight \
    --num-frames 16 \
    --size 224
```

Decodes every video once into a single `uint8` memmap. Decoding is ~95% of epoch time,
so doing it once turns a 25-minute epoch into a 40-second one. It's resumable — if it
dies, rerun the same command.

Output:

```
prepared 192 clips (192 decoded, 0 failed)
  train    144 clips  (fight=72, normal=72)
  val       24 clips  (fight=12, normal=12)
  test      24 clips  (fight=12, normal=12)
```

**Check the class balance here, before anything else.** If it says `normal=1800,
fight=40`, accuracy is meaningless from this point on and you must judge the run on
PR-AUC and recall.

Sizing: `n_clips × num_frames × size × size × 3` bytes. 2,000 clips at 16×224×224 is
**4.8 GB**. At `--size 112` it's 1.2 GB.

**Warning to take seriously:** if you see `only N distinct groups`, you don't have
enough cameras/sessions for a trustworthy split. With 3 cameras, whichever one lands in
val dominates your result. Collect more, or use k-fold and report error bars.

---

## 3. Train

### On the DGX — all 8 GPUs

```bash
torchrun --nproc_per_node=8 -m cctv_sentinel.training.train \
    --cache-dir data/cache \
    --out-dir runs/videomae \
    --arch videomae \
    --epochs 15 \
    --batch-size 8 \
    --head-lr 1e-4 \
    --backbone-lr 1e-5 \
    --amp-dtype bf16 \
    --num-workers 8 \
    --monitor pr_auc
```

`torchrun --nproc_per_node=8` starts 8 processes, one per GPU, using
DistributedDataParallel. `--batch-size 8` is **per GPU**, so the effective batch is 64.

Use **bf16** on A100/H100 — same numeric range as fp32, so no loss scaling needed.

> **The dataloader is usually your bottleneck, not the GPU.** If `nvidia-smi` shows low
> utilisation, raise `--num-workers` and make sure `data/cache` is on local NVMe, not
> network storage. On a DGX this is the difference between 8 GPUs working and 8 GPUs
> waiting.

### On one cloud GPU

Same command without `torchrun`:

```bash
python -m cctv_sentinel.training.train \
    --cache-dir data/cache --out-dir runs/videomae \
    --arch videomae --epochs 15 --batch-size 8 --accum-steps 4
```

`--accum-steps 4` accumulates gradients over 4 batches, giving an effective batch of 32
without the memory of one. This is the fix for `CUDA out of memory`.

Rough guide: a T4 (16 GB) handles batch 4–8 for VideoMAE; an A100 handles 16–32.

### Resuming

Checkpoints are written every epoch. Add `--resume` to the same command and it picks up
where it stopped. Do this by default on a shared/preemptible machine.

### Reading the output

```
epoch   3 |  12.4s | lr 1.50e-03 | train 0.2174 | val 0.1198 | acc 1.0000 | pr_auc 1.0000
  new best pr_auc: 1.0000 -> runs/tiny/best.pt
```

Watch **train vs val loss**. Both falling is healthy. Train falling while val rises is
overfitting, and the trainer warns you explicitly when it sees three consecutive epochs
of it. Use `best.pt` (best val score), never `last.pt`.

### The flags that matter most

| Flag | Why |
|---|---|
| `--arch` | `videomae` for best accuracy, `r3d_18` for a fast solid baseline, `tiny3dcnn` for CPU testing |
| `--batch-size` | Lower it first when you hit OOM |
| `--accum-steps` | Bigger effective batch without more memory |
| `--head-lr` / `--backbone-lr` | Head 10× the backbone. See [primer §8](00-primer.md#8-transfer-learning) |
| `--monitor pr_auc` | What "best" means. PR-AUC when imbalanced |
| `--early-stop-patience 5` | Stop when val stops improving |
| `--no-class-weights` | Turn OFF the automatic inverse-frequency weighting. Leave it on if imbalanced |

---

## 4. Evaluate

```bash
python -m cctv_sentinel.training.evaluate \
    --checkpoint runs/videomae/best.pt \
    --cache-dir data/cache \
    --out runs/videomae/eval.json
```

This deliberately does things in the credible order:

1. Picks the decision threshold on **validation**.
2. Applies that threshold **unchanged** to test.
3. Reports a per-group breakdown, so a model that works on 4 cameras and fails on the
   5th can't hide behind an average.

```
                       predicted fight      predicted normal
  actual fight               12 (TP)               0 (FN)  <- missed
  actual normal               0 (FP)              12 (TN)
                         ^ false alarms

  precision          1.0000   of 12 flagged, 12 were real
  recall             1.0000   of 12 real events, 12 caught
  PR-AUC             1.0000   <- prefer this one when imbalanced
```

If val PR-AUC greatly exceeds test, it says so — usually a sign the val set isn't
representative, which points back at the group split.

---

## 5. Always run the baselines

```bash
python scripts/compare_baselines.py --cache-dir data/cache --epochs 8
```

**Bring this table to the meeting instead of a single number.** Real output from the
synthetic demo:

```
model                params   train   PR-AUC  ROC-AUC      F1  recall   prec
----------------------------------------------------------------------------
motion_baseline       0.00M      9s    1.000    1.000   1.000   1.000  1.000
frame_average         0.09M     31s    0.397    0.292   0.667   0.917  0.524
tiny3dcnn             0.29M     15s    1.000    1.000   1.000   1.000  1.000

WARNING: tiny3dcnn beats plain motion energy by only +0.000 PR-AUC.
```

That warning is the tool working. On this synthetic data "fight" *is* just fast motion,
so a neural network adds nothing — and it says so rather than letting you report 1.000
as a success.

How to read it on your real data:

| Result | Meaning |
|---|---|
| Your model ≈ `motion_baseline` | The task is "how much movement is there". You don't need a neural network. Look at the data. |
| Your model ≈ `frame_average` | The signal is appearance, not motion. **Suspect leakage** — check the group split. |
| Clear gap over both | Temporal modelling is genuinely earning its complexity. This is the result you want. |

`frame_average` is the sharper diagnostic: it sees every frame but not their *order*, so
the gap between it and a real video model is exactly the value of temporal modelling.

---

## 6. Export and run it live

```bash
python -m cctv_sentinel.training.export \
    --checkpoint runs/videomae/best.pt --out-dir weights --size 224

python -m cctv_sentinel.cli run --config configs/trained.yaml --source demo.mp4
```

Export verifies the TorchScript output matches the source model numerically rather than
assuming it — tracing silently bakes in whatever control flow it happened to take.

`configs/trained.yaml` is set up for this. The one thing to get right:

```yaml
model:
  classes: [normal, fight]     # MUST match weights/classes.txt, in the same order
```

---

## 7. Things that will go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too big | Lower `--batch-size`, add `--accum-steps`, keep AMP on |
| Loss becomes `NaN` | LR too high, or fp16 instability | Lower LR 10×; use `--amp-dtype bf16` |
| Val accuracy ~50% and flat | LR too low, or labels wrong | Check the manifest; try LR 10× higher |
| Val ≫ test | Leakage, or unrepresentative val | Check groups actually differ across splits |
| 99% accuracy immediately | Class imbalance | Look at the balance printout; judge on PR-AUC |
| GPUs at low utilisation | Dataloader starved | Raise `--num-workers`; move cache to local NVMe |
| Perfect scores on real data | Almost always leakage | Verify no group appears in two splits |

That last one deserves emphasis: **if your first real run scores 99%, assume something
is wrong before you assume you succeeded.** Check the group split first.

---

## 8. A realistic plan

**Week 1 — get honest numbers**
Organise and label footage. Run `prepare`. Look hard at the class balance and group
count. Run `compare_baselines.py`. You now know whether the problem is tractable.

**Week 2 — the real model**
Fine-tune `videomae` on the DGX. Report PR-AUC with the threshold sweep, plus the
baseline table. Do **not** touch the test set yet.

**Week 3 — make it trustworthy**
K-fold cross-validation for error bars. Per-camera breakdown. Ablations: does night
footage hurt? does augmentation help? Evaluate on test **once**, at the end.

**Week 4 — demo**
Export, run the live pipeline on held-out video, show detections firing in real time.

If asked "how long to a paper-quality result": **the data work is the long pole, not
the training.** Training VideoMAE on 2,000 clips is under an hour on a DGX. Labelling
those 2,000 clips is weeks.
