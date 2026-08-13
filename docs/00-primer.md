# 00 — The Primer

**Everything conceptual, from zero.** Written for someone who has not trained a model
before and has a meeting with a senior ML engineer. No prior knowledge assumed.

Read it start to finish once. Then use [§11](#11-glossary) as a lookup table and
[§12](#12-questions-to-ask-in-the-meeting) on the day.

**Contents**

1. [Why a model at all — why not just write rules?](#1-why-a-model-at-all)
2. [What "a model" actually is](#2-what-a-model-actually-is)
3. [How video becomes numbers](#3-how-video-becomes-numbers)
4. [Frames, FPS, and clips](#4-frames-fps-and-clips)
5. [The training loop](#5-the-training-loop)
6. [Epochs, batches, and steps](#6-epochs-batches-and-steps)
7. [Overfitting — the thing that will actually go wrong](#7-overfitting)
8. [Transfer learning — why you don't start from zero](#8-transfer-learning)
9. [Model architectures for video](#9-model-architectures-for-video)
10. [Measuring whether it works](#10-measuring-whether-it-works)
11. [Glossary](#11-glossary)
12. [Questions to ask in the meeting](#12-questions-to-ask-in-the-meeting)

---

## 1. Why a model at all?

The obvious first instinct is to write rules. "If lots of pixels change quickly, it's a
fight." That's a reasonable thing to try, and it fails immediately:

- A camera panning changes every pixel. Not a fight.
- Someone walking past close to the lens changes many pixels. Not a fight.
- A light switching on changes every pixel. Not a fight.
- Two people fighting quietly in a corner change very few pixels. **Is** a fight.

You could keep adding rules — but you'd need one for every lighting condition, camera
angle, crowd density, and clothing colour. Nobody can enumerate those, because **"fight"
has no closed-form definition in terms of pixels.** You know one when you see one, but
you cannot write down what makes it one.

That's the exact situation machine learning is for. Instead of writing the rule, you
provide **examples** — thousands of clips labelled "fight" or "not fight" — and an
algorithm searches for a rule that fits them. The rule it finds is far too complicated
for a human to have written or to read afterwards. That's the trade: you give up
understanding the rule, and in exchange you get a rule that works.

> **The one-sentence version for the meeting:** we use ML because the concept we're
> detecting can be demonstrated with examples but cannot be specified with rules.

---

## 2. What "a model" actually is

Strip away the mystique and a neural network is **a mathematical function with a very
large number of adjustable knobs.**

- **Input:** a clip of video, expressed as numbers.
- **Output:** two numbers, e.g. `[0.02, 0.98]` — "2% normal, 98% fight".
- **In between:** millions of multiplications and additions, where the specific numbers
  being multiplied are the **parameters** (also called **weights**).

A model with "87 million parameters" has 87 million knobs. **Training is the process of
turning those knobs so that the outputs match the labels on your examples.** That's it.
Everything else in this document is detail about how the turning is done.

Before training, the knobs are random and the output is meaningless. After training,
they're set so the function reliably maps fight-looking clips to a high fight score.

Two terms you'll hear constantly:

- **Architecture** — the *shape* of the function. How many layers, arranged how. This is
  chosen by a human, up front. ("We're using VideoMAE.")
- **Weights** (or **checkpoint**) — the *values* of the knobs. This is what training
  produces, and what you save to a file. ("Load the epoch-8 checkpoint.")

Same architecture + different weights = a completely different model. A file called
`best.pt` is weights, not architecture; you need both to run anything.

---

## 3. How video becomes numbers

A model does arithmetic, so everything must be numbers first.

**A pixel** is 3 numbers: red, green, blue, each 0–255. `(0,0,0)` is black,
`(255,255,255)` white.

**An image** is a grid of pixels. A 1920×1080 image is 1920 × 1080 × 3 = **6,220,800
numbers**.

**A video** is a sequence of images ("frames") shown fast enough to look like motion.

So a 5-second 1080p clip at 30 fps is 150 frames × 6.2M = **933 million numbers**. That
is far too much to feed a model, which is why nearly every practical decision below is
about throwing data away intelligently.

### Tensors and shape

A **tensor** is just "an array with more than 2 dimensions" — the container these
numbers live in. The thing that confuses every beginner is **shape**: the size of each
dimension, written as a tuple.

For video you'll see these letters constantly:

| Letter | Means | Typical |
|---|---|---|
| **B** | Batch — how many clips processed at once | 8 |
| **C** | Channels — colour (3 = RGB) | 3 |
| **T** | Time — how many frames in the clip | 16 |
| **H** | Height in pixels | 224 |
| **W** | Width in pixels | 224 |

So `(8, 3, 16, 224, 224)` means: 8 clips, 3 colour channels, 16 frames each, 224×224
pixels. That's 8 × 3 × 16 × 224 × 224 = 19,267,584 numbers in one batch.

**Different libraries order these differently and this is a genuine, constant source of
bugs.** OpenCV gives you `(H, W, C)`. PyTorch convolutions want `(C, T, H, W)`.
HuggingFace's VideoMAE wants `(B, T, C, H, W)` — note T and C swapped. If you get it
wrong you usually get a loud crash, but sometimes you get silently garbage results,
which is worse.

You'll see `.permute(0, 2, 1, 3, 4)` in the code — that's reordering dimensions, and
that specific one swaps T and C.

### Normalisation

Raw pixels are 0–255. Models train badly on numbers that big, so we rescale to roughly
−1…+1 by dividing by 255, then subtracting a **mean** and dividing by a **standard
deviation**:

```
mean = [0.485, 0.456, 0.406]     # these exact numbers are from ImageNet
std  = [0.229, 0.224, 0.225]
```

**Those magic numbers must match whatever the pretrained model used.** Get them wrong
and the model still runs, produces no error, and is quietly a few points worse. It's a
classic silent bug — always check the training and inference code use identical
normalisation.

---

## 4. Frames, FPS, and clips

**FPS (frames per second)** is how many images per second the camera produces. CCTV is
typically 15–30.

### Do you need every frame? No.

Consecutive frames are nearly identical — at 30 fps, frame 1 and frame 2 differ by
1/30th of a second of movement. That redundancy is wasted compute. So we
**sub-sample**: take 16 frames spread evenly across the clip.

For a 5-second clip at 30 fps (150 frames), taking 16 evenly spaced frames means every
~9th frame — covering the full 5 seconds at about 3.2 fps effective.

**Why 16?** Because that's what the pretrained models expect. It's not magic; it's
convention that pretrained weights are built around. Deviating means you can't use
their pretrained weights as cleanly.

### The critical trade-off

- **Too few frames / too short a span** → the model can't see the action. A punch takes
  ~0.3 s; if your 16 frames span 10 seconds, you sample every 0.6 s and may miss it.
- **Too many frames / too long a span** → more compute, more memory, and the actual
  event is diluted among lots of nothing.

**2–3 seconds is the standard span for action recognition**, because most human actions
take about that long. That's the reasoning behind `clip_frames: 16` and
`clip_span_s: 2.0` in the configs.

### Resolution

We resize to **224×224**. Two reasons:

1. Pretrained models were trained at that size.
2. Compute scales with the *square* of resolution. 448×448 is 4× the work of 224×224
   for typically 1–2 points of accuracy.

Yes, you lose detail. For "are these two people fighting" you don't need to resolve a
face — the signal is in the motion, and that survives downscaling fine.

### The unit of classification: the clip

This matters conceptually. **The model does not classify "a video" or "a frame". It
classifies a short clip.**

- A *frame* has no motion in it, so it can't distinguish a punch from a wave.
- A *video* is too long and contains many different things.
- A *clip* (16 frames / ~2 s) is the right unit — long enough to contain an action,
  short enough to be about one thing.

At inference you slide this window along: score seconds 0–2, then 1–3, then 2–4. Your
video-level answer is then built from the clip-level scores.

---

## 5. The training loop

Here is the entire process. Everything else is refinement.

```
repeat many times:
    1. take a batch of clips + their correct labels
    2. run them through the model                -> predictions       (FORWARD PASS)
    3. compare predictions to labels             -> loss (a number)
    4. compute which direction each weight       -> gradients         (BACKWARD PASS)
       should move to reduce the loss
    5. nudge every weight a small step           -> updated weights   (OPTIMIZER STEP)
       in that direction
```

### Step 2 — forward pass

Push the clip through the model, get out `[0.3, 0.7]`. Early in training this is
essentially random.

### Step 3 — the loss function

The **loss** is a single number measuring how wrong the model is. Low = good.

For classification we use **cross-entropy loss**. Intuition: it's the penalty for being
confidently wrong. If the true label is "fight":

| Model says | Loss | |
|---|---|---|
| 99% fight | 0.01 | tiny penalty |
| 60% fight | 0.51 | moderate |
| 10% fight | 2.30 | large |
| 1% fight | 4.61 | huge — confident *and* wrong is the worst case |

That asymmetry is deliberate: it pushes the model toward being uncertain rather than
confidently wrong.

### Step 4 — backpropagation

This is the clever part, and you can treat it as a black box. It answers: *"for each of
the 87 million weights, if I increased it slightly, would the loss go up or down, and by
how much?"* The answer for each weight is its **gradient**.

Backpropagation computes all 87 million gradients efficiently in one backward sweep
through the network, using the chain rule from calculus. **You never implement this** —
`loss.backward()` does it. You just need to know that's what the line means.

### Step 5 — the optimizer and learning rate

Now every weight knows which way to move. The **optimizer** moves them.

The **learning rate** is how big a step to take, and it is *the* most important setting
in all of training:

- **Too high** → you overshoot, the loss bounces around or becomes `NaN`. Training
  "diverges".
- **Too low** → training works but takes forever, or gets stuck.
- Typical: `1e-4` (0.0001) for a fresh head, `1e-5` for a pretrained backbone.

Classic analogy: you're descending a foggy hill trying to reach the bottom. The gradient
tells you which way is downhill. The learning rate is your stride length. Huge strides
overshoot the valley; tiny ones take all day.

**Optimizers you'll hear named:**

- **SGD** — plain gradient descent. Simple, still used for some vision tasks.
- **Adam** — adapts the step size per weight. Robust, converges fast.
- **AdamW** — Adam with correct weight decay. **The default for transformers, and what
  you should use.**

**Weight decay** gently pulls weights toward zero, discouraging over-reliance on any
single one. It's a mild anti-overfitting measure. `0.05` is typical.

**Learning-rate schedule** — you lower the LR over training. Big steps early to cover
ground, small steps later to settle precisely. **Cosine annealing** (smooth decay
following a cosine curve) is the standard. **Warmup** — starting tiny for the first few
hundred steps — is common with transformers, which are unstable early.

---

## 6. Epochs, batches, and steps

These three get mixed up constantly. Precise definitions:

- **Batch** — the group of clips processed together in one forward/backward pass.
- **Step** (or **iteration**) — one batch going through steps 2–5 above. Weights update
  once per step.
- **Epoch** — one complete pass through the entire training set.

Concretely, with 1,600 training clips and batch size 8:

```
steps per epoch = 1600 / 8 = 200 steps
10 epochs       = 2,000 steps = the model sees each clip 10 times
```

### Why batches at all?

Why not one clip at a time, or all 1,600 at once?

- **One at a time** — the gradient is noisy (that single clip may be unrepresentative)
  and GPUs are idle, since they're built for parallel work.
- **All at once** — won't fit in memory. Not close.
- **A batch** — averages the gradient over several examples (less noise) and keeps the
  GPU busy. Good compromise.

### Batch size is limited by VRAM

**VRAM** is memory *on the GPU*, separate from system RAM. During training the GPU holds
the weights, the gradients, the optimizer state, **and all intermediate activations for
every clip in the batch** (needed for the backward pass). Activations dominate, and they
scale linearly with batch size.

Video is brutal here because of that T dimension — a video clip is ~16× the data of an
image. Rough figures for VideoMAE-Base at 224×224, 16 frames:

| GPU | VRAM | Realistic batch size |
|---|---|---|
| Colab T4 | 16 GB | 4–8 |
| A100 | 40/80 GB | 16–32 |
| H100 | 80 GB | 32–64 |

`CUDA out of memory` is the single most common error you will hit. Fixes, in order of
what to try: lower batch size → enable mixed precision → gradient accumulation →
gradient checkpointing → smaller model.

### Gradient accumulation

Bigger batches give more stable gradients, but you're capped by VRAM. **Gradient
accumulation** fakes a bigger batch: run 4 batches of 4, add up the gradients, then do
one weight update.

```
batch_size 4 x accum_steps 4 = effective batch 16
```

Same result as batch 16, 4× less memory, slightly slower. This is why you'll see both
`BATCH` and `ACCUM` in training scripts.

### Mixed precision (AMP)

By default, numbers are stored as 32-bit floats (**fp32**). **Mixed precision** uses
16-bit (**fp16**, or **bf16** on A100/H100) for most operations while keeping a few
sensitive ones in fp32.

Result: **~2× faster and ~half the memory**, for essentially no accuracy loss. On any
modern GPU there's no reason not to use it. On your DGX, prefer **bf16** — it has the
same numeric range as fp32 so it doesn't need the "loss scaling" workaround fp16
requires.

---

## 7. Overfitting

**This is what will actually go wrong.** Not crashes — those you can see. This one hides.

**Overfitting is when the model memorises your training examples instead of learning the
general concept.** It scores 99% on data it has seen and 60% on anything new. It has
learned *"this specific clip is a fight"* rather than *"this is what fighting looks
like"*.

Why it happens: 87 million parameters against 1,600 clips. The model has more than
enough capacity to simply memorise all of them, and memorising is *easier* than
understanding.

### How you detect it: the split

You split your data into three parts **before doing anything else**:

| Split | Share | Purpose |
|---|---|---|
| **Train** | ~70% | The model learns from these |
| **Validation** | ~15% | Checked each epoch — tells you if it's generalising |
| **Test** | ~15% | Touched **once**, at the very end |

The tell:

```
epoch 1:  train loss 0.69  val loss 0.68     both falling - learning
epoch 5:  train loss 0.31  val loss 0.35     both falling - still fine
epoch 9:  train loss 0.08  val loss 0.42     <-- train down, val UP = OVERFITTING
```

**When val loss starts rising while train loss keeps falling, you are overfitting.**
Everything after that point is getting worse, not better, no matter what the training
loss says.

### Why the test set must stay sealed

If you check val after every epoch and pick the best one, you've *tuned* on val — so val
is now slightly optimistic. The test set, looked at exactly once at the end, is your
only honest estimate. **Every time you peek at test and change something, it becomes a
little less trustworthy.** This is the single biggest credibility issue in research ML,
and a senior engineer will absolutely ask how you split your data.

### Data leakage — the CCTV-specific trap

**This is the mistake most likely to embarrass you in the meeting, so understand it
properly.**

If you split clips randomly, clips from the *same source video* land in both train and
val. The model then sees the same corridor, same lighting, same people in both. It
recognises the *scene*, not the *action*, and your val accuracy is inflated — sometimes
by 20+ points. You report 95%, deploy, and get 65%.

**The fix: split by group, not by clip.** All clips from one camera / one recording
session / one set of subjects must land entirely on one side of the split.

```
BAD:   camera01_clip03 -> train      GOOD:  all of camera01 -> train
       camera01_clip04 -> val               all of camera02 -> val
       (same corridor on both sides)        (val is genuinely unseen)
```

This is exactly why `prepare.py` in this repo does group-aware splitting by default and
warns loudly if you have too few groups. With your own footage, **group by camera and by
recording day at minimum.**

### The fixes for overfitting

1. **More data.** Always the best answer.
2. **Data augmentation.** Randomly alter each clip every time it's used — flip it,
   crop it, shift brightness, simulate night. The model never sees the exact same input
   twice, so memorising is harder. Free extra data, effectively.
3. **Early stopping.** Keep the checkpoint from the epoch with the best *val* score, not
   the last epoch.
4. **A smaller model**, or freeze more of the backbone.
5. **Weight decay / dropout.** Mild built-in regularisation.

**Augment toward your deployment conditions.** For CCTV, night/infrared simulation and
low-bitrate simulation matter far more than the standard flips — half your real incidents
happen after dark, and IR footage looks nothing like daylight footage.

---

## 8. Transfer learning

**You are not going to train a model from scratch, and you should be clear about why.**

Training a video model from random weights needs millions of clips and thousands of GPU
hours. You have hundreds of clips.

**Transfer learning:** start from a model already trained on a huge dataset, and adapt
it. A model trained on **Kinetics-400** (650,000 clips, 400 human actions) has already
learned what edges, textures, bodies, and motion look like. Those are the same features
you need. Only the final decision differs.

Terms:

- **Pretraining** — the original expensive training on the big dataset. Someone else
  already paid for this.
- **Fine-tuning** — continuing training on *your* data, from those weights.
- **Backbone** — the big feature-extracting body of the network. Keep it.
- **Head** — the final small layer producing your classes. **Replace it**, because
  Kinetics has 400 outputs and you need 2.
- **Freezing** — holding some layers fixed so they don't update. Faster, less
  overfitting, but less adaptation.

### Discriminative learning rates

You'll see the backbone and head given *different* learning rates:

```python
head: lr = 1e-4     # random weights, needs to move fast
body: lr = 1e-5     # already knows what motion looks like - nudge gently
```

The intuition matters: the head is brand new and knows nothing, so it should learn
quickly. The backbone already contains valuable knowledge, and large updates would
destroy it — that's **catastrophic forgetting**.

### Self-supervised pretraining

**VideoMAE** (Masked Autoencoder) is worth understanding because it's what we use.

It's pretrained by hiding ~90% of the video and training the model to reconstruct the
missing parts. That needs **no labels at all** — the video is its own answer key. To
fill in a hidden region you must understand objects, motion, and physics, so the model
develops a strong general representation.

Practically: **self-supervised pretraining fine-tunes better on small datasets**, which
is exactly your situation.

---

## 9. Model architectures for video

Enough to follow the conversation.

### The core problem

Images have 2 spatial dimensions. Video adds **time**. Every architecture is a different
answer to "how do we handle time?"

### CNNs (Convolutional Neural Networks)

A **convolution** slides a small filter across the image looking for a pattern (an edge,
a texture). Stack many and early layers detect edges, later ones detect object parts.

- **2D CNN + averaging** — run an image CNN on each frame, average the results. Fast,
  but **completely blind to motion** — it can't tell a person falling from a person
  standing up, because averaging destroys frame order. Useful as a baseline to prove
  motion matters.
- **3D CNN** — the filter slides through *time* as well as space, so it sees motion
  directly. **C3D**, **I3D**, **X3D**, **SlowFast**.
- **R(2+1)D** — factorises a 3D convolution into a 2D spatial one plus a 1D temporal
  one. Cheaper, often better.

**X3D** is worth knowing: designed for maximum accuracy per unit of compute, tiny
(3.8M params), the practical choice when compute is tight.

**SlowFast** has two pathways — a "slow" one at low frame rate for detail, a "fast" one
at high frame rate for motion. Elegant idea, commonly cited.

### Transformers

Newer, currently strongest. Based on **attention**: every patch of every frame can
directly relate to every other patch, rather than only its neighbours. This captures
long-range relationships a convolution cannot.

- **TimeSformer**, **MViT**, **VideoSwin**, **VideoMAE**.
- Downside: **transformers need more data**. Which is exactly why the self-supervised
  pretraining in VideoMAE matters — it supplies the data appetite without needing labels.

### Skeleton / pose based

Different approach entirely: first detect people and their joints (**pose estimation** —
YOLO-pose, RTMPose), then classify the *movement of the skeleton* rather than the pixels.

- **Massively cheaper** — a skeleton is ~17 points, not 150,000 pixels.
- **Robust** to lighting and clothing.
- **Explainable** — you can point at why it fired.
- **Best-in-class for falls**, where the geometry is unmistakable: the body's aspect
  ratio flips from tall to wide, the hip drops fast, and then stays down.
- Weaker for fights, where contact and intent matter and pose estimators struggle with
  people overlapping.

**ST-GCN** (Spatio-Temporal Graph Convolutional Network) is the standard model for
classifying skeleton sequences — it treats the body as a graph of connected joints.

### Two-stream and optical flow

**Optical flow** is a per-pixel map of motion between frames. **Two-stream** networks run
one branch on RGB and one on flow.

Historically strong; now less used because computing flow is expensive and modern 3D
models learn motion themselves. Worth recognising the term — it comes up in older papers.

### Rough comparison

| Model | Params | Speed | Data appetite | Good for |
|---|---|---|---|---|
| 2D CNN + avg | 25M | fastest | low | baseline only — motion-blind |
| X3D-M | 3.8M | fast | medium | compute-constrained |
| R(2+1)D-18 | 33M | medium | medium | solid baseline |
| SlowFast | 34M | medium | high | strong accuracy |
| VideoMAE-Base | 87M | slower | low* | **best fine-tuning on small data** |

\* low *because* of self-supervised pretraining — the architecture itself is data-hungry.

---

## 10. Measuring whether it works

**Getting this right is what separates a credible result from a meaningless one.**

### The confusion matrix

Every prediction falls into one of four buckets:

```
                    model says FIGHT      model says NORMAL
actually FIGHT      True Positive  (TP)   False Negative (FN)   <- MISSED EVENT
actually NORMAL     False Positive (FP)   True Negative  (TN)
                    ^ FALSE ALARM
```

Both errors matter, and they matter differently:

- **False Negative** — a real fight goes undetected. The system failed at its job.
- **False Positive** — a false alarm. Individually harmless, but in bulk they're what
  actually kills these systems: people stop trusting the alerts.

### Accuracy, and why it lies

```
accuracy = (TP + TN) / everything
```

**Accuracy is dangerously misleading on imbalanced data.** If 1% of your clips are
fights, a model that always predicts "normal" is **99% accurate** and completely
useless. Real CCTV is far more skewed than 1%.

**Never report accuracy alone.** A senior engineer will immediately ask about class
balance, and "99% accurate" without that context reads as naive.

### Precision and recall

```
precision = TP / (TP + FP)    "when it says fight, how often is it right?"
recall    = TP / (TP + FN)    "of all real fights, how many did it catch?"
```

They trade off against each other. Lower your threshold and you catch more real fights
(recall ↑) but raise more false alarms (precision ↓).

- **High recall, low precision** — catches everything, cries wolf constantly.
- **High precision, low recall** — when it fires you believe it, but it misses a lot.

**F1 score** is their harmonic mean — one number when you want a single summary.

### Threshold, and why AUC exists

The model outputs a probability like 0.73. **You** choose the cutoff. At 0.5 that's a
fight; at 0.8 it isn't. **The threshold is a product decision, not a model property.**

Since every threshold gives different precision/recall, we use metrics that summarise
across *all* thresholds:

- **ROC-AUC** — 0.5 is random guessing, 1.0 is perfect. Good general summary.
- **PR-AUC / Average Precision** — **better when classes are imbalanced**, which yours
  are. Prefer this one, and say so — it signals you understand the imbalance.

### For a research write-up, report

1. **PR-AUC and ROC-AUC** — threshold-independent, the honest headline.
2. **A precision/recall table across thresholds** — shows the trade-off explicitly.
3. **The confusion matrix** at your chosen threshold.
4. **Class balance** — how many fight vs normal clips. Always.
5. **How you split** — and specifically that it was group-aware.

### Cross-validation

With a small dataset, a single train/val split is luck-dependent — you might get an easy
val set. **K-fold cross-validation** splits the data K ways, trains K times with a
different fold held out each time, and averages. You report mean ± standard deviation.

It costs K× the compute, but with a few hundred clips it's the difference between a
result and an anecdote. **On a DGX you can afford it — and offering it unprompted will
land well in the meeting.**

---

## 11. Glossary

Terms you're likely to hear, alphabetically.

| Term | Meaning |
|---|---|
| **Activation** | Intermediate output inside the network |
| **AMP** | Automatic Mixed Precision — fp16/bf16 for ~2× speed, half memory |
| **Annotation** | The human labelling process |
| **AUC** | Area Under Curve — threshold-independent quality summary |
| **Augmentation** | Randomly altering training data to prevent memorisation |
| **Backbone** | The main feature-extracting body of a network |
| **Backpropagation** | Algorithm computing every weight's gradient |
| **Batch** | Group of samples processed together |
| **bf16** | brain float 16 — fp32's range at half the size. Preferred on A100/H100 |
| **Checkpoint** | Saved weights (+ optimizer state) at a point in training |
| **Class imbalance** | Unequal numbers of examples per class |
| **Clip** | Short fixed-length sequence of frames — the unit of classification |
| **Convergence** | Training has settled; loss stops improving |
| **Cross-entropy** | The standard classification loss |
| **DDP** | DistributedDataParallel — PyTorch's multi-GPU training. Use over DataParallel |
| **Diverge** | Training blows up; loss → NaN. Usually LR too high |
| **Dropout** | Randomly zeroing activations during training to reduce overfitting |
| **Epoch** | One full pass over the training set |
| **F1** | Harmonic mean of precision and recall |
| **Fine-tuning** | Continuing training a pretrained model on your data |
| **fp16 / fp32** | 16- / 32-bit floating point |
| **FLOPs** | Floating-point operations — a compute-cost measure |
| **Gradient** | Direction and size of a weight's effect on the loss |
| **Gradient accumulation** | Summing gradients over batches to fake a larger batch |
| **Gradient checkpointing** | Recompute activations instead of storing them — saves VRAM, costs time |
| **Ground truth** | The correct label |
| **Head** | Final layer(s) producing class outputs |
| **Hyperparameter** | A setting *you* choose (LR, batch size) vs. a weight the model learns |
| **Inference** | Using a trained model to predict. No learning happens |
| **Kinetics-400** | 650k-clip action dataset; the standard video pretraining source |
| **Latency** | Time for one prediction |
| **Leakage** | Train/test contamination that inflates results |
| **Learning rate (LR)** | Step size for weight updates. The most important hyperparameter |
| **Logits** | Raw model outputs before softmax |
| **Loss** | A number measuring wrongness. Lower is better |
| **MIL** | Multiple Instance Learning — training from video-level labels without timestamps |
| **Normalisation** | Rescaling inputs to a standard range |
| **Optical flow** | Per-pixel motion map between frames |
| **Optimizer** | The algorithm that updates weights (SGD, Adam, AdamW) |
| **Overfitting** | Memorising training data; poor on new data |
| **Parameters** | The learnable numbers — the "knobs". Same as weights |
| **Pose estimation** | Detecting human joint positions |
| **Precision** | Of flagged items, how many were right |
| **Pretraining** | Initial training on a large general dataset |
| **Recall** | Of real events, how many were caught. AKA sensitivity, TPR |
| **Regularisation** | Any technique that reduces overfitting |
| **Softmax** | Converts logits into probabilities summing to 1 |
| **Step / iteration** | One batch's forward + backward + update |
| **Tensor** | Multi-dimensional numeric array |
| **Throughput** | Samples processed per second |
| **Transfer learning** | Reusing knowledge from a pretrained model |
| **VRAM** | GPU memory. The usual limiting resource |
| **Warmup** | Starting with a tiny LR and ramping up |
| **Weight decay** | Penalty pulling weights toward zero |
| **Weights** | Same as parameters |

---

## 12. Questions to ask in the meeting

Asking good questions signals more competence than pretending to know things. These are
all genuinely open on this project.

**On data — ask these first, they matter most**

1. "How many hours of footage do we have, and how many actual incidents are in it?"
   *(Incident count is the real constraint. 500 hours with 12 fights is a 12-example
   problem.)*
2. "Who labels it, and what's the labelling protocol? Do we mark start/end times or just
   tag whole clips?"
3. "Can we split by camera and by recording day, so we don't leak scene identity between
   train and validation?" *(This one question will do more for your credibility than
   anything else on this list.)*
4. "How many distinct cameras and locations? Do we have night and daytime footage of
   the same events?"
5. "Are we cleared to use this footage for model training — consent, DPDP, retention?"

**On scope**

6. "Are we detecting specific named events, or open-set 'anything unusual'?" *(These are
   very different problems — see [docs/03](03-data-and-training.md). Open-set is much
   harder.)*
7. "Is the deliverable a paper, a benchmark number, or a demo?"
8. "What would count as success — a target PR-AUC, or beating a specific baseline?"

**On method**

9. "Should we start from Kinetics-pretrained VideoMAE, or is there a domain-specific
   checkpoint you'd prefer?"
10. "Do we want a pose-based branch for falls alongside the video classifier?"
11. "Given the dataset size, should we do k-fold cross-validation rather than a single
    split?"

**On infrastructure**

12. "How many GPUs on the DGX can we actually use, and is it shared or scheduled?"
13. "Is the data on fast local NVMe or on network storage?" *(This decides whether the
    dataloader or the GPU is your bottleneck — usually the dataloader.)*
14. "Do we need a cloud fallback, and are we allowed to move footage off-premises?"
    *(Often the answer is no, and that decides it.)*

**Good things to say**

- "I'd like to check the class balance before trusting any accuracy number."
- "I want to make sure we're splitting by camera, not randomly, or we'll leak scene
  identity."
- "Can we agree the test set is sealed until the end?"
- "What's our baseline? I'd like a motion-energy baseline and a frame-averaged 2D CNN
  so we can show the video model is actually earning its complexity."

That last one is genuinely good practice: **always have a dumb baseline.** If your 87M
parameter transformer barely beats motion energy, that's important to know early — and
this repo ships both baselines so you can show it.

---

**Next:** [docs/07-training-guide.md](07-training-guide.md) — the hands-on walkthrough
of actually running training on your own footage.
