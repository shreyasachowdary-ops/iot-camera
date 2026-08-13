# 04 — Pros, cons, and the things that kill projects like this

## What's genuinely in your favour

- **Falls and fights are visually distinctive.** Unlike "suspicious behaviour", these
  have real motion signatures. Fall detection via pose is nearly a solved problem.
- **Pretrained video backbones do the heavy lifting.** You are fine-tuning 2,000
  clips, not training from scratch on a million.
- **The compute story is fine.** Gates + substreams mean one ₹30k box covers 8
  cameras. This is not a datacentre project.
- **The market is real and underserved.** Most installed CCTV in India is
  record-only; nobody watches the monitors. "Alert me when something happens" is the
  actual unmet need. Schools, hospitals, factories, elder care, hostels, warehouses.
- **You don't need identity.** Detecting a fight needs no face recognition — which
  removes your largest legal and ethical exposure. Keep it that way.

## What will hurt

### 1. False alarms are the whole game

A model at 95% accuracy on a benchmark, run on 10 cameras at one window per second,
generates **43,200 windows/day**. Even a 0.1% false-positive rate is **43 false alerts
a day**. Guards mute it, and your system is now a screensaver.

This is why the design is gates → smoothing → hysteresis → cooldown → human confirm.
Each stage exists to cut this number. Budget: **< 1 false alert/camera/day.**

### 2. The domain gap

Public datasets are not your cameras. Different height, angle, lens distortion,
compression, lighting, clothing, crowd density. Expect a 20–30 point drop from
benchmark to site. The only fix is site data, and there's no shortcut.

### 3. Rare events mean you can't measure recall honestly

If real fights happen twice a month at your pilot site, you cannot get a
statistically meaningful recall number in a quarter. You will have to rely on staged
events for recall and real footage for false-alarm rate. Be upfront about this with
whoever is paying — it's a limitation of the domain, not of your work.

### 4. Night, IR, rain, occlusion

IR footage is greyscale with a totally different noise profile and blown-out
highlights near the emitter. Treat night as a separate evaluation slice; it will very
likely need its own threshold, possibly its own fine-tune.

### 5. Liability

The moment you say "notifies the authorities", you own the failure case. A missed
event where someone was harmed, and a system that claimed to detect it, is a
different conversation from a camera that only recorded. Position it as **"an
assistive alerting layer, not a replacement for supervision"** in writing, in the
contract, and in the UI. Never auto-dial emergency services without human
confirmation.

### 6. Privacy and Indian law

You are processing personal data (people's images) at scale. Under the **DPDP Act,
2023** you are likely a Data Fiduciary, or a Data Processor for your client:

- **Notice and consent / legitimate use** — surveillance in a workplace or public
  premises generally runs on notice + signage, not individual consent. Get signage up.
- **Purpose limitation** — you may use it for safety alerting. Using it for
  productivity monitoring, or training on customer footage without agreement, is a
  different purpose and needs its own basis.
- **Data minimisation and retention** — your architecture is already the best possible
  answer here: nothing is stored except alert clips. Say this loudly in sales; it is a
  genuine competitive advantage over "we upload everything to the cloud".
- **No biometrics.** Face recognition drags in a much heavier regime. Don't.
- **Processing location.** Keeping video on the edge box, on-premises, sidesteps
  cross-border transfer questions entirely. Another point for architecture (A).
- **Training on client footage requires an explicit clause in the contract.** Get it
  in writing before the pilot, or you'll have a dataset you legally can't use.

Get an actual lawyer to review before the first paid deployment. This section is
orientation, not legal advice.

### 7. Bias

Violence detectors trained on public datasets have documented failure patterns
skewing by skin tone, clothing, and crowd density. Rough play, dancing, sports, and
religious/festival gatherings routinely trip fight detectors. If the system
disproportionately flags one group at a site, that's a serious problem — for the
people affected and for you. Evaluate per-site, log every alert with its outcome, and
review the confirmed-vs-dismissed breakdown regularly.

## Build vs. buy

Worth knowing what you're competing with: Axis, Avigilon, BriefCam, Verkada, and
Ambarella-based cameras all ship some form of this, and Frigate/DeepStack cover the
open-source hobbyist end. Your wedge is not "better model" — it's price point,
retrofit onto existing Indian camera stock, on-prem privacy, and local support. Know
this before you position it.

## Honest cost sketch (pilot, one site, 8 cameras)

| Item | One-time | Monthly |
|---|---|---|
| Jetson Orin Nano 8 GB + storage + PSU | ₹35,000 | — |
| Colab Pro (training, ~3 months) | — | ₹1,000 |
| Backend VM + object storage for clips | — | ₹2,000 |
| WhatsApp Business API / SMS (DLT-registered) | ₹5,000 setup | ₹500 |
| Labelling (staged collection + annotation) | ₹25,000 | — |
| **Total** | **~₹65,000** | **~₹3,500** |

Engineering time is the real cost: budget 3–4 months of one focused person to a
credible pilot. The hardware is noise.
