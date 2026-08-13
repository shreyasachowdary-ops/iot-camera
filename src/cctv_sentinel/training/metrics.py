"""Classification metrics, implemented in numpy.

No sklearn dependency, and the implementations are short enough to read -- which is the
point. If you are going to defend a number in a meeting, you should be able to say
exactly how it was computed.

See docs/00-primer.md section 10 for what these mean and which to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    support: dict[str, int] = field(default_factory=dict)

    def report(self, classes: list[str] | None = None) -> str:
        positive = classes[1] if classes and len(classes) > 1 else "positive"
        negative = classes[0] if classes else "negative"
        total = self.tp + self.fp + self.tn + self.fn
        return f"""
                       predicted {positive:<10} predicted {negative:<10}
  actual {positive:<10}  {self.tp:>10} (TP)      {self.fn:>10} (FN)  <- missed
  actual {negative:<10}  {self.fp:>10} (FP)      {self.tn:>10} (TN)
                         ^ false alarms

  threshold          {self.threshold:.3f}
  accuracy           {self.accuracy:.4f}   <- ignore this if classes are imbalanced
  balanced accuracy  {self.balanced_accuracy:.4f}
  precision          {self.precision:.4f}   of {self.tp + self.fp} flagged, {self.tp} were real
  recall             {self.recall:.4f}   of {self.tp + self.fn} real events, {self.tp} caught
  f1                 {self.f1:.4f}
  ROC-AUC            {self.roc_auc:.4f}
  PR-AUC             {self.pr_auc:.4f}   <- prefer this one when imbalanced
  n                  {total}  ({', '.join(f'{k}={v}' for k, v in self.support.items())})
""".rstrip()


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney U) identity, with ties handled.

    Equivalent to the area under the ROC curve, and cheaper: it is the probability that
    a random positive outranks a random negative.
    """
    y_true = np.asarray(y_true).astype(int)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined with only one class present

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average the ranks within each group of tied scores, or ties bias the result.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and thresholds, at every distinct score."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = max(int((y_true == 1).sum()), 1)

    # Only evaluate where the score actually changes; inside a run of ties you cannot
    # separate the samples anyway.
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]

    precision = tp[idx] / np.maximum(tp[idx] + fp[idx], 1)
    recall = tp[idx] / total_pos
    return precision, recall, s_sorted[idx]


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as the step-wise sum used by sklearn's average_precision_score."""
    if int((np.asarray(y_true) == 1).sum()) == 0:
        return float("nan")
    precision, recall, _ = pr_curve(y_true, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def compute(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
    classes: list[str] | None = None,
) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    names = classes or ["negative", "positive"]
    return Metrics(
        accuracy=(tp + tn) / max(len(y_true), 1),
        balanced_accuracy=(recall + specificity) / 2,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-12),
        roc_auc=roc_auc(y_true, scores),
        pr_auc=average_precision(y_true, scores),
        threshold=threshold,
        tp=tp, fp=fp, tn=tn, fn=fn,
        support={name: int((y_true == i).sum()) for i, name in enumerate(names)},
    )


def threshold_sweep(
    y_true: np.ndarray, scores: np.ndarray,
    thresholds: list[float] | None = None,
) -> str:
    """The precision/recall trade-off, laid out.

    The threshold is a product decision, not a model property -- so report the table and
    let whoever owns the false-alarm budget pick the row.
    """
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lines = [f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"]
    for t in thresholds:
        m = compute(y_true, scores, threshold=t)
        lines.append(
            f"{t:>7.2f} {m.precision:>10.3f} {m.recall:>8.3f} {m.f1:>7.3f} "
            f"{m.tp:>5} {m.fp:>5} {m.fn:>5}"
        )
    return "\n".join(lines)


def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str = "f1") -> float:
    """Threshold maximising a metric on THIS set.

    Choose it on validation, never on test -- picking it on test means reporting the
    best of many tries as if it were a single honest estimate.
    """
    precision, recall, thresholds = pr_curve(y_true, scores)
    if metric == "f1":
        scores_at = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    elif metric == "precision":
        scores_at = precision
    elif metric == "recall":
        scores_at = recall
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return float(thresholds[int(np.argmax(scores_at))])
