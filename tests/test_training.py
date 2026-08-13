"""Tests for the training pipeline.

The metrics tests check against hand-computable cases -- if you are going to defend a
number in a meeting you want the implementation pinned down, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cctv_sentinel.training import metrics
from cctv_sentinel.training.prepare import ClipRecord, assign_splits, parse_group

torch = pytest.importorskip("torch")

from cctv_sentinel.training import models  # noqa: E402  (needs torch)


# -- metrics ---------------------------------------------------------------


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics.compute(y, scores, threshold=0.5)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.pr_auc == pytest.approx(1.0)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 0, 2)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)


def test_inverted_scores_give_zero_auc():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert metrics.roc_auc(y, scores) == pytest.approx(0.0)


def test_auc_of_uninformative_ranking_is_half():
    """AUC is the chance a random positive outranks a random negative.

    Positives at 0.4 and 0.1, negatives at 0.3 and 0.2: two of the four pairs rank
    correctly, so 0.5 -- the model orders no better than a coin flip.
    """
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    assert metrics.roc_auc(y, scores) == pytest.approx(0.5)


def test_auc_counts_correctly_ordered_pairs():
    # positives 0.2, 0.4 vs negatives 0.1, 0.3 -> 3 of 4 pairs ordered correctly
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert metrics.roc_auc(y, scores) == pytest.approx(0.75)


def test_auc_handles_ties():
    """All-identical scores rank nothing, so AUC must be exactly 0.5, not 0 or 1."""
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.roc_auc(y, scores) == pytest.approx(0.5)


def test_auc_undefined_with_one_class():
    y = np.array([1, 1, 1])
    assert np.isnan(metrics.roc_auc(y, np.array([0.1, 0.5, 0.9])))
    assert np.isnan(metrics.average_precision(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])))


def test_accuracy_is_misleading_on_imbalanced_data():
    """The core reason docs say never to report accuracy alone.

    99 normals, 1 fight, model always says normal: 99% accurate, catches nothing.
    """
    y = np.array([0] * 99 + [1])
    scores = np.zeros(100)
    m = metrics.compute(y, scores, threshold=0.5)
    assert m.accuracy == pytest.approx(0.99)
    assert m.recall == pytest.approx(0.0)
    assert m.balanced_accuracy == pytest.approx(0.5)


def test_threshold_changes_precision_recall_tradeoff():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.3, 0.6, 0.7, 0.9])
    low = metrics.compute(y, scores, threshold=0.5)
    high = metrics.compute(y, scores, threshold=0.8)
    assert low.recall > high.recall          # lower threshold catches more
    assert high.precision >= low.precision   # but is right more often when it fires


def test_best_threshold_is_selected_from_the_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = metrics.best_threshold(y, scores, metric="f1")
    assert 0.3 < t <= 0.7
    assert metrics.compute(y, scores, threshold=t).f1 == pytest.approx(1.0)


def test_average_precision_matches_hand_computation():
    # ranked: 1, 0, 1, 1 -> precision at each hit = 1/1, 2/3, 3/4; recall step = 1/3 each
    y = np.array([1, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    expected = (1 / 3) * (1.0 + 2 / 3 + 3 / 4)
    assert metrics.average_precision(y, scores) == pytest.approx(expected)


def test_sweep_and_report_render():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert "precision" in metrics.threshold_sweep(y, scores)
    assert "PR-AUC" in metrics.compute(y, scores).report(["normal", "fight"])


# -- group-aware splitting -------------------------------------------------


def test_parse_group_uses_filename_prefix():
    from pathlib import Path

    assert parse_group(Path("cam01_2024-05-03_0007.mp4"), None) == "cam01"
    assert parse_group(Path("nounderscore.mp4"), None) == "nounderscore"


def test_parse_group_with_regex():
    from pathlib import Path

    assert parse_group(Path("site3-cam07-0001.mp4"), r"(cam\d+)") == "cam07"


def _records(n_groups: int, per_group: int = 10) -> list[ClipRecord]:
    from pathlib import Path

    return [
        ClipRecord(path=Path(f"g{g}_{i}.mp4"), label=i % 2,
                   label_name="fight" if i % 2 else "normal", group=f"g{g}")
        for g in range(n_groups) for i in range(per_group)
    ]


def test_no_group_appears_in_two_splits():
    """The property that makes validation meaningful. If this breaks, every reported
    number in the project is inflated."""
    records = _records(20)
    assign_splits(records, (0.7, 0.15, 0.15), seed=0)

    splits_by_group: dict[str, set[str]] = {}
    for record in records:
        splits_by_group.setdefault(record.group, set()).add(record.split)
    assert all(len(s) == 1 for s in splits_by_group.values())


def test_every_clip_is_assigned():
    records = _records(20)
    assign_splits(records, (0.7, 0.15, 0.15), seed=0)
    assert all(r.split in {"train", "val", "test"} for r in records)


def test_split_ratios_are_approximately_respected():
    records = _records(40, per_group=5)
    assign_splits(records, (0.7, 0.15, 0.15), seed=1)
    counts = {s: sum(1 for r in records if r.split == s) for s in ["train", "val", "test"]}
    total = len(records)
    assert 0.55 <= counts["train"] / total <= 0.85
    assert counts["val"] > 0 and counts["test"] > 0


def test_split_is_deterministic_given_a_seed():
    a, b = _records(15), _records(15)
    assign_splits(a, (0.7, 0.15, 0.15), seed=42)
    assign_splits(b, (0.7, 0.15, 0.15), seed=42)
    assert [r.split for r in a] == [r.split for r in b]


def test_too_few_groups_warns(caplog):
    records = _records(3)
    with caplog.at_level("WARNING"):
        assign_splits(records, (0.7, 0.15, 0.15), seed=0)
    assert "distinct groups" in caplog.text


# -- models ----------------------------------------------------------------


@pytest.mark.parametrize("arch", ["motion_baseline", "frame_average", "tiny3dcnn"])
def test_model_shapes(arch):
    """Every architecture must accept (B,C,T,H,W) and return (B,num_classes).

    One convention across the zoo is what lets training, export and inference stay
    ignorant of which backbone is loaded.
    """
    model = models.build(arch, num_classes=2)
    out = model(torch.randn(2, 3, 16, 64, 64))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        models.build("not_a_model")


def test_frame_average_is_motion_blind():
    """It sees every frame but not their ORDER -- that is the point of the baseline.

    Shuffling time must not change its output, which is exactly why the gap between it
    and a real video model measures the value of temporal modelling.
    """
    model = models.build("frame_average", num_classes=2).eval()
    clip = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        forward = model(clip)
        reversed_ = model(clip.flip(2))
    assert torch.allclose(forward, reversed_, atol=1e-5)


def test_tiny3dcnn_is_not_motion_blind():
    """The contrast that makes the baseline meaningful: a 3D CNN sees frame order."""
    model = models.build("tiny3dcnn", num_classes=2).eval()
    clip = torch.cumsum(torch.randn(1, 3, 16, 64, 64), dim=2)  # temporally structured
    with torch.no_grad():
        forward = model(clip)
        reversed_ = model(clip.flip(2))
    assert not torch.allclose(forward, reversed_, atol=1e-4)


def test_motion_baseline_scores_movement_higher():
    model = models.build("motion_baseline", num_classes=2).eval()
    still = torch.ones(1, 3, 16, 32, 32)
    moving = torch.randn(1, 3, 16, 32, 32)
    with torch.no_grad():
        assert model(moving)[0, 1] > model(still)[0, 1]


def test_param_groups_separate_head_from_backbone():
    """Discriminative LR: the head is random and must move fast; the backbone already
    knows what motion looks like and large updates destroy it."""
    model = models.build("tiny3dcnn", num_classes=2)
    groups = models.param_groups(model, head_lr=1e-3, backbone_lr=1e-5)
    assert len(groups) == 2
    assert {g["lr"] for g in groups} == {1e-3, 1e-5}
    head = next(g for g in groups if g["lr"] == 1e-3)
    assert sum(p.numel() for p in head["params"]) < 1000  # just the final Linear


# -- engine ----------------------------------------------------------------


def test_overfitting_warning_triggers_on_diverging_losses():
    from cctv_sentinel.training.engine import EpochResult, overfitting_warning

    history = [
        EpochResult(epoch=0, train_loss=0.9, val_loss=0.9),
        EpochResult(epoch=1, train_loss=0.7, val_loss=0.6),
        EpochResult(epoch=2, train_loss=0.5, val_loss=0.7),
        EpochResult(epoch=3, train_loss=0.3, val_loss=0.8),
        EpochResult(epoch=4, train_loss=0.1, val_loss=0.9),
    ]
    assert "overfitting" in (overfitting_warning(history, patience=3) or "")


def test_no_overfitting_warning_when_both_losses_fall():
    from cctv_sentinel.training.engine import EpochResult, overfitting_warning

    history = [EpochResult(epoch=i, train_loss=1.0 - i * 0.1, val_loss=1.0 - i * 0.08)
               for i in range(6)]
    assert overfitting_warning(history, patience=3) is None


def test_single_process_is_not_distributed():
    from cctv_sentinel.training import engine

    assert engine.get_rank() == 0
    assert engine.get_world_size() == 1
    assert engine.is_main()
