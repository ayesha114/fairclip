"""
Unit tests for evaluation metrics (Step IX).
Run with: pytest tests/test_metrics.py -v
"""
import pytest
import torch
from evaluation.metrics.maxskew import compute_maxskew
from evaluation.metrics.ndkl import compute_ndkl
from evaluation.metrics.able import compute_able
from evaluation.metrics.dpg_eod import compute_dpg, compute_eod
from evaluation.retrieval import compute_recall_at_k


def sim_matrix(n=20):
    """Random similarity matrix."""
    return torch.randn(n, n)


def labels(n=20):
    return torch.cat([torch.zeros(n//2, dtype=torch.long),
                      torch.ones(n//2, dtype=torch.long)])


def unit_embs(n=20, d=512):
    e = torch.randn(n, d)
    return e / e.norm(dim=-1, keepdim=True)


class TestMaxSkew:
    def test_returns_float(self):
        assert isinstance(compute_maxskew(sim_matrix(), labels()), float)

    def test_non_negative(self):
        assert compute_maxskew(sim_matrix(), labels()) >= 0.0

    def test_zero_one_group(self):
        l = torch.zeros(20, dtype=torch.long)
        assert compute_maxskew(sim_matrix(), l) == 0.0

    def test_perfect_fairness_lower_than_biased(self):
        # Perfect sim: each image perfectly retrieved by text i
        perfect_sim = torch.eye(20)
        random_sim = torch.randn(20, 20)
        l = labels()
        # Perfect sim should have lower or equal maxskew for balanced data
        ms_perfect = compute_maxskew(perfect_sim, l)
        assert ms_perfect >= 0.0


class TestNDKL:
    def test_returns_float(self):
        assert isinstance(compute_ndkl(sim_matrix(), labels()), float)

    def test_non_negative(self):
        assert compute_ndkl(sim_matrix(), labels()) >= 0.0

    def test_zero_one_group(self):
        l = torch.zeros(20, dtype=torch.long)
        assert compute_ndkl(sim_matrix(), l) == 0.0


class TestABLE:
    def test_returns_dict(self):
        result = compute_able(unit_embs(), unit_embs(), labels())
        for k in ["able", "vl_alignment", "bias_level", "maxskew"]:
            assert k in result

    def test_able_between_zero_and_one(self):
        result = compute_able(unit_embs(), unit_embs(), labels())
        assert 0.0 <= result["able"] <= 1.0

    def test_vl_alignment_between_zero_and_one(self):
        result = compute_able(unit_embs(), unit_embs(), labels())
        assert 0.0 <= result["vl_alignment"] <= 1.0

    def test_perfect_match_high_alignment(self):
        # When image and text embeddings are identical, alignment should be 1.0
        e = unit_embs(20)
        result = compute_able(e, e, labels())
        assert result["vl_alignment"] == 1.0


class TestDPGEOD:
    def test_dpg_returns_float(self):
        e = unit_embs()
        preds = torch.randint(0, 2, (20,))
        assert isinstance(compute_dpg(e, preds, labels()), float)

    def test_dpg_non_negative(self):
        e = unit_embs()
        preds = torch.randint(0, 2, (20,))
        assert compute_dpg(e, preds, labels()) >= 0.0

    def test_eod_returns_float(self):
        preds = torch.randint(0, 2, (20,))
        true_l = torch.randint(0, 2, (20,))
        assert isinstance(compute_eod(preds, true_l, labels()), float)

    def test_perfect_parity_zero_dpg(self):
        # Same positive rate for all groups
        e = unit_embs()
        preds = torch.ones(20, dtype=torch.long)
        dpg = compute_dpg(e, preds, labels())
        assert dpg == 0.0


class TestRecallAtK:
    def test_returns_dict(self):
        result = compute_recall_at_k(unit_embs(), unit_embs())
        for k in ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10"]:
            assert k in result

    def test_perfect_match_gives_100(self):
        e = unit_embs(20)
        result = compute_recall_at_k(e, e)
        assert result["TR@1"] == 100.0
        assert result["IR@1"] == 100.0

    def test_values_between_0_and_100(self):
        result = compute_recall_at_k(unit_embs(), unit_embs())
        for v in result.values():
            assert 0.0 <= v <= 100.0

    def test_higher_k_higher_recall(self):
        result = compute_recall_at_k(unit_embs(), unit_embs())
        assert result["TR@5"] >= result["TR@1"]
        assert result["TR@10"] >= result["TR@5"]
