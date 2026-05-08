"""
Unit Tests for Steps VI, VII, VIII
Run with: pytest tests/test_losses.py -v
"""
import pytest
import torch
from losses.fairness_regularizer import InfoNCELoss, GroupVariancePenalty, FairnessAwareLoss
from models.adaptive_temperature import AdaptiveTemperatureController
from models.bias_removal import BiasRemover


# ── Helpers ──────────────────────────────────────────────────────────────────

def random_embs(n=16, d=512):
    e = torch.randn(n, d)
    return e / e.norm(dim=-1, keepdim=True)

def gender_labels(n=16):
    return torch.cat([torch.zeros(n//2, dtype=torch.long),
                      torch.ones(n//2, dtype=torch.long)])


# ── InfoNCE ───────────────────────────────────────────────────────────────────

class TestInfoNCE:
    def test_returns_scalar(self):
        loss = InfoNCELoss()(random_embs(), random_embs())
        assert loss.shape == ()

    def test_positive(self):
        loss = InfoNCELoss()(random_embs(), random_embs())
        assert loss.item() > 0

    def test_identical_pairs_lower_loss(self):
        e = random_embs()
        loss_matched = InfoNCELoss()(e, e).item()
        loss_random  = InfoNCELoss()(e, random_embs()).item()
        assert loss_matched < loss_random


# ── GroupVariancePenalty ──────────────────────────────────────────────────────

class TestGroupVariancePenalty:
    def test_returns_scalar(self):
        p = GroupVariancePenalty()(random_embs(), gender_labels())
        assert p.shape == ()

    def test_zero_when_one_group(self):
        labels = torch.zeros(16, dtype=torch.long)
        p = GroupVariancePenalty()(random_embs(), labels)
        assert p.item() == 0.0

    def test_higher_for_separated_groups(self):
        d = 512
        # Groups far apart
        e_far = torch.cat([
            torch.ones(8, d),
            -torch.ones(8, d),
        ])
        e_far = e_far / e_far.norm(dim=-1, keepdim=True)

        # Groups close together
        e_close = random_embs(16, d)

        p_far   = GroupVariancePenalty()(e_far, gender_labels()).item()
        p_close = GroupVariancePenalty()(e_close, gender_labels()).item()
        assert p_far > p_close

    def test_ignores_minus_one_labels(self):
        labels = gender_labels()
        labels[:4] = -1
        p = GroupVariancePenalty()(random_embs(), labels)
        assert p.item() >= 0.0


# ── FairnessAwareLoss ─────────────────────────────────────────────────────────

class TestFairnessAwareLoss:
    def test_returns_tensor_and_dict(self):
        fn = FairnessAwareLoss(lambda_fair_image=0.1, lambda_fair_text=0.1)
        loss, components = fn(random_embs(), random_embs(), gender_labels())
        assert loss.shape == ()
        for k in ["infonce", "fairness_image", "fairness_text", "total"]:
            assert k in components

    def test_lambda_zero_equals_infonce(self):
        img, txt = random_embs(), random_embs()
        labels = gender_labels()
        loss_fair, _ = FairnessAwareLoss(0.0, 0.0)(img, txt, labels)
        loss_clip    = InfoNCELoss()(img, txt)
        assert abs(loss_fair.item() - loss_clip.item()) < 1e-5

    def test_higher_lambda_higher_loss(self):
        img, txt = random_embs(), random_embs()
        labels = gender_labels()
        loss_low, _  = FairnessAwareLoss(0.01, 0.01)(img, txt, labels)
        loss_high, _ = FairnessAwareLoss(1.0,  1.0 )(img, txt, labels)
        assert loss_high.item() >= loss_low.item()

    def test_gradients_flow(self):
        img = random_embs().requires_grad_(True)
        txt = random_embs()
        loss, _ = FairnessAwareLoss()(img, txt, gender_labels())
        loss.backward()
        assert img.grad is not None


# ── AdaptiveTemperatureController ────────────────────────────────────────────

class TestAdaptiveTemperature:
    def test_returns_float(self):
        ctrl = AdaptiveTemperatureController()
        tau = ctrl(random_embs(), gender_labels())
        assert isinstance(tau, float)

    def test_within_bounds(self):
        ctrl = AdaptiveTemperatureController(tau_min=0.01, tau_max=0.5)
        for _ in range(10):
            tau = ctrl(random_embs(), gender_labels())
            assert ctrl.tau_min <= tau <= ctrl.tau_max

    def test_higher_disparity_higher_tau(self):
        ctrl = AdaptiveTemperatureController(alpha=1.0)
        d = 512

        # Embeddings with HIGH group separation
        far = torch.cat([torch.ones(8, d), -torch.ones(8, d)])
        far = far / far.norm(dim=-1, keepdim=True)
        tau_far = ctrl(far, gender_labels())

        ctrl.reset_history()

        # Embeddings with LOW group separation
        close = random_embs()
        tau_close = ctrl(close, gender_labels())

        assert tau_far >= tau_close

    def test_alpha_zero_equals_base(self):
        ctrl = AdaptiveTemperatureController(tau_base=0.07, alpha=0.0)
        tau = ctrl(random_embs(), gender_labels())
        assert abs(tau - 0.07) < 1e-6

    def test_history_tracked(self):
        ctrl = AdaptiveTemperatureController()
        for _ in range(5):
            ctrl(random_embs(), gender_labels())
        assert len(ctrl.temperature_history) == 5

    def test_reset_clears_history(self):
        ctrl = AdaptiveTemperatureController()
        ctrl(random_embs(), gender_labels())
        ctrl.reset_history()
        assert len(ctrl.temperature_history) == 0


# ── BiasRemover ───────────────────────────────────────────────────────────────

class TestBiasRemover:
    def _make_remover(self, d=512, k=3):
        B = torch.randn(d, k)
        B = B / B.norm(dim=0, keepdim=True)
        remover = BiasRemover()
        remover.set_bias_subspace(B, B)
        return remover

    def test_output_shape(self):
        r = self._make_remover()
        out = r.remove_image_bias(random_embs())
        assert out.shape == (16, 512)

    def test_output_normalized(self):
        r = self._make_remover()
        out = r.remove_image_bias(random_embs())
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=0.01)

    def test_reduces_bias_component(self):
        d, k = 256, 2
        B = torch.zeros(d, k)
        B[0, 0] = 1.0   # bias direction = dimension 0
        B[1, 1] = 1.0   # bias direction = dimension 1
        remover = BiasRemover()
        remover.set_bias_subspace(B, B)

        e = torch.randn(8, d)
        e = e / e.norm(dim=-1, keepdim=True)
        debiased = remover.remove_image_bias(e)

        # After projection, components along bias directions should be ~0
        assert debiased[:, 0].abs().max().item() < 0.01
        assert debiased[:, 1].abs().max().item() < 0.01

    def test_forward_debiases_both(self):
        r = self._make_remover()
        img, txt = random_embs(), random_embs()
        d_img, d_txt = r(img, txt)
        assert d_img.shape == img.shape
        assert d_txt.shape == txt.shape

    def test_raises_before_set(self):
        remover = BiasRemover()
        with pytest.raises(RuntimeError):
            remover.remove_image_bias(random_embs())
