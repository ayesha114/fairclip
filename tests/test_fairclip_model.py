"""
Integration test for the full FairCLIP model.
Tests that all Steps III–VIII work together correctly.
Run with: pytest tests/test_fairclip_model.py -v
"""
import pytest
import torch
from models.fairclip import FairCLIP


@pytest.fixture
def model():
    """Create a CPU FairCLIP model for testing."""
    return FairCLIP(
        model_name="ViT-B/32",
        device="cpu",
        n_bias_directions=2,
        lambda_fair_image=0.1,
        lambda_fair_text=0.1,
    )


def make_fake_embeddings(n=20, d=512):
    e = torch.randn(n, d)
    return e / e.norm(dim=-1, keepdim=True)


def make_labels(n=20):
    return torch.cat([
        torch.zeros(n // 2, dtype=torch.long),
        torch.ones(n // 2, dtype=torch.long),
    ])


class TestFairCLIPModel:

    def test_model_initializes(self, model):
        assert model is not None
        assert not model._bias_subspace_fitted

    def test_repr(self, model):
        r = repr(model)
        assert "FairCLIP" in r
        assert "ViT-B/32" in r

    def test_embedding_dim(self, model):
        assert model.get_embedding_dim() == 512

    def test_fit_bias_subspace(self, model):
        img_embs = make_fake_embeddings(20)
        txt_embs = make_fake_embeddings(2)
        img_labels = make_labels(20)
        txt_labels = torch.arange(2)

        model.fit_bias_subspace(img_embs, txt_embs, img_labels)
        assert model._bias_subspace_fitted

    def test_training_step_requires_fitted_subspace(self, model):
        """training_step should raise if bias subspace not fitted."""
        images = torch.randn(4, 3, 224, 224)
        texts = ["a photo"] * 4
        labels = torch.zeros(4, dtype=torch.long)
        with pytest.raises(RuntimeError):
            model.training_step(images, texts, labels)

    def test_encode_and_debias_after_fitting(self, model):
        """After fitting, encode_and_debias should return clean embeddings."""
        img_embs = make_fake_embeddings(20)
        txt_embs = make_fake_embeddings(2)
        img_labels = make_labels(20)

        model.fit_bias_subspace(img_embs, txt_embs, img_labels)

        test_images = torch.randn(4, 3, 224, 224)
        debiased_img, _ = model.encode_and_debias(images=test_images)

        assert debiased_img is not None
        assert debiased_img.shape == (4, 512)

        # Should be unit vectors
        norms = debiased_img.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=0.01)

    def test_save_and_load(self, model, tmp_path):
        """Save and reload should restore bias subspace."""
        img_embs = make_fake_embeddings(20)
        txt_embs = make_fake_embeddings(2)
        model.fit_bias_subspace(img_embs, txt_embs, make_labels(20))

        path = str(tmp_path / "fairclip.pt")
        model.save(path)

        model2 = FairCLIP(model_name="ViT-B/32", device="cpu", n_bias_directions=2)
        model2.load(path)

        assert model2._bias_subspace_fitted
