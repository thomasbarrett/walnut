import torch

from walnut.sampler import Sampler, SamplingParams

sample = Sampler()


def test_greedy_returns_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    out = sample(logits, SamplingParams(temperature=0.0))
    assert out.shape == (1, 1)
    assert out.item() == 1


def test_greedy_ignores_generator():
    logits = torch.tensor([[0.0, 0.0, 9.0]])
    a = sample(logits, SamplingParams(temperature=0.0))
    b = sample(logits, SamplingParams(temperature=0.0))
    assert a.item() == b.item() == 2


def test_seed_makes_sampling_reproducible():
    logits = torch.randn(1, 50)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sample(logits, SamplingParams(temperature=1.0), g1)
    b = sample(logits, SamplingParams(temperature=1.0), g2)
    assert a.item() == b.item()


def test_top_k_one_forces_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(10):
        assert sample(logits, SamplingParams(temperature=1.0, top_k=1), g).item() == 1


def test_top_k_larger_than_vocab_is_clamped():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    g = torch.Generator().manual_seed(0)
    # top_k beyond the vocab size must not raise and stays in range.
    out = sample(logits, SamplingParams(temperature=1.0, top_k=100), g)
    assert out.item() in {0, 1, 2}


def test_top_p_restricts_to_top_mass():
    # A peaked distribution under a small top_p collapses to the top token.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(1)
    draws = {
        sample(logits, SamplingParams(temperature=1.0, top_p=0.5), g).item()
        for _ in range(30)
    }
    assert draws == {0}


def test_top_p_keeps_exact_nucleus():
    # softmax(log p) == p, so the token probabilities are known: [.5, .3, .15, .05].
    # Nucleus at 0.6 admits {0, 1} (cumulative .8, and .5 alone is < .6); at 0.85 it
    # additionally admits token 2 (cumulative .95) but never token 4's mass.
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    g = torch.Generator().manual_seed(0)
    kept_06 = {
        sample(logits, SamplingParams(temperature=1.0, top_p=0.6), g).item()
        for _ in range(300)
    }
    kept_085 = {
        sample(logits, SamplingParams(temperature=1.0, top_p=0.85), g).item()
        for _ in range(300)
    }
    assert kept_06 == {0, 1}
    assert kept_085 == {0, 1, 2}


def test_batched_logits_sample_per_row():
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    out = sample(logits, SamplingParams(temperature=0.0))
    assert out.shape == (2, 1)
    assert out[0].item() == 0
    assert out[1].item() == 1
