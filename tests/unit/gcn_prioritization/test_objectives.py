import pytest

torch = pytest.importorskip("torch")

from bioGraph.gcn_prioritization.objectives import pairwise_ranking_loss  # noqa: E402


def test_pairwise_ranking_loss_matches_the_training_contract():
    positive = torch.tensor([2.0, -1.0])
    negative = torch.tensor([0.5, 1.0])

    actual = pairwise_ranking_loss(positive, negative)
    expected = torch.nn.functional.softplus(
        -(positive - negative)
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_pairwise_ranking_loss_rewards_larger_positive_margins():
    negative = torch.tensor([0.0])

    small_margin = pairwise_ranking_loss(torch.tensor([0.1]), negative)
    large_margin = pairwise_ranking_loss(torch.tensor([2.0]), negative)

    assert large_margin < small_margin
