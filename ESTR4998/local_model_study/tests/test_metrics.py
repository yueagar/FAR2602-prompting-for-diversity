"""Metric tests, including the edge cases the research plan names explicitly."""

import math

import numpy as np
import pytest

from localstudy import metrics


def test_distinct_n_hand_calculated():
    # Pooled unigrams: [a, b, a, c] -> 3 unique / 4 total.
    # Pooled bigrams:  [(a,b), (a,c)] -> 2 unique / 2 total.
    texts = ["a b", "a c"]
    assert metrics.distinct_n(texts, 1) == pytest.approx(3 / 4)
    assert metrics.distinct_n(texts, 2) == pytest.approx(2 / 2)


def test_distinct_n_is_pooled_not_averaged():
    # Two identical responses: every unigram of the second is already seen, so a
    # set-level metric must drop below the per-response value of 1.0.
    assert metrics.distinct_n(["a b c", "a b c"], 1) == pytest.approx(3 / 6)


def test_distinct_n_length_bias_is_real():
    """Guards the caveat in the docstring rather than a bug: longer = lower."""
    short = metrics.distinct_n(["one two", "three four"], 1)
    long = metrics.distinct_n(["one two one two one", "three four three four three"], 1)
    assert short > long


def test_duplicate_outputs_score_zero_diversity():
    embeddings = np.tile(np.array([[0.6, 0.8]]), (5, 1))
    assert metrics.mean_pairwise_cosine_distance(embeddings) == pytest.approx(0.0, abs=1e-12)
    assert metrics.exact_duplicate_rate(["same", "same", "same"]) == pytest.approx(2 / 3)
    assert metrics.partition_count_proxy(embeddings, threshold=0.8) == 1


def test_empty_outputs_are_safe():
    assert metrics.distinct_n(["", ""], 2) == 0.0
    assert metrics.total_tokens(["", ""]) == 0
    assert metrics.mean_pairwise_cosine_distance(np.zeros((3, 4))) == pytest.approx(1.0)


def test_diagonal_exclusion_hand_calculated():
    # Three orthogonal unit vectors: every off-diagonal cosine is 0, so every
    # pairwise distance is 1 and the mean is exactly 1. Including the diagonal
    # would give (3 pairs * 1 + 3 zeros) / 6 = 0.5, so this pins the convention.
    embeddings = np.eye(3)
    assert metrics.mean_pairwise_cosine_distance(embeddings) == pytest.approx(1.0)

    # Two vectors 60 degrees apart: distance 1 - cos(60) = 0.5.
    pair = np.array([[1.0, 0.0], [math.cos(math.pi / 3), math.sin(math.pi / 3)]])
    assert metrics.mean_pairwise_cosine_distance(pair) == pytest.approx(0.5)


def test_single_response_set_has_no_pairs():
    assert metrics.mean_pairwise_cosine_distance(np.array([[1.0, 0.0]])) == 0.0


def test_partition_proxy_counts_near_distinct_responses():
    # Two tight clusters plus one outlier -> three classes at threshold 0.9.
    embeddings = np.array(
        [[1.0, 0.0], [0.999, 0.045], [0.0, 1.0], [0.045, 0.999], [0.707, -0.707]]
    )
    assert metrics.partition_count_proxy(embeddings, threshold=0.9) == 3
    # A threshold of 0 merges everything with non-negative similarity; a
    # threshold above 1 can never merge, so every response is its own class.
    assert metrics.partition_count_proxy(embeddings, threshold=1.01) == 5


def test_set_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="mis-pairs"):
        metrics.set_metrics(["a", "b"], np.eye(3), 0.8)


def test_degeneracy_flags():
    assert metrics.degeneracy_flags("")["empty"]
    assert metrics.degeneracy_flags("too short")["very_short"]
    assert metrics.degeneracy_flags("ha ha ha ha ha ha ha ha ha ha")["repetitive"]
    assert metrics.degeneracy_flags(
        "Here is a creative and diverse answer to your question about pies."
    )["instruction_leak"]
    clean = metrics.degeneracy_flags("A short but perfectly ordinary sentence about pie.")
    assert not any(clean.values())


def test_cosine_matrix_handles_zero_vectors():
    matrix = metrics.cosine_similarity_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == pytest.approx(1.0)
