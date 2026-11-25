"""Synthetic empty-input regression; no classifier, data or model loading."""
import pytest

from localstudy.functional import FunctionalJudge


@pytest.mark.parametrize("reference", [None, [], ["reference"]])
def test_empty_response_set_is_rejected_before_inference(reference):
    # No initialized model/tokenizer: reaching either would fail this test.
    judge = object.__new__(FunctionalJudge)
    with pytest.raises(ValueError, match="response set is empty"):
        judge.measure([], reference)


def test_nonempty_synthetic_measurement_retains_its_definition():
    judge = object.__new__(FunctionalJudge)
    judge.equivalent = lambda pairs: [a == b for a, b in pairs]
    judge.exceeds_window = lambda text: False
    result = judge.measure(["same", "same", "new"], ["same"])
    assert result.functional_classes == 2
    assert result.functional_labels == [0, 0, 1]
    assert result.functional_novel == [False, False, True]
    assert result.functional_novel_share == pytest.approx(1 / 3)
    assert result.functional_window_truncated_share == 0
