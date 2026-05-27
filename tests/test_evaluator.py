from core.evaluator import IREvaluator


def test_evaluator_returns_correct_metrics_for_perfect_match() -> None:
    metrics = IREvaluator.evaluate([1, 2, 3], [1, 2, 3])
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0


def test_evaluator_returns_zero_for_no_relevant_documents() -> None:
    metrics = IREvaluator.evaluate([1, 2, 3], [])
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['f1_score'] == 0.0


def test_evaluator_computes_partial_retrieval() -> None:
    metrics = IREvaluator.evaluate([1, 2, 4], [1, 2, 3])
    assert metrics['precision'] == 0.6667
    assert metrics['recall'] == 0.6667
    assert metrics['f1_score'] == 0.6667


def test_evaluator_handles_duplicate_ids() -> None:
    metrics = IREvaluator.evaluate([1, 1, 2, 2], [1, 2])
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0
