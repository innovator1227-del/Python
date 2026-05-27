"""
Evaluation module for Information Retrieval effectiveness metrics.

This module provides a clean implementation of standard IR metrics
including Precision, Recall, and F1-Score for comparing retrieved
documents against a known relevance set.
"""

from typing import Dict, List


class IREvaluator:
    """Evaluator for standard Information Retrieval metrics."""

    @staticmethod
    def evaluate(retrieved_ids: List[int], relevant_ids: List[int]) -> Dict[str, float]:
        """Calculate Precision, Recall, and F1-Score for a retrieval result.

        Args:
            retrieved_ids (List[int]): Document IDs returned by the system.
            relevant_ids (List[int]): Ground truth document IDs deemed relevant.

        Returns:
            Dict[str, float]: Dictionary containing precision, recall, and f1_score.
        """
        if retrieved_ids is None or relevant_ids is None:
            raise ValueError("Retrieved IDs and relevant IDs must not be None")

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        total_retrieved = len(retrieved_set)
        total_relevant = len(relevant_set)
        relevant_retrieved = len(retrieved_set & relevant_set)

        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

        if precision + recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2.0 * precision * recall / (precision + recall)

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }
