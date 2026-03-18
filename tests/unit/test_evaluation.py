"""Tests for QAC evaluation — metrics and comparison."""
import numpy as np
import pytest

from qac.evaluation import (
    ClassificationResult,
    ComparisonResult,
    compare_results,
    compute_metrics,
)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestCompareResults:
    def _make_result(self, model_type, accuracy):
        return ClassificationResult(
            model_type=model_type,
            model_path="test",
            model_hash="abc",
            accuracy=accuracy,
            f1_weighted=accuracy,
            f1_macro=accuracy,
            precision=accuracy,
            recall=accuracy,
            confusion_matrix=[[1, 0], [0, 1]],
            training_time_s=1.0,
            inference_time_s=0.1,
            y_pred=np.array([0, 1]),
            hyperparameters={},
            dataset_metadata={},
        )

    def test_quantum_advantage_detected(self):
        results = [
            self._make_result("svm", 0.80),
            self._make_result("vqc", 0.85),
        ]
        comp = compare_results(results)
        assert comp.quantum_advantage is True
        assert comp.accuracy_delta > 0

    def test_no_quantum_advantage(self):
        results = [
            self._make_result("svm", 0.95),
            self._make_result("vqc", 0.70),
        ]
        comp = compare_results(results)
        assert comp.quantum_advantage is False
        assert comp.accuracy_delta < 0

    def test_best_model_identified(self):
        results = [
            self._make_result("svm", 0.80),
            self._make_result("vqc", 0.85),
        ]
        comp = compare_results(results)
        assert comp.best_model.model_type == "vqc"

    def test_to_dict(self):
        results = [
            self._make_result("svm", 0.80),
            self._make_result("vqc", 0.85),
        ]
        comp = compare_results(results)
        d = comp.to_dict()
        assert "comparison_table" in d
        assert len(d["comparison_table"]) == 2
        assert d["quantum_advantage"] is True
