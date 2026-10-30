"""Métricas que van más allá de la precisión global."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    rows = []
    for index in range(n_classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(matrix[index, :].sum()),
            }
        )
    return rows


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [row["recall"] for row in per_class_metrics(y_true, y_pred, n_classes)]
    return float(np.mean(recalls))


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: diferencia media entre confianza y acierto, ponderada por bin."""

    confidence = np.max(probabilities, axis=1)
    correct = (np.argmax(probabilities, axis=1) == np.asarray(y_true, dtype=int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        ece += np.sum(mask) / len(y_true) * abs(np.mean(correct[mask]) - np.mean(confidence[mask]))
    return float(ece)


def reliability_bins(
    probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    confidence = np.max(probabilities, axis=1)
    correct = (np.argmax(probabilities, axis=1) == np.asarray(y_true, dtype=int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(np.sum(mask)),
                "confidence": float(np.mean(confidence[mask])) if np.any(mask) else 0.0,
                "accuracy": float(np.mean(correct[mask])) if np.any(mask) else 0.0,
            }
        )
    return bins


def top_two_gap(urgency: np.ndarray) -> np.ndarray:
    """Diferencia entre la urgencia más alta y la segunda."""

    ordered = np.sort(np.asarray(urgency, dtype=float), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def ambiguity_summary(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    urgency: np.ndarray,
    margin: float = 0.05,
) -> dict[str, float]:
    """Precisión global frente a precisión en casos casi empatados."""

    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == np.asarray(y_true, dtype=int)
    gap = top_two_gap(urgency)
    ambiguous = gap < margin
    return {
        "accuracy": float(np.mean(correct)),
        "accuracy_ambiguous": float(np.mean(correct[ambiguous])) if np.any(ambiguous) else 0.0,
        "fraction_ambiguous": float(np.mean(ambiguous)),
        "accuracy_clear": float(np.mean(correct[~ambiguous])) if np.any(~ambiguous) else 0.0,
        "margin": float(margin),
    }


def entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)
