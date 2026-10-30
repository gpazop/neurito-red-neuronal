"""Entrenamiento, evaluación y guardado del modelo de Neurito."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .brain import ACTIONS, NEEDS, NeuralNet
from .data import Dataset, make_dataset
from .metrics import (
    ambiguity_summary,
    confusion_matrix,
    expected_calibration_error,
    per_class_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "neurito.npz"
METADATA_PATH = PROJECT_ROOT / "models" / "neurito_metadata.json"

# Hiperparámetros del modelo que se usa en la mascota.
DEFAULT_EPOCHS = 800
DEFAULT_LR = 0.5
DEFAULT_HIDDEN = 6
DEFAULT_L2 = 1e-4
DEFAULT_NOISE = 0.05


@dataclass
class TrainConfig:
    hidden: int = DEFAULT_HIDDEN
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    l2: float = DEFAULT_L2
    seed: int = 42
    label_noise: float = DEFAULT_NOISE
    n_samples: int = 3000
    data_seed: int = 2026
    record_every: int = 1


@dataclass
class TrainResult:
    model: NeuralNet
    data: Dataset
    config: TrainConfig
    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    diverged: bool = False


def fit(
    model: NeuralNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
    record_every: int = 1,
    verbose: bool = False,
) -> tuple[dict[str, list[float]], bool]:
    """Descenso de gradiente de lote completo.

    Es deliberadamente simple: no hay Adam ni mini-lotes. Así se ve con claridad
    qué hace la tasa de aprendizaje y cuándo el entrenamiento se vuelve inestable.
    """

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    diverged = False
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        gradients = model.gradients(X_train, y_train, l2=l2)
        previous = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())

        model.W1 -= lr * gradients["W1"]
        model.b1 -= lr * gradients["b1"]
        model.W2 -= lr * gradients["W2"]
        model.b2 -= lr * gradients["b2"]

        if not all(
            np.all(np.isfinite(parameter))
            for parameter in (model.W1, model.b1, model.W2, model.b2)
        ):
            model.W1, model.b1, model.W2, model.b2 = previous
            diverged = True
            break

        should_record = epoch == 1 or epoch == epochs or epoch % record_every == 0
        if should_record:
            train_loss = model.loss(X_train, y_train, l2=l2)
            train_accuracy = model.accuracy(X_train, y_train)
            if X_val is not None and y_val is not None:
                val_loss = model.loss(X_val, y_val, l2=0.0)
                val_accuracy = model.accuracy(X_val, y_val)
            else:
                val_loss = float("nan")
                val_accuracy = float("nan")
            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_accuracy"].append(float(train_accuracy))
            history["val_accuracy"].append(float(val_accuracy))

            if verbose and (epoch == 1 or epoch % max(1, epochs // 8) == 0):
                print(
                    f"época {epoch:4d} | pérdida {train_loss:7.4f} | "
                    f"val {val_loss:7.4f} | precisión {train_accuracy:6.2%}"
                )

    if verbose:
        print(f"Entrenamiento terminado en {time.perf_counter() - started:.2f} s.")
    return history, diverged


def evaluate(model: NeuralNet, data: Dataset, l2: float = 0.0) -> dict[str, Any]:
    """Métricas completas sobre las tres particiones."""

    probabilities_test = model.predict_proba(data.X_test)
    predictions_test = np.argmax(probabilities_test, axis=1)
    matrix = confusion_matrix(data.y_test, predictions_test, data.n_classes)

    metrics: dict[str, Any] = {
        "n_params": model.n_params,
        "train_loss_noisy": model.loss(data.X_train, data.y_train, l2=l2),
        "train_loss_clean": model.loss(data.X_train, data.y_train_clean),
        "train_accuracy_noisy": model.accuracy(data.X_train, data.y_train),
        "train_accuracy_clean": model.accuracy(data.X_train, data.y_train_clean),
        "val_loss": model.loss(data.X_val, data.y_val),
        "val_accuracy": model.accuracy(data.X_val, data.y_val),
        "test_loss": model.loss(data.X_test, data.y_test),
        "test_accuracy": model.accuracy(data.X_test, data.y_test),
        "test_error": float(1.0 - model.accuracy(data.X_test, data.y_test)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class_metrics(data.y_test, predictions_test, data.n_classes),
        "ece": expected_calibration_error(probabilities_test, data.y_test, n_bins=10),
    }
    metrics["ambiguity"] = ambiguity_summary(
        probabilities_test, data.y_test, data.urgency_test, margin=0.05
    )
    return metrics


def train_default(
    config: TrainConfig | None = None,
    *,
    data: Dataset | None = None,
    save: bool = True,
    verbose: bool = True,
) -> TrainResult:
    config = config or TrainConfig()
    if data is None:
        data = make_dataset(
            n=config.n_samples,
            seed=config.data_seed,
            label_noise=config.label_noise,
        )
    model = NeuralNet(
        n_inputs=data.n_inputs,
        hidden=config.hidden,
        n_outputs=data.n_classes,
        seed=config.seed,
    )
    started = time.perf_counter()
    history, diverged = fit(
        model,
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        epochs=config.epochs,
        lr=config.lr,
        l2=config.l2,
        record_every=config.record_every,
        verbose=verbose,
    )
    seconds = time.perf_counter() - started
    metrics = evaluate(model, data, l2=config.l2)

    result = TrainResult(
        model=model,
        data=data,
        config=config,
        history=history,
        metrics=metrics,
        seconds=seconds,
        diverged=diverged,
    )

    if save:
        metadata = {
            "actions": list(data.actions),
            "needs": list(data.needs),
            "config": asdict(config),
            "metrics": metrics,
            "seconds": seconds,
            "diverged": diverged,
            "description": "Cerebro de Neurito: 3 necesidades -> ReLU -> 3 acciones.",
        }
        model.save(MODEL_PATH, metadata)
        METADATA_PATH.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return result


def print_report(result: TrainResult) -> None:
    metrics = result.metrics
    print()
    print("=" * 62)
    print("  NEURITO · RESUMEN DEL ENTRENAMIENTO")
    print("=" * 62)
    print(f"  Arquitectura       : 3 -> {result.config.hidden} (ReLU) -> 3 (Softmax)")
    print(f"  Épocas / lr        : {result.config.epochs} / {result.config.lr}")
    print(f"  Ruido de etiquetas : {result.config.label_noise:.0%}")
    print(f"  Parámetros         : {metrics['n_params']}")
    print(f"  Pérdida prueba     : {metrics['test_loss']:.4f}")
    print(f"  Precisión prueba   : {metrics['test_accuracy']:.2%}")
    print(f"  Precisión val      : {metrics['val_accuracy']:.2%}")
    print(f"  ECE                : {metrics['ece']:.4f}")
    ambiguity = metrics["ambiguity"]
    print(
        f"  Casos casi empatados: {ambiguity['fraction_ambiguous']:.1%} del test, "
        f"precisión {ambiguity['accuracy_ambiguous']:.2%}"
    )
    print("-" * 62)
    print("  Matriz de confusión (filas = realidad, columnas = predicción)")
    print("                 " + "".join(f"{action:>10}" for action in ACTIONS))
    for index, action in enumerate(ACTIONS):
        row = "".join(f"{value:>10d}" for value in metrics["confusion_matrix"][index])
        print(f"  {action:<14}{row}")
    print("=" * 62)
    if result.diverged:
        print("  ¡Atención! El entrenamiento divergió y se restauró el último paso válido.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena el cerebro de Neurito.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = TrainConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        label_noise=args.noise,
        n_samples=args.samples,
    )
    result = train_default(config, verbose=not args.quiet)
    print_report(result)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
