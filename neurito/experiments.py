"""Los cinco experimentos del proyecto, con sus tablas y gráficas.

Ejecutar con:

    python -m neurito.experiments

Genera CSV y JSON en ``results/`` y PNG en ``results/`` y ``docs/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .brain import ACTION_LABELS, ACTIONS, NEEDS, NeuralNet
from .data import (
    FOUR_ACTION_LABELS,
    FOUR_ACTIONS,
    FOUR_NEEDS,
    Dataset,
    four_needs_to_urgency,
    make_dataset,
    make_four_action_dataset,
    needs_to_urgency,
    teacher_actions,
    teacher_actions_four,
)
from .metrics import (
    ambiguity_summary,
    balanced_accuracy,
    confusion_matrix,
    expected_calibration_error,
    per_class_metrics,
    reliability_bins,
    top_two_gap,
)
from .plots import (
    BLUE,
    CORAL,
    GOLD,
    GREEN,
    PALETTE,
    architecture_diagram,
    bar_chart,
    combine_vertical,
    heatmap,
    line_chart,
    probability_chart,
)
from .train import TrainConfig, train_default

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
DOCS = PROJECT_ROOT / "docs"
ACTION_NAMES = [ACTION_LABELS[action] for action in ACTIONS]
ACTION_COLORS = [GOLD, CORAL, BLUE]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_input(variable: str, values: np.ndarray, fixed: dict[str, float]) -> np.ndarray:
    X = np.empty((len(values), 3), dtype=float)
    for column, need in enumerate(NEEDS):
        X[:, column] = fixed.get(need, 0.0)
    X[:, NEEDS.index(variable)] = values
    return X


def _find_transitions(
    values: np.ndarray,
    labels: np.ndarray,
    predict_one: Callable[[float], int] | None = None,
) -> list[dict]:
    transitions = []
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            continue
        low, high = float(values[index - 1]), float(values[index])
        if predict_one is not None:
            left_label = int(labels[index - 1])
            for _ in range(30):
                middle = (low + high) / 2
                if predict_one(middle) == left_label:
                    low = middle
                else:
                    high = middle
        transitions.append(
            {
                "from": int(labels[index - 1]),
                "to": int(labels[index]),
                "value": float((low + high) / 2),
            }
        )
    return transitions


def run_sweeps(model: NeuralNet, out: Path) -> dict:
    """Experimento 1: dos necesidades fijas y una variable."""

    scenarios = [
        {
            "name": "energia",
            "variable": "energia",
            "fixed": {"hambre": 0.70, "aburrimiento": 0.45},
            "subtitle": "Hambre 0.70 y aburrimiento 0.45: el umbral teórico está en energía = 0.30.",
            "xlabel": "energía (0 = agotado, 1 = descansado)",
        },
        {
            "name": "hambre",
            "variable": "hambre",
            "fixed": {"energia": 0.75, "aburrimiento": 0.55},
            "subtitle": "Energía 0.75 y aburrimiento 0.55: el umbral teórico está en hambre = 0.55.",
            "xlabel": "hambre (0 = saciado, 1 = hambriento)",
        },
        {
            "name": "aburrimiento",
            "variable": "aburrimiento",
            "fixed": {"energia": 0.75, "hambre": 0.60},
            "subtitle": "Energía 0.75 y hambre 0.60: el umbral teórico está en aburrimiento = 0.60.",
            "xlabel": "aburrimiento (0 = entretenido, 1 = aburrido)",
        },
    ]

    rows = []
    summary: dict[str, Any] = {}
    for scenario in scenarios:
        values = np.linspace(0.0, 1.0, 1001)
        X = _build_input(scenario["variable"], values, scenario["fixed"])
        probabilities = model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        urgency = needs_to_urgency(X)
        teacher = teacher_actions(X)

        for index, value in enumerate(values):
            rows.append(
                {
                    "escenario": scenario["name"],
                    "variable": scenario["variable"],
                    "valor": round(float(value), 4),
                    "p_dormir": round(float(probabilities[index, 0]), 6),
                    "p_comer": round(float(probabilities[index, 1]), 6),
                    "p_jugar": round(float(probabilities[index, 2]), 6),
                    "prediccion": ACTIONS[predictions[index]],
                    "profesor": ACTIONS[teacher[index]],
                    "u_sueno": round(float(urgency[index, 0]), 6),
                    "u_hambre": round(float(urgency[index, 1]), 6),
                    "u_aburrimiento": round(float(urgency[index, 2]), 6),
                }
            )

        def predict_one(value: float) -> int:
            point = _build_input(scenario["variable"], np.array([value]), scenario["fixed"])
            return int(model.predict(point)[0])

        network_transitions = _find_transitions(values, predictions, predict_one)
        teacher_transitions = _find_transitions(values, teacher)
        for transition in network_transitions:
            matches = [
                item
                for item in teacher_transitions
                if item["from"] == transition["from"] and item["to"] == transition["to"]
            ]
            transition["teacher_value"] = matches[0]["value"] if matches else None
            transition["offset"] = (
                round(transition["value"] - matches[0]["value"], 4) if matches else None
            )

        summary[scenario["name"]] = {
            "variable": scenario["variable"],
            "fixed": scenario["fixed"],
            "network_transitions": network_transitions,
            "teacher_transitions": teacher_transitions,
            "expected_thresholds": [round(item["value"], 4) for item in teacher_transitions],
        }

        probability_chart(
            out / f"barrido_{scenario['name']}.png",
            title=f"Experimento 1 · varía {scenario['variable']}",
            subtitle=scenario["subtitle"],
            x=values,
            probabilities=probabilities,
            xlabel=scenario["xlabel"],
            action_names=ACTION_NAMES,
            colors=ACTION_COLORS,
            teacher_thresholds=[item["value"] for item in teacher_transitions],
        )

    _write_csv(out / "barrido_acciones.csv", rows)
    _write_json(out / "barrido_resumen.json", summary)
    combine_vertical(
        [out / f"barrido_{name}.png" for name in ("energia", "hambre", "aburrimiento")],
        DOCS / "barridos.png",
        gap=18,
    )
    return summary


def run_near_ties(model: NeuralNet, out: Path) -> dict:
    """Experimento 2: prioridades casi iguales y sensibilidad de Softmax."""

    cases = [
        {
            "name": "dos",
            "fixed": {"energia": 0.50, "aburrimiento": 0.20},
            "values": np.linspace(0.46, 0.54, 161),
            "subtitle": "Sueño y hambre compiten casi empatados; jugar queda por debajo.",
            "xlabel": "hambre",
        },
        {
            "name": "tres",
            "fixed": {"energia": 0.50, "aburrimiento": 0.50},
            "values": np.linspace(0.49, 0.51, 161),
            "subtitle": "Las tres urgencias pasan por 0.50: el empate triple vuelve la salida ambigua.",
            "xlabel": "hambre",
        },
    ]

    rows = []
    summary: dict[str, Any] = {}
    for case in cases:
        X = _build_input("hambre", case["values"], case["fixed"])
        probabilities = model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        urgency = needs_to_urgency(X)
        teacher = teacher_actions(X)
        entropy = -np.sum(
            np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)),
            axis=1,
        )
        delta = case["values"] - 0.50

        for index, value in enumerate(case["values"]):
            rows.append(
                {
                    "caso": case["name"],
                    "hambre": round(float(value), 5),
                    "delta": round(float(delta[index]), 5),
                    "p_dormir": round(float(probabilities[index, 0]), 6),
                    "p_comer": round(float(probabilities[index, 1]), 6),
                    "p_jugar": round(float(probabilities[index, 2]), 6),
                    "prediccion": ACTIONS[predictions[index]],
                    "profesor": ACTIONS[teacher[index]],
                    "entropia": round(float(entropy[index]), 6),
                }
            )

        tie_index = int(np.argmin(np.abs(delta)))
        selected = {}
        for target in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
            index = int(np.argmin(np.abs(delta - target)))
            selected[f"{target:+.3f}"] = {
                "hambre": round(float(case["values"][index]), 4),
                "p_dormir": round(float(probabilities[index, 0]), 3),
                "p_comer": round(float(probabilities[index, 1]), 3),
                "p_jugar": round(float(probabilities[index, 2]), 3),
                "accion": ACTIONS[predictions[index]],
                "entropia": round(float(entropy[index]), 3),
            }

        step = float(case["values"][1] - case["values"][0])
        max_step_change = float(np.max(np.abs(np.diff(probabilities, axis=0))))
        summary[case["name"]] = {
            "fixed": case["fixed"],
            "at_tie": {
                "p_dormir": round(float(probabilities[tie_index, 0]), 3),
                "p_comer": round(float(probabilities[tie_index, 1]), 3),
                "p_jugar": round(float(probabilities[tie_index, 2]), 3),
                "entropia": round(float(entropy[tie_index]), 3),
                "prediccion": ACTIONS[predictions[tie_index]],
            },
            "selected": selected,
            "max_probability_change_per_0.001": round(max_step_change / step * 0.001, 4),
            "probability_span_comer": round(
                float(probabilities[:, 1].max() - probabilities[:, 1].min()), 3
            ),
        }

        probability_chart(
            out / f"empate_{case['name']}.png",
            title=f"Experimento 2 · empate {'triple' if case['name'] == 'tres' else 'a dos'}",
            subtitle=case["subtitle"],
            x=case["values"],
            probabilities=probabilities,
            xlabel=case["xlabel"],
            action_names=ACTION_NAMES,
            colors=ACTION_COLORS,
            teacher_thresholds=[0.50],
        )

    combine_vertical(
        [out / "empate_dos.png", out / "empate_tres.png"],
        DOCS / "empates.png",
        gap=18,
    )
    _write_csv(out / "empates.csv", rows)
    _write_json(out / "empates_resumen.json", summary)
    return summary


def run_hidden_widths(data: Dataset, out: Path) -> dict:
    """Experimento 3: 3, 6, 10 y 20 neuronas ocultas, con 5 semillas."""

    widths = [3, 6, 10, 20]
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for width in widths:
        for seed in seeds:
            result = train_default(
                TrainConfig(hidden=width, epochs=800, lr=0.5, seed=seed, record_every=50),
                data=data,
                save=False,
                verbose=False,
            )
            metrics = result.metrics
            rows.append(
                {
                    "neuronas_ocultas": width,
                    "semilla": seed,
                    "parametros": metrics["n_params"],
                    "perdida_entrenamiento": round(metrics["train_loss_noisy"], 5),
                    "perdida_prueba": round(metrics["test_loss"], 5),
                    "precision_entrenamiento": round(metrics["train_accuracy_clean"], 4),
                    "precision_prueba": round(metrics["test_accuracy"], 4),
                    "precision_ambiguos": round(metrics["ambiguity"]["accuracy_ambiguous"], 4),
                    "ece": round(metrics["ece"], 4),
                    "segundos": round(result.seconds, 3),
                    "divergio": result.diverged,
                }
            )

    summary = {}
    for width in widths:
        subset = [row for row in rows if row["neuronas_ocultas"] == width]
        entry = {"neuronas_ocultas": width, "parametros": subset[0]["parametros"]}
        for key in (
            "perdida_entrenamiento",
            "perdida_prueba",
            "precision_entrenamiento",
            "precision_prueba",
            "precision_ambiguos",
            "ece",
            "segundos",
        ):
            values = np.array([row[key] for row in subset], dtype=float)
            entry[f"{key}_media"] = float(np.mean(values))
            entry[f"{key}_std"] = float(np.std(values))
        accuracy_values = np.array([row["precision_prueba"] for row in subset], dtype=float)
        entry["precision_min"] = float(np.min(accuracy_values))
        entry["precision_max"] = float(np.max(accuracy_values))
        entry["precision_mediana"] = float(np.median(accuracy_values))
        summary[str(width)] = entry

    _write_csv(out / "ancho_oculto.csv", rows)
    _write_json(out / "ancho_oculto_resumen.json", summary)

    categories = [str(width) for width in widths]
    accuracy_values = [summary[str(width)]["precision_prueba_media"] for width in widths]
    accuracy_errors = [summary[str(width)]["precision_prueba_std"] for width in widths]
    loss_values = [summary[str(width)]["perdida_prueba_media"] for width in widths]
    loss_errors = [summary[str(width)]["perdida_prueba_std"] for width in widths]

    accuracy_path = bar_chart(
        out / "ancho_oculto_precision.png",
        title="Experimento 3 · precisión de prueba",
        subtitle="Media y desviación estándar de 5 semillas. El azar es 33.3%.",
        categories=categories,
        series=[
            {
                "name": "precisión de prueba",
                "values": accuracy_values,
                "errors": accuracy_errors,
                "color": GREEN,
            }
        ],
        ylabel="precisión",
        y_max=1.0,
        value_format="{:.1%}",
        legend=False,
    )
    loss_path = bar_chart(
        out / "ancho_oculto_perdida.png",
        title="Experimento 3 · pérdida de prueba",
        subtitle="Menos pérdida no siempre significa mejor comportamiento en los casos límite.",
        categories=categories,
        series=[
            {
                "name": "pérdida de prueba",
                "values": loss_values,
                "errors": loss_errors,
                "color": CORAL,
            }
        ],
        ylabel="entropía cruzada",
        value_format="{:.3f}",
        legend=False,
    )
    combine_vertical([accuracy_path, loss_path], DOCS / "ancho_oculto.png", gap=18)
    return summary


def _status(result, metrics: dict[str, Any]) -> str:
    if result.diverged or not np.isfinite(metrics["test_loss"]):
        return "inestable"
    if metrics["test_accuracy"] < 0.50:
        return "inestable"
    if metrics["test_accuracy"] < 0.92:
        return "se estanca"
    return "mejora"


def run_training_schedule(data: Dataset, out: Path) -> dict:
    """Experimento 4: épocas y tasa de aprendizaje."""

    learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    lr_rows = []
    lr_series = []
    for lr in learning_rates:
        result = train_default(
            TrainConfig(hidden=6, epochs=800, lr=lr, seed=42, record_every=10),
            data=data,
            save=False,
            verbose=False,
        )
        metrics = result.metrics
        status = _status(result, metrics)
        lr_rows.append(
            {
                "tasa_aprendizaje": lr,
                "epocas": 800,
                "perdida_entrenamiento": round(metrics["train_loss_noisy"], 5),
                "perdida_prueba": round(metrics["test_loss"], 5),
                "precision_prueba": round(metrics["test_accuracy"], 4),
                "precision_ambiguos": round(metrics["ambiguity"]["accuracy_ambiguous"], 4),
                "divergio": result.diverged,
                "estado": status,
                "segundos": round(result.seconds, 3),
            }
        )
        lr_series.append(
            {
                "name": f"lr = {lr:g}",
                "x": result.history["epoch"],
                "y": result.history["train_loss"],
                "color": PALETTE[len(lr_series) % len(PALETTE)],
            }
        )

    lr_path = line_chart(
        out / "tasa_aprendizaje.png",
        title="Experimento 4 · tasa de aprendizaje",
        subtitle="Pérdida de entrenamiento con 6 neuronas ocultas y 800 épocas.",
        series=lr_series,
        xlabel="época",
        ylabel="pérdida de entrenamiento",
        y_min=0.0,
        y_max=1.25,
        x_min=0,
        x_max=800,
        y_decimals=2,
        x_decimals=0,
        annotation="lr = 5 no llega a converger; lr = 0.01 avanza demasiado despacio",
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lr_path, DOCS / "tasa_aprendizaje.png")

    epochs_list = [50, 100, 200, 400, 800, 1600]
    epoch_rows = []
    for epochs in epochs_list:
        runs = []
        for seed in (0, 1, 2):
            result = train_default(
                TrainConfig(
                    hidden=6,
                    epochs=epochs,
                    lr=0.5,
                    seed=seed,
                    record_every=max(1, epochs // 20),
                ),
                data=data,
                save=False,
                verbose=False,
            )
            runs.append(result)
        accuracies = np.array([run.metrics["test_accuracy"] for run in runs])
        losses = np.array([run.metrics["test_loss"] for run in runs])
        epoch_rows.append(
            {
                "epocas": epochs,
                "precision_media": round(float(np.mean(accuracies)), 4),
                "precision_std": round(float(np.std(accuracies)), 4),
                "perdida_media": round(float(np.mean(losses)), 5),
                "perdida_std": round(float(np.std(losses)), 5),
                "precision_ambiguos_media": round(
                    float(np.mean([run.metrics["ambiguity"]["accuracy_ambiguous"] for run in runs])),
                    4,
                ),
            }
        )
    accuracy_path = line_chart(
        out / "epocas_precision.png",
        title="Experimento 4 · efecto de las épocas",
        subtitle="Media de 3 semillas; las barras de error están en el CSV.",
        series=[
            {
                "name": "precisión de prueba",
                "x": [row["epocas"] for row in epoch_rows],
                "y": [row["precision_media"] for row in epoch_rows],
                "color": GREEN,
            }
        ],
        xlabel="épocas",
        ylabel="precisión de prueba",
        y_min=0.0,
        y_max=1.0,
        x_min=0,
        x_max=1700,
        y_decimals=2,
        x_decimals=0,
        markers=True,
        annotation="con 50 épocas la red aún no aprendió; después la mejora se aplana",
    )
    loss_path = line_chart(
        out / "epocas_perdida.png",
        title="Experimento 4 · pérdida frente a épocas",
        subtitle="La pérdida de prueba baja y luego se estanca; no se observa sobreajuste fuerte.",
        series=[
            {
                "name": "pérdida de prueba",
                "x": [row["epocas"] for row in epoch_rows],
                "y": [row["perdida_media"] for row in epoch_rows],
                "color": CORAL,
            }
        ],
        xlabel="épocas",
        ylabel="pérdida de prueba",
        y_min=0.0,
        y_max=0.75,
        x_min=0,
        x_max=1700,
        y_decimals=2,
        x_decimals=0,
        markers=True,
        annotation="con 1600 épocas la pérdida ya casi no cambia",
    )
    combine_vertical([accuracy_path, loss_path], DOCS / "epocas.png", gap=18)

    _write_csv(out / "tasa_aprendizaje.csv", lr_rows)
    _write_csv(out / "epocas.csv", epoch_rows)
    _write_json(
        out / "entrenamiento_resumen.json",
        {"tasa_aprendizaje": lr_rows, "epocas": epoch_rows},
    )
    return {"tasa_aprendizaje": lr_rows, "epocas": epoch_rows}


def run_calibration(model: NeuralNet, data: Dataset, out: Path) -> dict:
    """Precisión global frente a precisión en los casos ambiguos."""

    probabilities = model.predict_proba(data.X_test)
    predictions = np.argmax(probabilities, axis=1)
    gap = top_two_gap(data.urgency_test)
    correct = predictions == data.y_test

    margin_edges = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]
    margin_labels = ["<0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", ">0.20"]
    margin_rows = []
    for lower, upper, label in zip(margin_edges[:-1], margin_edges[1:], margin_labels):
        mask = (gap >= lower) & (gap < upper)
        margin_rows.append(
            {
                "margen": label,
                "casos": int(np.sum(mask)),
                "precision": round(float(np.mean(correct[mask])), 4) if np.any(mask) else 0.0,
            }
        )

    bins = reliability_bins(probabilities, data.y_test, n_bins=10)
    bins_for_plot = [item for item in bins if item["count"] > 0]
    ece = expected_calibration_error(probabilities, data.y_test, n_bins=10)

    majority = np.ones(1000, dtype=int)
    majority[:100] = 0
    majority[100:200] = 2
    constant = np.ones(1000, dtype=int)
    majority_accuracy = float(np.mean(constant == majority))
    majority_balanced = balanced_accuracy(majority, constant, 3)
    majority_per_class = per_class_metrics(majority, constant, 3)

    summary = {
        "precision_global": round(float(np.mean(correct)), 4),
        "precision_ambiguos": round(
            float(np.mean(correct[gap < 0.05])) if np.any(gap < 0.05) else 0.0, 4
        ),
        "fraccion_ambiguos": round(float(np.mean(gap < 0.05)), 4),
        "ece": round(float(ece), 4),
        "margenes": margin_rows,
        "bins_confianza": bins,
        "ejemplo_desbalance": {
            "clase_mayoritaria": "comer",
            "proporcion": 0.8,
            "precision_clasificador_constante": round(majority_accuracy, 4),
            "precision_balanceada": round(majority_balanced, 4),
            "recall_por_clase": [round(row["recall"], 4) for row in majority_per_class],
            "explicacion": (
                "Un clasificador que siempre dice 'comer' acierta el 80% si esa clase "
                "es el 80% del test, pero tiene 0% de recall en las otras dos."
            ),
        },
    }

    reliability_path = bar_chart(
        out / "calibracion_fiabilidad.png",
        title="Calibración · confianza vs. acierto",
        subtitle=f"ECE = {ece:.3f}. Una barra alta de confianza no garantiza acertar.",
        categories=[f"{item['lower']:.1f}-{item['upper']:.1f}" for item in bins_for_plot],
        series=[
            {
                "name": "confianza media",
                "values": [item["confidence"] for item in bins_for_plot],
                "color": BLUE,
            },
            {
                "name": "precisión real",
                "values": [item["accuracy"] for item in bins_for_plot],
                "color": GREEN,
            },
        ],
        ylabel="proporción",
        y_max=1.0,
        value_format="{:.0%}",
        legend_side="left",
    )
    margin_path = bar_chart(
        out / "calibracion_margenes.png",
        title="Precisión según el margen entre prioridades",
        subtitle="El margen es la diferencia entre la urgencia más alta y la segunda.",
        categories=[row["margen"] for row in margin_rows],
        series=[
            {
                "name": "precisión",
                "values": [row["precision"] for row in margin_rows],
                "color": CORAL,
            }
        ],
        ylabel="precisión",
        y_max=1.0,
        value_format="{:.0%}",
        legend=False,
    )
    combine_vertical([reliability_path, margin_path], DOCS / "calibracion.png", gap=18)

    _write_csv(out / "calibracion_margenes.csv", margin_rows)
    _write_json(out / "calibracion.json", summary)
    return summary


def run_fourth_action(out: Path) -> dict:
    """Experimento 5: qué haría falta para una cuarta acción (bañar)."""

    data = make_four_action_dataset(n=4000, seed=7, label_noise=0.05)
    result = train_default(
        TrainConfig(hidden=10, epochs=800, lr=0.5, seed=42, record_every=50),
        data=data,
        save=False,
        verbose=False,
    )
    probabilities = result.model.predict_proba(data.X_test)
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(data.y_test, predictions, 4)
    metrics = per_class_metrics(data.y_test, predictions, 4)

    sample_rows = []
    for index in range(10):
        values = data.X_test[index]
        sample_rows.append(
            {
                "energia": round(float(values[0]), 3),
                "hambre": round(float(values[1]), 3),
                "aburrimiento": round(float(values[2]), 3),
                "suciedad": round(float(values[3]), 3),
                "sueno": round(float(1 - values[0]), 3),
                "urgencia_hambre": round(float(values[1]), 3),
                "urgencia_aburrimiento": round(float(values[2]), 3),
                "urgencia_suciedad": round(float(values[3]), 3),
                "etiqueta": FOUR_ACTIONS[int(data.y_test[index])],
                "prediccion": FOUR_ACTIONS[int(predictions[index])],
            }
        )

    summary = {
        "necesidades": list(FOUR_NEEDS),
        "acciones": list(FOUR_ACTIONS),
        "n_entrenamiento": int(len(data.X_train)),
        "n_validacion": int(len(data.X_val)),
        "n_prueba": int(len(data.X_test)),
        "parametros": result.model.n_params,
        "precision_prueba": round(result.metrics["test_accuracy"], 4),
        "perdida_prueba": round(result.metrics["test_loss"], 4),
        "precision_por_clase": [
            {"accion": FOUR_ACTIONS[index], **{key: round(value, 4) for key, value in row.items()}}
            for index, row in enumerate(metrics)
        ],
        "matriz_confusion": matrix.tolist(),
        "ejemplos": sample_rows,
        "que_hace_falta": [
            "Una cuarta entrada: suciedad (0 = limpio, 1 = muy sucio).",
            "Una cuarta salida en la capa final y una neurona más de capacidad.",
            "Etiquetas para cada combinación de las cuatro necesidades; no basta con inventar la acción.",
            "Ejemplos donde la higiene compita cerca con sueño, hambre y aburrimiento.",
            "Una animación nueva para bañar y una forma de actualizar la necesidad de higiene.",
            "Volver a entrenar desde cero: la capa de salida cambia de 3 a 4 neuronas.",
        ],
    }

    _write_csv(out / "cuarta_accion_ejemplos.csv", sample_rows)
    _write_json(out / "cuarta_accion.json", summary)
    heatmap(
        out / "cuarta_accion_matriz.png",
        title="Experimento 5 · cuarta acción: bañar",
        subtitle=f"Precisión de prueba {result.metrics['test_accuracy']:.1%} con 4 necesidades y 4 acciones.",
        matrix=matrix,
        row_labels=[FOUR_ACTION_LABELS[action] for action in FOUR_ACTIONS],
        col_labels=[FOUR_ACTION_LABELS[action] for action in FOUR_ACTIONS],
        row_title="realidad",
        col_title="predicción",
    )
    return summary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown_summary(summary: dict[str, Any], out: Path) -> Path:
    main = summary["principal"]
    lines = [
        "# Resultados de los experimentos",
        "",
        "Generado automáticamente por `python -m neurito.experiments`.",
        "",
        "## Modelo principal",
        "",
        f"- Arquitectura: 3 → 6 (ReLU) → 3 (Softmax), {main['n_params']} parámetros.",
        f"- Precisión de prueba: **{_pct(main['test_accuracy'])}**.",
        f"- Pérdida de prueba: **{main['test_loss']:.4f}**.",
        f"- ECE: **{main['ece']:.4f}**.",
        f"- Casos casi empatados: {_pct(main['ambiguity']['fraction_ambiguous'])} del test; "
        f"precisión allí: **{_pct(main['ambiguity']['accuracy_ambiguous'])}**.",
        "",
        "## Experimento 1 · dos necesidades fijas",
        "",
        "| Escenario | Cambio de acción | Umbral de la red | Umbral del profesor | Desfase |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in summary["barridos"].items():
        teacher_by_pair = {
            (transition["from"], transition["to"]): transition["value"]
            for transition in item["teacher_transitions"]
        }
        network_by_pair = {
            (transition["from"], transition["to"]): transition["value"]
            for transition in item["network_transitions"]
        }
        ordered_pairs = list(teacher_by_pair) + [
            pair for pair in network_by_pair if pair not in teacher_by_pair
        ]
        for pair in ordered_pairs:
            teacher_value = teacher_by_pair.get(pair)
            network_value = network_by_pair.get(pair)
            offset = (
                network_value - teacher_value
                if teacher_value is not None and network_value is not None
                else None
            )
            if teacher_value is not None and network_value is not None:
                lines.append(
                    f"| {name} | {ACTIONS[pair[0]]} → {ACTIONS[pair[1]]} | "
                    f"{network_value:.3f} | {teacher_value:.3f} | {offset:+.3f} |"
                )
            elif network_value is not None:
                lines.append(
                    f"| {name} | {ACTIONS[pair[0]]} → {ACTIONS[pair[1]]} | "
                    f"{network_value:.3f} | — | — |"
                )
            else:
                lines.append(
                    f"| {name} | {ACTIONS[pair[0]]} → {ACTIONS[pair[1]]} | — | "
                    f"{teacher_value:.3f} | — |"
                )

    lines += [
        "",
        "## Experimento 2 · prioridades casi iguales",
        "",
        "| Caso | p(dormir) | p(comer) | p(jugar) | entropía | acción |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in summary["empates"].items():
        tie = item["at_tie"]
        lines.append(
            f"| {name} | {tie['p_dormir']:.3f} | {tie['p_comer']:.3f} | "
            f"{tie['p_jugar']:.3f} | {tie['entropia']:.3f} | {ACTION_LABELS[tie['prediccion']]} |"
        )

    lines += [
        "",
        "## Experimento 3 · neuronas ocultas",
        "",
        "| Neuronas | Parámetros | Precisión (media ± desv.) | Rango de precisión | Pérdida (media ± desv.) | Precisión en ambiguos |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width, item in summary["anchos"].items():
        lines.append(
            f"| {width} | {item['parametros']} | "
            f"{_pct(item['precision_prueba_media'])} ± {item['precision_prueba_std'] * 100:.1f} pp | "
            f"{_pct(item['precision_min'])}–{_pct(item['precision_max'])} | "
            f"{item['perdida_prueba_media']:.3f} ± {item['perdida_prueba_std']:.3f} | "
            f"{_pct(item['precision_ambiguos_media'])} |"
        )

    lines += [
        "",
        "## Experimento 4 · épocas y tasa de aprendizaje",
        "",
        "| lr | Precisión | Pérdida de prueba | Estado |",
        "|---:|---:|---:|---|",
    ]
    for row in summary["entrenamiento"]["tasa_aprendizaje"]:
        lines.append(
            f"| {row['tasa_aprendizaje']:g} | {_pct(row['precision_prueba'])} | "
            f"{row['perdida_prueba']:.4f} | {row['estado']} |"
        )

    lines += [
        "",
        "| Épocas | Precisión media | Pérdida media |",
        "|---:|---:|---:|",
    ]
    for row in summary["entrenamiento"]["epocas"]:
        lines.append(
            f"| {row['epocas']} | {_pct(row['precision_media'])} | {row['perdida_media']:.4f} |"
        )

    lines += [
        "",
        "## Experimento 5 · cuarta acción",
        "",
        f"- Precisión con 4 acciones: **{_pct(summary['cuarta']['precision_prueba'])}**.",
        f"- Parámetros: {summary['cuarta']['parametros']}.",
        f"- Datos: {summary['cuarta']['n_entrenamiento']} entrenamiento, "
        f"{summary['cuarta']['n_validacion']} validación, {summary['cuarta']['n_prueba']} prueba.",
        "",
        "## Calibración",
        "",
        f"- Precisión global: **{_pct(summary['calibracion']['precision_global'])}**.",
        f"- Precisión en ambiguos: **{_pct(summary['calibracion']['precision_ambiguos'])}**.",
        f"- ECE: **{summary['calibracion']['ece']:.4f}**.",
        "",
    ]
    path = out / "RESULTADOS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ejecuta los experimentos de Neurito.")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("[1/6] Entrenando el modelo principal...")
    data = make_dataset(n=3000, seed=2026, label_noise=0.05)
    main_result = train_default(TrainConfig(), data=data, save=True, verbose=False)
    architecture_diagram(main_result.model, DOCS / "arquitectura.png")
    print(
        f"      precisión de prueba: {main_result.metrics['test_accuracy']:.2%} | "
        f"pérdida: {main_result.metrics['test_loss']:.4f}"
    )

    print("[2/6] Experimento 1: barridos de necesidades...")
    sweeps = run_sweeps(main_result.model, out)

    print("[3/6] Experimento 2: prioridades casi iguales...")
    ties = run_near_ties(main_result.model, out)

    print("[4/6] Experimento 3: 3, 6, 10 y 20 neuronas ocultas...")
    widths = run_hidden_widths(data, out)

    print("[5/6] Experimento 4: épocas y tasa de aprendizaje...")
    training = run_training_schedule(data, out)

    print("[6/6] Calibración, cuarta acción y resumen...")
    calibration = run_calibration(main_result.model, data, out)
    fourth = run_fourth_action(out)

    summary = {
        "principal": main_result.metrics,
        "config_principal": asdict(main_result.config),
        "segundos_principales": round(main_result.seconds, 3),
        "barridos": sweeps,
        "empates": ties,
        "anchos": widths,
        "entrenamiento": training,
        "calibracion": calibration,
        "cuarta": fourth,
    }
    _write_json(out / "resumen.json", summary)
    markdown_path = write_markdown_summary(summary, out)
    elapsed = time.perf_counter() - started
    print(f"\nListo en {elapsed:.1f} s.")
    print(f"Resumen: {markdown_path}")
    print(f"Gráficas: {DOCS}")


if __name__ == "__main__":
    main()
