"""Datos sintéticos y etiquetas para entrenar a Neurito.

El "profesor" del proyecto es una política de prioridades muy simple:

    urgencia de dormir = 1 - energía
    urgencia de comer  = hambre
    urgencia de jugar  = aburrimiento

La etiqueta es la acción con mayor urgencia. Los empates exactos se resuelven
con el orden dormir > comer > jugar (el primero que aparece en ``ACTIONS``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import ACTIONS, NEEDS

FOUR_NEEDS = ("energia", "hambre", "aburrimiento", "suciedad")
FOUR_ACTIONS = ("dormir", "comer", "jugar", "banar")
FOUR_ACTION_LABELS = {
    "dormir": "Dormir",
    "comer": "Comer",
    "jugar": "Jugar",
    "banar": "Bañar",
}


@dataclass(frozen=True)
class Dataset:
    """Particiones ya separadas y sus metadatos."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    urgency_train: np.ndarray
    urgency_val: np.ndarray
    urgency_test: np.ndarray
    y_train_clean: np.ndarray
    needs: tuple[str, ...] = NEEDS
    actions: tuple[str, ...] = ACTIONS
    label_noise: float = 0.0

    @property
    def n_classes(self) -> int:
        return len(self.actions)

    @property
    def n_inputs(self) -> int:
        return len(self.needs)

    def counts(self, split: str = "train") -> dict[str, int]:
        y = getattr(self, f"y_{split}")
        return {action: int(np.sum(y == index)) for index, action in enumerate(self.actions)}


def needs_to_urgency(X: np.ndarray) -> np.ndarray:
    """Convierte necesidades [energía, hambre, aburrimiento] en urgencias."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Se esperaba una matriz de forma (n, 3).")
    return np.column_stack((1.0 - X[:, 0], X[:, 1], X[:, 2]))


def four_needs_to_urgency(X: np.ndarray) -> np.ndarray:
    """Igual que la anterior, añadiendo la urgencia de bañarse."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError("Se esperaba una matriz de forma (n, 4).")
    return np.column_stack((1.0 - X[:, 0], X[:, 1], X[:, 2], X[:, 3]))


def teacher_actions(X: np.ndarray) -> np.ndarray:
    """Etiqueta del profesor para las tres acciones originales."""

    return np.argmax(needs_to_urgency(X), axis=1)


def teacher_actions_four(X: np.ndarray) -> np.ndarray:
    """Etiqueta del profesor cuando existe la necesidad de higiene."""

    return np.argmax(four_needs_to_urgency(X), axis=1)


def _split_indices(n: int, seed: int, val_fraction: float, test_fraction: float):
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Las fracciones de validación y prueba deben estar entre 0 y 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validación + prueba deben sumar menos de 1.")

    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_idx = order[:n_test]
    val_idx = order[n_test : n_test + n_val]
    train_idx = order[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _add_label_noise(
    y_clean: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    n_classes: int,
) -> np.ndarray:
    """Cambia algunas etiquetas por otra clase (ruido de etiqueta)."""

    y = y_clean.copy()
    if noise <= 0.0:
        return y
    flip = rng.random(len(y)) < noise
    offsets = rng.integers(1, n_classes, size=int(np.sum(flip)))
    y[flip] = (y_clean[flip] + offsets) % n_classes
    return y


def make_dataset(
    n: int = 3000,
    *,
    seed: int = 2026,
    label_noise: float = 0.05,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dataset:
    """Genera necesidades uniformes y etiquetas del profesor.

    El ruido se aplica solo a las etiquetas de entrenamiento, para simular un
    cuidador que a veces se equivoca. Validación y prueba conservan la política
    limpia, de modo que miden qué tanto aprendió la red y no cuánto memorizó.
    """

    if n < 30:
        raise ValueError("Se necesitan al menos 30 ejemplos.")
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise debe estar en [0, 1).")

    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    urgency = needs_to_urgency(X)
    y_clean = np.argmax(urgency, axis=1)

    train_idx, val_idx, test_idx = _split_indices(n, seed, val_fraction, test_fraction)
    y_train_clean = y_clean[train_idx].copy()
    y_train = _add_label_noise(y_train_clean, rng, label_noise, n_classes=3)

    return Dataset(
        X_train=X[train_idx],
        y_train=y_train,
        X_val=X[val_idx],
        y_val=y_clean[val_idx],
        X_test=X[test_idx],
        y_test=y_clean[test_idx],
        urgency_train=urgency[train_idx],
        urgency_val=urgency[val_idx],
        urgency_test=urgency[test_idx],
        y_train_clean=y_train_clean,
        label_noise=label_noise,
    )


def make_four_action_dataset(
    n: int = 4000,
    *,
    seed: int = 2026,
    label_noise: float = 0.05,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dataset:
    """Versión de juguete con una cuarta necesidad (suciedad) y acción (bañar)."""

    if n < 40:
        raise ValueError("Se necesitan al menos 40 ejemplos.")
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4))
    urgency = four_needs_to_urgency(X)
    y_clean = np.argmax(urgency, axis=1)

    train_idx, val_idx, test_idx = _split_indices(n, seed, val_fraction, test_fraction)
    y_train_clean = y_clean[train_idx].copy()
    y_train = _add_label_noise(y_train_clean, rng, label_noise, n_classes=4)

    return Dataset(
        X_train=X[train_idx],
        y_train=y_train,
        X_val=X[val_idx],
        y_val=y_clean[val_idx],
        X_test=X[test_idx],
        y_test=y_clean[test_idx],
        urgency_train=urgency[train_idx],
        urgency_val=urgency[val_idx],
        urgency_test=urgency[test_idx],
        y_train_clean=y_train_clean,
        needs=FOUR_NEEDS,
        actions=FOUR_ACTIONS,
        label_noise=label_noise,
    )


def example_rows(X: np.ndarray, actions: tuple[str, ...], limit: int = 12) -> list[dict]:
    """Filas legibles para documentar el formato de los datos."""

    rows = []
    for values in X[:limit]:
        if len(values) == 3:
            names = NEEDS
            urgency = needs_to_urgency(values.reshape(1, -1))[0]
        else:
            names = FOUR_NEEDS
            urgency = four_needs_to_urgency(values.reshape(1, -1))[0]
        label = int(np.argmax(urgency))
        rows.append(
            {
                **{name: round(float(value), 3) for name, value in zip(names, values)},
                "accion": actions[label],
                **{
                    f"u_{actions[i]}": round(float(urgency[i]), 3)
                    for i in range(len(actions))
                },
            }
        )
    return rows
