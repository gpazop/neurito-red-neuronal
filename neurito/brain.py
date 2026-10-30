"""El cerebro de Neurito: una red neuronal pequeña escrita solo con NumPy.

Arquitectura por defecto:

    3 necesidades  ->  6 neuronas ReLU  ->  3 acciones  ->  Softmax

La red no "sabe" qué es dormir o comer: solo aprende, a partir de ejemplos,
qué acción suele corresponder a cada combinación de necesidades.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# El orden de estas tuplas es parte del contrato del modelo. Si se cambia,
# hay que volver a entrenar y actualizar la interfaz.
NEEDS = ("energia", "hambre", "aburrimiento")
ACTIONS = ("dormir", "comer", "jugar")

NEED_LABELS = {
    "energia": "Energía",
    "hambre": "Hambre",
    "aburrimiento": "Aburrimiento",
}
ACTION_LABELS = {
    "dormir": "Dormir",
    "comer": "Comer",
    "jugar": "Jugar",
}
URGENCY_LABELS = {
    "dormir": "sueño",
    "comer": "hambre",
    "jugar": "aburrimiento",
}


def relu(values: np.ndarray) -> np.ndarray:
    """ReLU: deja pasar lo positivo y convierte lo negativo en cero."""
    return np.maximum(0.0, values)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable para un lote de vectores.

    Restar el máximo de cada fila no cambia el resultado matemático, pero evita
    que ``exp`` se desborde cuando los logits son grandes.
    """

    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


class NeuralNet:
    """Perceptrón multicapa 3 -> H -> 3 entrenado con descenso de gradiente."""

    def __init__(
        self,
        n_inputs: int = 3,
        hidden: int = 6,
        n_outputs: int = 3,
        seed: int = 42,
        *,
        W1: np.ndarray | None = None,
        b1: np.ndarray | None = None,
        W2: np.ndarray | None = None,
        b2: np.ndarray | None = None,
    ) -> None:
        self.n_inputs = int(n_inputs)
        self.hidden = int(hidden)
        self.n_outputs = int(n_outputs)

        if W1 is None:
            rng = np.random.default_rng(seed)
            # Inicialización He: pensada para capas ReLU.
            self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.n_inputs), (self.n_inputs, self.hidden))
            self.b1 = np.zeros((1, self.hidden))
            self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden), (self.hidden, self.n_outputs))
            self.b2 = np.zeros((1, self.n_outputs))
        else:
            if any(value is None for value in (b1, W2, b2)):
                raise ValueError("Si se pasan pesos, hay que pasar W1, b1, W2 y b2.")
            self.W1 = np.asarray(W1, dtype=np.float64)
            self.b1 = np.asarray(b1, dtype=np.float64)
            self.W2 = np.asarray(W2, dtype=np.float64)
            self.b2 = np.asarray(b2, dtype=np.float64)
            self._validate_shapes()

    def _validate_shapes(self) -> None:
        expected = {
            "W1": (self.n_inputs, self.hidden),
            "b1": (1, self.hidden),
            "W2": (self.hidden, self.n_outputs),
            "b2": (1, self.n_outputs),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} tiene forma {actual}; se esperaba {shape}.")

    @property
    def n_params(self) -> int:
        return sum(parameter.size for parameter in (self.W1, self.b1, self.W2, self.b2))

    def forward(self, X: np.ndarray, *, cache: bool = False):
        """Devuelve probabilidades (y, opcionalmente, los valores intermedios)."""

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_inputs:
            raise ValueError(f"Se esperaban {self.n_inputs} entradas, llegaron {X.shape[1]}.")

        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probabilities = softmax(z2)

        if cache:
            return probabilities, {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probabilities}
        return probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
        """Entropía cruzada media (+ regularización L2 opcional)."""

        probabilities = self.predict_proba(X)
        y = np.asarray(y, dtype=int)
        n = len(y)
        safe = np.clip(probabilities[np.arange(n), y], 1e-12, 1.0)
        data_loss = -np.mean(np.log(safe))
        regularization = 0.5 * l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return float(data_loss + regularization)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def gradients(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> dict[str, np.ndarray]:
        """Retropropagación manual. Devuelve los gradientes de los 4 parámetros."""

        probabilities, cache = self.forward(X, cache=True)
        X = cache["X"]
        z1, a1 = cache["z1"], cache["a1"]
        y = np.asarray(y, dtype=int)
        n = len(y)

        # dL/dz2 para softmax + entropía cruzada: p - onehot.
        dz2 = probabilities.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n

        dW2 = a1.T @ dz2 + l2 * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0.0)
        dW1 = X.T @ dz1 + l2 * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 1,
            "needs": list(NEEDS[: self.n_inputs]),
            "actions": list(ACTIONS[: self.n_outputs]),
            "n_inputs": self.n_inputs,
            "hidden": self.hidden,
            "n_outputs": self.n_outputs,
        }
        if metadata:
            meta.update(metadata)
        np.savez_compressed(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            metadata=np.array(json.dumps(meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["NeuralNet", dict[str, Any]]:
        with np.load(Path(path), allow_pickle=False) as data:
            W1 = data["W1"]
            b1 = data["b1"]
            W2 = data["W2"]
            b2 = data["b2"]
            metadata = json.loads(str(data["metadata"]))
        model = cls(
            n_inputs=W1.shape[0],
            hidden=W1.shape[1],
            n_outputs=W2.shape[1],
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
        )
        return model, metadata
