# Resultados de los experimentos

Generado automáticamente por `python -m neurito.experiments`.

## Modelo principal

- Arquitectura: 3 → 6 (ReLU) → 3 (Softmax), 45 parámetros.
- Precisión de prueba: **95.3%**.
- Pérdida de prueba: **0.1949**.
- ECE: **0.1039**.
- Casos casi empatados: 15.1% del test; precisión allí: **69.1%**.

## Experimento 1 · dos necesidades fijas

| Escenario | Cambio de acción | Umbral de la red | Umbral del profesor | Desfase |
|---|---|---:|---:|---:|
| energia | dormir → comer | 0.287 | 0.300 | -0.013 |
| hambre | jugar → comer | 0.539 | 0.550 | -0.010 |
| aburrimiento | comer → jugar | 0.604 | 0.601 | +0.003 |

## Experimento 2 · prioridades casi iguales

| Caso | p(dormir) | p(comer) | p(jugar) | entropía | acción |
|---|---:|---:|---:|---:|---|
| dos | 0.435 | 0.516 | 0.049 | 0.851 | Comer |
| tres | 0.314 | 0.352 | 0.335 | 1.098 | Comer |

## Experimento 3 · neuronas ocultas

| Neuronas | Parámetros | Precisión (media ± desv.) | Rango de precisión | Pérdida (media ± desv.) | Precisión en ambiguos |
|---:|---:|---:|---:|---:|---:|
| 3 | 24 | 76.2% ± 24.1 pp | 37.6%–95.8% | 0.486 ± 0.362 | 56.2% |
| 6 | 45 | 96.8% ± 1.0 pp | 95.1%–98.0% | 0.202 ± 0.014 | 78.8% |
| 10 | 73 | 97.5% ± 0.7 pp | 96.4%–98.7% | 0.190 ± 0.005 | 83.5% |
| 20 | 143 | 97.1% ± 0.1 pp | 96.9%–97.3% | 0.189 ± 0.005 | 80.9% |

## Experimento 4 · épocas y tasa de aprendizaje

| lr | Precisión | Pérdida de prueba | Estado |
|---:|---:|---:|---|
| 0.01 | 61.3% | 0.9116 | se estanca |
| 0.05 | 90.2% | 0.4537 | se estanca |
| 0.1 | 96.7% | 0.2553 | mejora |
| 0.5 | 95.3% | 0.1949 | mejora |
| 1 | 94.7% | 0.1851 | mejora |
| 2 | 83.3% | 0.3499 | se estanca |
| 5 | 37.6% | 1.0970 | inestable |

| Épocas | Precisión media | Pérdida media |
|---:|---:|---:|
| 50 | 83.1% | 0.6179 |
| 100 | 90.8% | 0.3920 |
| 200 | 94.2% | 0.2611 |
| 400 | 95.6% | 0.2204 |
| 800 | 96.7% | 0.2072 |
| 1600 | 96.7% | 0.2014 |

## Experimento 5 · cuarta acción

- Precisión con 4 acciones: **97.2%**.
- Parámetros: 94.
- Datos: 2800 entrenamiento, 600 validación, 600 prueba.

## Calibración

- Precisión global: **95.3%**.
- Precisión en ambiguos: **69.1%**.
- ECE: **0.1039**.
