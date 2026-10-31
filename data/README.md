# Datos

**No se versionan los datos crudos** (varios GB). Se descargan automáticamente desde el bucket público de
Google Cloud del dataset **Sen1Floods11 v1.1** (`gs://sen1floods11`) al ejecutar el notebook o
`python -m src.run_experiments`; se guardan en `data_sen1floods11/` (ignorado por git).

| Conjunto | Chips | Uso |
|---|---|---|
| Dorado `train` | 252 | Entrenamiento |
| Dorado `valid` | 89 | Selección de modelo (solo validación) |
| Dorado `test` | 90 | Evaluación en distribución |
| Dorado `bolivia` | 15 | Generalización geográfica (nunca entra al entrenamiento) |
| Débil (Otsu, `S1OtsuLabelWeak`) | 600 de 4 160 elegibles | Ablación de supervisión (`quick`) |

Etiquetas: `1` agua · `0` no agua · `-1` sin dato (se ignora en pérdida y métricas).

## `resultados/`
Tablas generadas por el notebook (modo `quick`, una semilla = 42):

| Archivo | Contenido |
|---|---|
| `ablacion.csv` | 5 experimentos: supervisión (G, G+W, W→G) y pérdida (BCE, Dice, combinada) |
| `estadistica_particiones.csv` | Prevalencia de agua y retrodispersión media (dB) por partición |
| `test_por_evento.csv` | Métricas en `test` desglosadas por evento |
| `brecha_bolivia.csv` | Test vs. Bolivia (Δ absoluto y relativo) |
| `bolivia_por_chip.csv` | Métricas por cada uno de los 15 chips de Bolivia |
| `firma_errores.csv` | Retrodispersión media de TP/TN/FP/FN (test + Bolivia) |
| `chips_analisis_fallos.csv` | Chips con mayor omisión y comisión usados en la figura 04 |
