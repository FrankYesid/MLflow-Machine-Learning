# MLflow-Machine-Learning

Predicción de la demanda de alquiler de bicicletas (*Bike Sharing Demand*) con
seguimiento de experimentos en **MLflow**.

- **Problema**: regresión — predecir el total de alquileres por hora (`count`).
- **Modelo**: `DecisionTreeRegressor` (scikit-learn) con tracking de MLflow.
- **Datos**: `data/train.csv`, `data/test.csv` (Bike Sharing Demand, Kaggle).

## Inicio rápido

```powershell
python -m venv mlops-env
.\mlops-env\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Ejecuta el notebook `src/notebooks/00_fy_description.ipynb` y luego `mlflow ui`
para revisar los experimentos.

## Documentación

La documentación completa y la guía general están en
[docs/GUIA.md](docs/GUIA.md): descripción del dataset, instalación, estructura del
proyecto, flujo del pipeline, resultados y buenas prácticas.
