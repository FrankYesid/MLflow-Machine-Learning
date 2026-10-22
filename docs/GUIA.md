# Guía general y documentación

Documentación del proyecto **MLflow-Machine-Learning**: predicción de la demanda de
alquiler de bicicletas (*Bike Sharing Demand*) con seguimiento de experimentos en MLflow.

---

## 1. Descripción general

Se trabaja con datos horarios de alquiler de bicicletas a lo largo de dos años. El objetivo
es **predecir el número total de bicicletas alquiladas (`count`) en cada hora**, usando
únicamente información disponible antes del periodo de alquiler.

El proyecto entrena un modelo de regresión (`DecisionTreeRegressor`) y registra parámetros,
métricas y artefactos en MLflow para dejar trazabilidad de cada experimento.

- **Dataset**: Bike Sharing Demand (Kaggle).
- **Tipo de problema**: Regresión.
- **Variable objetivo**: `count` (total de alquileres: `casual` + `registered`).
- **Métrica principal**: RMSE (error cuadrático medio), métrica oficial de la competencia.

---

## 2. Dataset

### 2.1 Partición de los datos

- **Entrenamiento (`data/train.csv`)**: primeros **19 días de cada mes**.
- **Prueba (`data/test.csv`)**: del **día 20 al final de cada mes**.

Esta división evita fuga de información (*data leakage*), ya que el modelo nunca ve el
mismo periodo mensual que debe predecir.

### 2.2 Campos disponibles

| Campo | Descripción |
| --- | --- |
| `datetime` | Fecha + hora (timestamp horario) |
| `season` | 1 = primavera, 2 = verano, 3 = otoño, 4 = invierno |
| `holiday` | Si el día es festivo |
| `workingday` | Si el día no es fin de semana ni festivo |
| `weather` | 1: despejado · 2: niebla/nubes · 3: lluvia/nieve ligera · 4: lluvia/nieve fuerte |
| `temp` | Temperatura en Celsius |
| `atemp` | Sensación térmica en Celsius |
| `humidity` | Humedad relativa |
| `windspeed` | Velocidad del viento |
| `casual` | Alquileres de usuarios no registrados |
| `registered` | Alquileres de usuarios registrados |
| `count` | **Total de alquileres (objetivo)** |

### 2.3 Archivos de datos

- `data/train.csv` — conjunto de entrenamiento.
- `data/test.csv` — conjunto de prueba.
- `data/sampleSubmission.csv` — formato esperado de entrega (`datetime`, `count`).

---

## 3. Requisitos e instalación

- **Python 3.12.4**.
- Entorno virtual del proyecto: `mlops-env/` (ignorado por Git).
- Dependencias declaradas en [`requirements.txt`](../requirements.txt).

```powershell
# 1. Crear el entorno virtual (solo la primera vez)
python -m venv mlops-env

# 2. Activar el entorno (Windows / PowerShell)
.\mlops-env\Scripts\Activate.ps1

# 3. Instalar las dependencias
pip install -r requirements.txt
```

> Nota: el notebook instala `missingno` con `%pip install missingno` en su primera celda,
> pero ya está incluido en `requirements.txt`.

---

## 4. Estructura del proyecto

```text
MLflow-Machine-Learning/
├── data/
│   ├── train.csv               # datos de entrenamiento
│   ├── test.csv                # datos de prueba
│   └── sampleSubmission.csv    # formato de entrega
├── src/
│   └── notebooks/
│       ├── 00_fy_description.ipynb   # notebook principal (EDA + entrenamiento)
│       └── mlruns/                   # almacén local de MLflow (experimentos)
├── docs/
│   └── GUIA.md                 # este documento
├── requirements.txt            # dependencias directas
├── README.md                   # resumen del proyecto
└── LICENSE
```

---

## 5. Guía de uso paso a paso

1. **Activar el entorno** (ver sección 3).
2. **Abrir JupyterLab** desde la raíz del proyecto:

   ```powershell
   jupyter lab
   ```

3. **Ejecutar el notebook** `src/notebooks/00_fy_description.ipynb` de arriba hacia abajo.
   Cada ejecución del bloque de entrenamiento crea un nuevo *run* en MLflow.
4. **Visualizar los experimentos** con la interfaz de MLflow (desde `src/notebooks/`,
   donde se genera la carpeta `mlruns/`):

   ```powershell
   mlflow ui
   ```

   Luego abrir `http://127.0.0.1:5000` en el navegador para comparar runs, métricas y
   artefactos.

---

## 6. Flujo del pipeline (notebook)

1. **Carga de datos**: lectura de `../../data/train.csv`.
2. **Diagnóstico de nulos**: `data.isnull().sum()` y matriz de `missingno` (no hay valores nulos).
3. **Tratamiento de outliers**: recorte de `windspeed` con `clip(upper=40)`.
4. **Limpieza**: se eliminan filas con `temp` negativa.
5. **Conversión de tipos**: `datetime` a objeto `datetime`.
6. **Ingeniería de características**:
   - `is_clear_weather`: 1 si `weather == 1`.
   - `is_rainy_weather`: 1 si `weather >= 3`.
   - `is_holiday_workingday`: 1 si es festivo y laborable a la vez.
   - Se descarta la columna `datetime`.
7. **Separación de variables**: `X` = todas las columnas excepto `count`; `y` = `count`.
8. **División train/test**: 80 / 20 con `random_state=42` (8708 / 2178 filas).
9. **Entrenamiento + tracking MLflow**:
   - Modelo: `DecisionTreeRegressor(max_depth=10, random_state=42)`.
   - Se registran parámetros (`model_type`, `max_depth`), métricas (`MAE`, `RMSE`),
     el gráfico de importancia de variables y el modelo serializado.

---

## 7. Resultados del último run registrado

| Parámetro / Métrica | Valor |
| --- | --- |
| `model_type` | `DecisionTreeRegressor` |
| `max_depth` | 10 |
| `MAE` | ≈ 3.41 |
| `RMSE` | ≈ 5.85 |
| Run name | `sedate-dog-722` |

> Los resultados se guardan en `src/notebooks/mlruns/`. Ejecuta `mlflow ui` para consultarlos
> y comparar nuevas variantes del modelo.

---

## 8. Buenas prácticas y siguientes pasos

- Versionar los experimentos vía MLflow en lugar de sobrescribir el notebook.
- Probar otros modelos (Random Forest, Gradient Boosting) y comparar RMSE.
- Añadir variables temporales (hora, día de la semana, mes) a partir de `datetime`
  antes de descartarla.
- Registrar el modelo con `mlflow.register_model` para pasar de experimento a despliegue.
- Mantener `requirements.txt` actualizado ante cualquier dependencia nueva.
