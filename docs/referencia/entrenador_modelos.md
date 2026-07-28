# Entrenador de Modelos

Módulo de entrenamiento de árboles de decisión por hitos bimestrales.

::: src.entrenador_modelos.EntrenadorModelos
    options:
      show_source: false
      members:
        - entrenar_y_guardar

---

## Ejemplo de uso

```python
from src.entrenador_modelos import EntrenadorModelos
from src.preparador_datos import PreparadorDatos

preparador = PreparadorDatos(datasets)
df_master = preparador.ejecutar_preparacion()
df_ml = preparador.preparar_dataset_ml(df_master)

entrenador = EntrenadorModelos()
df_metricas = entrenador.entrenar_y_guardar(df_ml, preparador)
print(df_metricas[["Hito", "Accuracy", "Recall (Abandono)"]])
```

---

## Artefactos generados

El entrenador guarda en la carpeta `modelos/` los siguientes archivos:

| Archivo | Contenido |
|---------|-----------|
| `arbol_b1.pkl` ... `arbol_b6.pkl` | Modelo DecisionTreeClassifier por bimestre |
| `columnas_b1.pkl` ... `columnas_b6.pkl` | Lista ordenada de columnas del modelo |
| `metricas.pkl` | DataFrame con métricas comparativas |

---

## Limitación conocida: hojas de 1-2 muestras (corregida con `min_samples_leaf=3`)

### Causa estructural: histórico de entrenamiento reducido

El histórico de alumnos con desenlace conocido (HISTÓRICO: `RECIBIDO`/`ABANDONO`/`BAJA`) tiene solo **97 registros en total**, del cual cada modelo bimestral entrena con un split 80/20 (~77 en train, ~20 en test). `DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")` no fijaba `min_samples_leaf` ni `min_samples_split` (por defecto, 1 y 2 respectivamente), así que con tan pocas muestras el árbol podía — y de hecho lo hacía — aislar alumnos individuales en hojas propias en vez de encontrar patrones generalizables.

Auditando los 6 modelos entrenados sin esa restricción se encontró que **entre el 8% y el 58% de las hojas de cada árbol tenían 1 o 2 muestras**. Una hoja con 1 muestra predice con 100% de confianza (`gini=0.0`) basándose en un solo alumno histórico. Caso real detectado: un alumno con notas y asistencia buenas en los cuatro bimestres (todas las notas > 7, asistencia > 75%) fue clasificado en riesgo ALTO con probabilidad 1.0 exclusivamente porque su vector de características caía en una hoja de una sola muestra del árbol de Bimestre 4.

### Fix aplicado: `min_samples_leaf=3`

Tras un análisis de sensibilidad se fijó `min_samples_leaf=3`: es el valor mínimo que garantiza **estructuralmente** que ninguna hoja tenga menos de 3 muestras (`min_samples_leaf=2` seguiría permitiendo hojas de exactamente 2). Valores mayores (5, 8) se descartaron por empeorar visiblemente el poder discriminativo dado el tamaño del histórico.

Métricas antes (`min_samples_leaf` sin fijar, equivalente a 1) y después, recalculadas sobre los mismos datos (ya sin la fuga de `sd_nota`/`sd_asist`/columnas `Unnamed: N` corregida en `PreparadorDatos`):

| Hito | Accuracy antes | Accuracy después | Recall antes | Recall después | Hojas antes (total / ≤2 muestras) | Hojas después (total / ≤2 muestras) |
|------|-----------------|-------------------|---------------|------------------|-------------------------------------|----------------------------------------|
| B1 | 0.65 | 0.55 | 0.14 | 0.14 | 13 / 6 (46.2%) | 10 / 0 (0.0%) |
| B2 | 0.70 | 0.70 | 0.43 | 0.43 | 10 / 3 (30.0%) | 9 / 0 (0.0%) |
| B3 | 0.70 | 0.75 | 0.57 | 0.57 | 17 / 9 (52.9%) | 12 / 0 (0.0%) |
| B4 | 0.75 | 0.60 | 0.57 | 0.43 | 12 / 5 (41.7%) | 10 / 0 (0.0%) |
| B5 | 0.75 | 0.60 | 0.29 | 0.43 | 12 / 6 (50.0%) | 10 / 0 (0.0%) |
| B6 | 0.75 | 0.60 | 0.29 | 0.43 | 12 / 5 (41.7%) | 9 / 0 (0.0%) |
| **Promedio** | **0.717** | **0.633** | **0.382** | **0.405** | — | — |

El coste en Accuracy media es moderado (−0.084) y el Recall medio mejora ligeramente (+0.023) — coherente con lo esperado de reducir el sobreajuste: el modelo deja de "acertar de memoria" casos individuales del train y generaliza algo mejor sobre el test, a costa de algunos aciertos triviales que dependían de memorizar una muestra concreta. En **todos** los bimestres, el porcentaje de hojas con ≤2 muestras pasa a **0.0%** (verificado con `tests/test_entrenador.py::test_ninguna_hoja_tiene_menos_de_min_samples_leaf`, que falla explícitamente si una hoja queda por debajo de 3 muestras en cualquier reentrenamiento futuro).

El alumno del caso detectado (notas/asistencia buenas en los cuatro bimestres) pasa de una hoja de 1 muestra con probabilidad 1.0 (ALTO) a una hoja de **25 muestras** con probabilidad 0.0 (BAJO) en el árbol de Bimestre 4 — la clasificación ahora tiene respaldo estadístico real.

### Esto no es un óptimo definitivo

`min_samples_leaf=3` es el valor **mínimo** que elimina el problema, elegido por eso — no por ser el que maximiza ninguna métrica. Con solo 97 alumnos etiquetados, cualquier valor de `min_samples_leaf` es un compromiso entre robustez estadística y capacidad de discriminación, y las métricas por bimestre (sobre un test de ~20 alumnos) son sensibles a pocos aciertos/fallos. A medida que el histórico de alumnos con desenlace conocido crezca en cursos futuros, este valor debería reevaluarse: más datos de entrenamiento permiten, previsiblemente, sostener un `min_samples_leaf` mayor (más robusto frente a outliers) sin penalizar tanto la Accuracy.
