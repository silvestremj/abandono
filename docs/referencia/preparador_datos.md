# Preparador de Datos

Módulo de integración, limpieza y preparación de datos para machine learning.

::: src.preparador_datos.PreparadorDatos
    options:
      show_source: false
      members:
        - ejecutar_preparacion
        - preparar_dataset_ml
        - filtrar_columnas_por_bimestre

---

## Ejemplo de uso

```python
from src.preparador_datos import PreparadorDatos

preparador = PreparadorDatos(datasets)

# Tabla maestra
df_master = preparador.ejecutar_preparacion()

# Dataset para ML (One-Hot Encoding + target binario)
df_ml = preparador.preparar_dataset_ml(df_master)

# Filtrar por bimestre (para entrenamiento temporal)
df_b2 = preparador.filtrar_columnas_por_bimestre(df_ml, bimestre_corte=2)

# Es un staticmethod: también se puede llamar sin instanciar PreparadorDatos
# (útil, por ejemplo, desde EvaluadorRiesgo.ejecutar_evaluacion con bimestre_corte)
df_b2 = PreparadorDatos.filtrar_columnas_por_bimestre(df_ml, bimestre_corte=2)
```

---

## Tareas de preprocesamiento

| Tarea | Descripción |
|-------|-------------|
| Normalización | Strip, mayúsculas, eliminación de tildes (NFKD + ASCII) |
| Deduplicación | Por `(n_siu, estudio)`: prevalece el registro más **informativo** (con `estado_actual` y/o `nota` reales) y, entre varios igual de informativos, el último |
| Nulos | Estados vacíos → `"SIN REGISTRO"`, notas → `0` |
| Type Casting | Conversión de coma decimal a punto |
| One-Hot Encoding | Variables categóricas → numéricas (`drop_first=True`) |
| Filtrado temporal | Elimina columnas de bimestres futuros |

### Deduplicación: por qué "último registro" a secas no basta

`ejecutar_preparacion` ordenaba las filas duplicadas por posición y usaba `drop_duplicates(keep="last")`, asumiendo que la última fila de cada `(n_siu, estudio)` es siempre la más actualizada. Auditando los datos de origen (`LSE_Notas_Inscrip_Baja_Actual.xlsx`) se encontró que, en **~13% de los grupos duplicados**, la fila que ese criterio conservaba estaba completamente en blanco (`estado_actual` y `nota` nulos) mientras una fila **anterior** del mismo alumno tenía su desenlace real (p.ej. `"Abandono"` con nota `7.09`) — descartando silenciosamente ese historial.

Ahora, antes de deduplicar, se ordena de forma estable por si la fila tiene datos reales (`estado_actual` no nulo o `nota` numérica no nula), de modo que `keep="last"` nunca prefiera una fila vacía sobre una informativa; entre varias filas igual de informativas se sigue conservando la última, como antes.

---

## Exclusión de columnas para ML (`COLUMNAS_EXCLUIDAS_ML`)

`COLUMNAS_EXCLUIDAS_ML` es la **fuente única de verdad** de qué columnas no deben llegar al modelo, tanto en `preparar_dataset_ml` (entrenamiento) como en `src.evaluador_riesgo.preprocesar_fila_alumno` (inferencia). Agrupa varias categorías:

| Categoría | Ejemplos | Motivo |
|-----------|----------|--------|
| Identificadores / fechas | `n_siu`, `fecha`, `fecha_nacimiento` | No predictivas |
| Desenlace ya conocido | `estado_actual`, `target`, `causa_baja` | Es lo que se quiere predecir |
| Medias globales (data leakage) | `nota`, `asist`, `sd_nota`, `sd_asist` | Resumen de **todos** los bimestres a la vez — un modelo por bimestre no debe ver el promedio/desviación final del alumno, incluye información de bimestres futuros al que se está evaluando |
| Basura de Excel | `Unnamed: 28`..`Unnamed: 31` | Columnas sin nombre real generadas por celdas sueltas del Excel de origen (comentarios de texto libre) |
| Alta cardinalidad | `ocupacion`, `ciudad` | No aportan señal generalizable y explotan el One-Hot Encoding |

La función `excluir_columnas_ml(df, columnas_excluir)` aplica esta lista **sin distinguir mayúsculas/minúsculas**: como `ejecutar_preparacion` normaliza las columnas del Excel de origen a minúsculas, una entrada de `COLUMNAS_EXCLUIDAS_ML` con otra capitalización (p.ej. las columnas `Unnamed: N`, que Excel siempre genera con mayúscula inicial) dejaría de coincidir en una comparación exacta sin que ningún error lo avisara. Se detectó así, empíricamente, que `"unnamed: 30_Evaluación postgrado"` (texto suelto de una celda de Excel) había colado como variable de split real en el árbol de Bimestre 1 antes de esta corrección — y que `sd_nota`/`sd_asist` (desviación típica de las mismas medias globales `nota`/`asist`) tampoco estaban excluidas pese a ser la misma fuga de datos.
