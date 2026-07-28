# Evaluador de Riesgo

Módulo de inferencia de riesgo de abandono mediante modelos por hitos bimestrales con explicabilidad local (XAI).

::: src.evaluador_riesgo.EvaluadorRiesgo
    options:
      show_source: false
      members:
        - ejecutar_evaluacion

---

## Ejemplo de uso

```python
from src.evaluador_riesgo import EvaluadorRiesgo

evaluador = EvaluadorRiesgo()

# Modo Automático (por defecto): cada alumno se evalúa en su propio
# bimestre real (el último con notas registradas).
df_final = evaluador.ejecutar_evaluacion(df_master)

# Modo "Fijar bimestre de corte": simula la evaluación como si hoy fuera
# el Bimestre 3, ignorando datos posteriores para todos los alumnos.
df_simulado = evaluador.ejecutar_evaluacion(df_master, bimestre_corte=3)

# Ver resultados
df_alto = df_final[df_final["nivel_riesgo"] == "ALTO"]
print(df_alto[["n_siu", "estudio", "bimestre_evaluado", "probabilidad_abandono", "justificacion_riesgo"]])
```

`justificacion_riesgo` contiene una frase en lenguaje natural (no la salida técnica cruda del árbol), por ejemplo:

```text
[Hito B2] El alumno se clasifica en riesgo ALTO porque su asistencia en el Bimestre 2 es igual o inferior al 60% y su nota media en el Bimestre 2 es igual o inferior a 4.0.
```

Se genera para **los tres niveles de riesgo** (ALTO, MEDIO y BAJO) de todo alumno evaluado — no solo para ALTO/MEDIO. Dejar `"N/A"` para BAJO era una inconsistencia: el árbol de decisión ya resalta la ruta coloreada igual que para ALTO/MEDIO (TASK-APP-03), así que también tiene sentido explicar por qué se le clasifica en riesgo BAJO. También se genera para `SIN_DATOS_SUFICIENTES` (ver más abajo), en cuyo caso no describe una ruta del árbol sino la ausencia de datos. `"N/A"` se reserva para quien nunca llega a evaluarse (HISTÓRICO, SIN REGISTRO, o sin modelo disponible para su bimestre).

---

## Estado `SIN_DATOS_SUFICIENTES`

`preparador_datos.ejecutar_preparacion` rellena con `0.0` las columnas `nota_bN`/`asist_bN` cuando un alumno no tiene ningún registro real en `LSE_Notas_Estadistica_Bimestre.csv` para ese bimestre. El problema: `nota_b1 == 0.0` es ambiguo, puede significar tanto "sacó un cero" (un predictor de abandono genuino y fuerte) como "todavía no hay dato" — y son casos con implicaciones opuestas para el alumno.

Antes de ejecutar la inferencia sobre un alumno PRONÓSTICO, `ejecutar_evaluacion` comprueba (`EvaluadorRiesgo._sin_datos_reales_bimestre`) si `nota_bN` **y** `asist_bN` valen `0.0` simultáneamente para el bimestre detectado (`N = bimestre_evaluado`). Se verificó empíricamente sobre los datos del proyecto que ese doble-cero conjunto es una señal fiable de ausencia de registro: de los alumnos que sí tienen algún dato parcial de un bimestre, siempre hay al menos uno de los dos valores (nota o asistencia) mayor que 0.

Si se cumple, el alumno **no pasa por el modelo** (ni `predict_proba` ni explicabilidad XAI: sería inferencia sobre ruido) y se le asigna:

- `nivel_riesgo = "SIN_DATOS_SUFICIENTES"` — distinto de `ALTO`/`MEDIO`/`BAJO` y también de `NO_CALCULABLE` (que es para quien nunca llega a evaluarse, p.ej. HISTÓRICO o SIN REGISTRO; `SIN_DATOS_SUFICIENTES` es, en cambio, un alumno evaluable pero sin señal todavía).
- `probabilidad_abandono = NaN` (no se calcula).
- `bimestre_evaluado = N` (se conserva, igual que para el resto de PRONÓSTICOS).
- `justificacion_riesgo` en lenguaje natural con el mismo prefijo `[Hito BN]`, p.ej.:

  ```text
  [Hito B1] Sin datos de asistencia ni de notas registrados hasta el Bimestre 1; no se puede evaluar el riesgo de forma fiable todavía.
  ```

En la interfaz web ([Visualizador](visualizador.md)), estos alumnos se muestran en un bloque aparte ("Alumnos sin datos suficientes para evaluar") tanto en "Resultados" como en "Alertas", sin mezclarse con los bloques de riesgo ALTO/MEDIO/BAJO.

---

## Codificación de categóricas en inferencia por fila — corregida

`preprocesar_fila_alumno` codifica con One-Hot Encoding la fila de **un solo alumno** para alinearla al esquema de `columnas_bN.pkl`. Hasta esta corrección usaba `drop_first=True`, igual que `preparar_dataset_ml` en `PreparadorDatos` — pero ahí es donde estaba el problema: una fila aislada solo puede tener **un** valor por variable categórica (una sola provincia, un solo tipo de estudio...), y `pd.get_dummies(..., drop_first=True)` sobre una columna con una única categoría presente genera **0 columnas dummy** para esa variable (no hay una "segunda" categoría de la que distinguirla). El paso siguiente, que rellena con `0` las `columnas_entrenamiento` que faltan en `df_ml`, terminaba entonces poniendo a `0` la categoría **real** del alumno exactamente igual que si no la tuviera — indistinguible de cualquier otra categoría ausente.

Esto **no afecta** a `preparar_dataset_ml` (entrenamiento): opera sobre el dataset completo, con múltiples alumnos y por tanto múltiples categorías realmente presentes por variable, donde `drop_first=True` sí tiene sentido (evita la trampa de la variable dummy) y no se ha tocado.

**Fix**: `preprocesar_fila_alumno` usa ahora `drop_first=False`. El alineado final a `columnas_entrenamiento` (que descarta cualquier columna dummy sobrante y rellena con `0` las que falten) ya garantiza el resultado correcto en ambos casos: si la categoría del alumno coincide con una columna del esquema de entrenamiento, queda en `1`; si es la categoría de referencia que el entrenamiento descartó (la que `drop_first=True` eliminó *durante el entrenamiento*, sobre el dataset completo), queda en `0` en todas las dummies de esa variable — el mismo comportamiento que tendría en el dataset de entrenamiento.

### Impacto verificado sobre datos reales

De los 8 alumnos PRONÓSTICO con inferencia real (excluyendo `SIN_DATOS_SUFICIENTES`), **2 (25%) cambiaron de nivel de riesgo**:

| Alumno | Bimestre | Nivel antes | Prob. antes | Nivel después | Prob. después |
|--------|----------|--------------|--------------|-----------------|------------------|
| E1430 / MSE | 5 | BAJO | 0.000 | MEDIO | 0.552 |
| A0102 / CESE | 4 | MEDIO | 0.481 | ALTO | 0.847 |

Los otros 6 alumnos (`2966003`/MSE, `E1311`/MSE, `2974`/MSE, `E1707`/MSE, `E1420`/CESE, `A0214`/CEIA) no cambiaron de nivel — el bug sí afectaba a su codificación categórica, pero no llegaba a mover la probabilidad al otro lado de un umbral.

**Las justificaciones XAI generadas en ejecuciones anteriores a este fix pueden haber descrito incorrectamente variables categóricas** (`país`, `provincia`, `ocupación`, `tipo de estudio`) — por ejemplo, afirmando que un alumno "no es" de una categoría que en realidad sí tiene. No deben tomarse como referencia histórica fiable en ese aspecto concreto; sí lo son en cuanto a las variables numéricas (`nota_bN`, `asist_bN`, `edad`), que nunca se vieron afectadas por este bug.

---

## Parámetro `bimestre_corte` y columna `bimestre_evaluado`

`ejecutar_evaluacion` acepta un parámetro opcional `bimestre_corte: Optional[int]`:

| Valor | Modo | Comportamiento |
|-------|------|----------------|
| `None` (por defecto) | Automático | Cada alumno se autodetecta en su propio bimestre real (`_detectar_bimestre_alumno` busca desde el último bimestre hacia B1). Es la "foto de hoy": cada alumno en su propio punto. |
| `N` (1..`max_bimestre`) | Fijar bimestre de corte (simulación) | Antes de autodetectar, se recortan del alumno las columnas `nota_bM`/`asist_bM` con M > N (reutilizando [`PreparadorDatos.filtrar_columnas_por_bimestre`](preparador_datos.md), ahora un `staticmethod`), de modo que ningún alumno se autodetecte más allá del Bimestre N. |

**`bimestre_corte` actúa como techo, no como valor forzado.** Si el bimestre real de un alumno es menor que `N` (todavía no tiene datos hasta ese hito), se evalúa con su bimestre real, no con datos forzados a `N`: `preparador_datos.py` rellena con `0.0` los bimestres sin datos, así que forzar `N` simularía un suspenso falso indistinguible de "sacó un cero".

Por esto, el DataFrame devuelto incluye la columna `bimestre_evaluado` (entero nulable, `Int64`): el bimestre con el que **realmente** se evaluó a cada alumno PRONÓSTICO (mismo valor que aparece como texto en el prefijo `[Hito BN]` de `justificacion_riesgo`, pero consultable sin parsear strings). Queda en `<NA>` para los alumnos que no se evalúan (HISTÓRICO / SIN REGISTRO / sin modelo disponible).

Comparar `bimestre_evaluado` contra el `bimestre_corte` solicitado permite detectar, en modo simulación, qué alumnos se evaluaron por debajo del corte elegido — el [Visualizador](visualizador.md) lo usa para avisarlo en la pestaña "Resultados" y en el panel de justificación de "Árbol de decisión".

---

## Clasificación de alumnos

| Tipo | Condición | Evaluación |
|------|-----------|------------|
| HISTÓRICO | Estado es RECIBIDO, ABANDONO o BAJA | No se evalúa (target conocido) |
| PRONÓSTICO | Estado es CURSO o PAUSA | Se infiere riesgo con ML |
| SIN REGISTRO | Sin estado válido | No se puede calcular |

---

## Umbrales de riesgo

| Nivel | Probabilidad | Acción recomendada |
|-------|-------------|-------------------|
| ALTO | ≥ 0.70 | Seguimiento inmediato |
| MEDIO | ≥ 0.40 | Monitoreo periódico |
| BAJO | < 0.40 | Sin acción específica |

Estos umbrales solo se aplican a un alumno PRONÓSTICO cuando sí hay datos reales para su bimestre detectado. Si no los hay, se le asigna `SIN_DATOS_SUFICIENTES` en su lugar (ver [arriba](#estado-sin_datos_suficientes)) y no se calcula ninguna probabilidad.
