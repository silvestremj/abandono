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

Se genera para **los tres niveles de riesgo** (ALTO, MEDIO y BAJO) de todo alumno evaluado — no solo para ALTO/MEDIO. Dejar `"N/A"` para BAJO era una inconsistencia: el árbol de decisión ya resalta la ruta coloreada igual que para ALTO/MEDIO (TASK-APP-03), así que también tiene sentido explicar por qué se le clasifica en riesgo BAJO. `"N/A"` se reserva para quien nunca llega a evaluarse (HISTÓRICO, SIN REGISTRO, o sin modelo disponible para su bimestre).

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
