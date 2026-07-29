# Visualizador

Módulo de presentación de resultados: consola, CSV y interfaz web (Streamlit).

::: src.visualizador.Visualizador
    options:
      show_source: false
      members:
        - mostrar_en_consola
        - exportar_csv
        - graficar_arbol
        - generar_interfaz_web

---

## Ejemplo de uso en consola

```python
from src.visualizador import Visualizador

vista = Visualizador()
vista.mostrar_en_consola(df_final)
ruta = vista.exportar_csv(df_final)
print(f"Exportado a: {ruta}")
```

## Ejemplo de uso en Streamlit

```bash
streamlit run src/visualizador.py
```

---

## Interfaz web

La interfaz Streamlit dispone de cuatro pestañas:

| Pestaña | Función |
|---------|---------|
| Carga y ejecución | Subida de archivos, **modo de evaluación** y ejecución del pipeline |
| Resultados | Métricas de entrenamiento, validación y pronóstico (incluye el conteo de `SIN_DATOS_SUFICIENTES`) |
| Alertas | Listado de alumnos por nivel de riesgo (ALTO, MEDIO, BAJO) con justificación XAI, más un bloque aparte para `SIN_DATOS_SUFICIENTES` |
| Árbol de decisión | Árbol único y en vivo por bimestre, con justificación y métricas cuando corresponde a la evaluación oficial del alumno |

---

## Modo de evaluación (pestaña "Carga y ejecución")

Antes de pulsar "Ejecutar cálculos", la pestaña ofrece un control de modo de evaluación que se pasa como `bimestre_corte` a [`EvaluadorRiesgo.ejecutar_evaluacion`](evaluador_riesgo.md):

- **Automático (recomendado)** — por defecto. Cada alumno se evalúa en su propio bimestre real autodetectado.
- **Fijar bimestre de corte (simulación)** — añade un `selectbox` de bimestre (1..`max_bimestre`). Simula "qué se sabría si hoy fuera el Bimestre N", ignorando datos posteriores para todos los alumnos (salvo los que aún no llegan a N, evaluados en su bimestre real — ver `bimestre_evaluado` en [evaluador_riesgo.md](evaluador_riesgo.md)).

La elección se guarda en `st.session_state.bimestre_corte_activo` (`None` en modo Automático) para que el resto de pestañas de esa sesión de resultados sepan qué modo produjo el `df_riesgo` actual:

- **Resultados** añade la columna `bimestre_evaluado` al listado completo y muestra un aviso si, en modo simulación, algún alumno se evaluó por debajo del corte elegido por falta de datos reales.
- **Árbol de decisión** la usa para el panel de justificación (ver abajo).

---

## Pestaña "Alertas"

Separa a los alumnos PRONÓSTICO en bloques independientes por nivel de riesgo, cada uno construido por el método privado `_render_seccion_riesgo` para que los cuatro compartan exactamente la misma estructura (mensaje de cabecera, un `st.expander` por alumno con su justificación XAI, botón de descarga CSV):

| Bloque | Tono si hay alumnos | Tono si está vacío |
|--------|---------------------|---------------------|
| Riesgo ALTO | `st.error` | `st.success` |
| Riesgo MEDIO | `st.warning` | `st.info` |
| Riesgo BAJO | `st.info` | `st.info` |
| `SIN_DATOS_SUFICIENTES` | `st.warning` | (bloque oculto si no hay ninguno) |

Los tres niveles de riesgo se muestran siempre, incluso con 0 alumnos: antes solo se listaba ALTO, y con pocos casos en ese nivel la pestaña quedaba casi vacía pese a haber decenas de alumnos en MEDIO/BAJO cuya justificación también es útil revisar. El bloque `SIN_DATOS_SUFICIENTES` se mantiene aparte de los tres de riesgo (con un simple listado en tabla, sin expanders individuales) porque no es una alerta de abandono sino una ausencia de información — mezclarlo con ALTO/MEDIO/BAJO confundiría "sin dato" con "riesgo bajo".

---

## Pestaña "Árbol de decisión"

Muestra un **único árbol, en vivo** (sin botón de por medio): cambiar el bimestre o el alumno del desplegable regenera inmediatamente el árbol en pantalla, de modo que el mensaje mostrado y el árbol representado nunca puedan desincronizarse ni coexistir dos árboles distintos a la vez (versión anterior de TASK-APP-03: un árbol "oficial" automático y otro exploratorio tras pulsar un botón podían quedar visualmente contradictorios entre sí durante el intervalo sin pulsar el botón).

- La ruta **solo se colorea por nivel de riesgo cuando el bimestre elegido coincide con el `bimestre_evaluado` real** del alumno seleccionado. Si se elige otro bimestre, la ruta se sigue dibujando (para poder comparar estructuras) pero sin colorear, junto con un aviso explícito indicando en qué bimestre fue evaluado oficialmente.
- El panel de justificación, bimestre evaluado y métricas **solo aparece cuando el árbol en pantalla es, además, la evaluación oficial** del alumno (mismo bimestre) — nunca describe un hito distinto al que se está viendo:
  - La frase de `justificacion_riesgo` en lenguaje natural.
  - El `bimestre_evaluado` y, en modo simulación, si coincide o no con el corte fijado.
  - Las métricas (`Accuracy`, `Recall (Abandono)`, `Variable Clave`) de `df_metricas` para ese mismo Hito.
- "Descargar imagen PNG" entrega el archivo directamente al navegador del usuario (a su carpeta de descargas habitual) sin escribir nada en el servidor: reutiliza la misma figura ya mostrada en pantalla (vía un buffer en memoria, `io.BytesIO`) y nunca regenera un árbol distinto para el archivo descargado. `graficar_arbol` ya no admite guardar una copia en disco del lado del servidor (parámetro `guardar_ruta`, eliminado): la única vía de descarga es esta, directa al equipo del usuario.
