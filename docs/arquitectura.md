# Arquitectura del Sistema

## Diagrama del Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  config.yaml │───▶│  IngestorDatos   │───▶│ GestorBaseDatos│
│  (reglas)    │    │  (CSV + Excel)   │    │  (SQLite RAW)  │
└─────────────┘    └──────────────────┘    └───────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Analizador  │◀──│ PreparadorDatos  │───▶│ GestorBaseDatos│
│  (estadíst.) │   │  (limpieza)      │    │  (master)      │
└─────────────┘    └────────┬─────────┘    └────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌──────────────┐ ┌───────────┐ ┌──────────────┐
     │EntrenadorMod.│ │Evaluador  │ │  Visualizador│
     │  (B1..B6)    │ │  Riesgo   │ │ (consola/web)│
     └──────────────┘ └───────────┘ └──────────────┘
              │             │
              ▼             ▼
        ┌────────────────────────┐
        │   modelos/*.pkl        │
        │   (árboles + columnas) │
        └────────────────────────┘
```

## Flujo de Datos

### 1. Ingesta (`IngestorDatos`)

Lee los archivos definidos en `config.yaml`:

| Dataset | Archivo | Formato |
|---------|---------|---------|
| `inscripciones` | `LSE_Inscrip_Baja_Recibido.csv` | CSV (`;`, latin1) |
| `notas_bimestre` | `LSE_Notas_Estadistica_Bimestre.csv` | CSV (`;`, latin1) |
| `actual` | `LSE_Notas_Inscrip_Baja_Actual.xlsx` | Excel |

`leer_datos` detecta el BOM de UTF-8 (`EF BB BF`) al inicio de un CSV con independencia del `encoding` declarado en `config.yaml`, y fuerza `utf-8-sig` si lo encuentra: leer un fichero con BOM bajo `latin1` corrompe el primer nombre de columna (p.ej. `n_siu` → `ï»¿n_siu`), rompiendo la normalización/cruce posterior sobre esa columna — caso real detectado en `LSE_Inscrip_Baja_Recibido.csv`.

### 2. Preparación (`PreparadorDatos`)

Pipeline de preprocesamiento aplicado:

1. **Normalización de texto**: Strip, mayúsculas, eliminación de tildes (NFKD + ASCII).
2. **Deduplicación**: Por `(n_siu, estudio)`, prevalece el registro más informativo (`estado_actual`/`nota` reales) y, entre varios igual de informativos, el último.
3. **Tratamiento de nulos**: Estados vacíos → `"SIN REGISTRO"`, notas faltantes → `0`.
4. **Mapeo de reglas de negocio**: Estados → valores numéricos de riesgo.
5. **Pivot de notas**: Conversión de formato largo a ancho (`nota_b1`, `asist_b1`, ..., `nota_b6`, `asist_b6`).
6. **Unión**: Merge de tabla base con notas por bimestre.
7. **Type Casting**: Conversión de tipos numéricos (coma → punto).

### 3. Entrenamiento (`EntrenadorModelos`)

Para cada bimestre (1-6):

- Se filtran solo las columnas disponibles hasta ese hito temporal.
- Se entrena un `DecisionTreeClassifier` con `class_weight="balanced"`.
- División 80% train / 20% test (`random_state=42`).
- Se guardan: modelo (`arbol_bN.pkl`), columnas (`columnas_bN.pkl`), métricas.

### 4. Evaluación (`EvaluadorRiesgo`)

- Clasifica alumnos: HISTÓRICO (target conocido) vs PRONÓSTICO (activo sin etiqueta).
- Para cada alumno PRONÓSTICO, detecta su bimestre actual y usa el modelo correspondiente.
  - **Modo de evaluación** (parámetro `bimestre_corte`, elegido en la pestaña "Carga y ejecución"):
    - *Automático* (`bimestre_corte=None`, por defecto): cada alumno se autodetecta en su propio bimestre real.
    - *Fijar bimestre de corte* (simulación): se recortan las columnas de bimestres posteriores a `N` (vía `PreparadorDatos.filtrar_columnas_por_bimestre`) antes de autodetectar, para que ningún alumno se evalúe con datos futuros a `N`. Actúa como techo: un alumno sin datos reales hasta `N` se evalúa igualmente en su bimestre real, no con ceros forzados.
  - El bimestre realmente usado para cada alumno queda registrado en la columna `bimestre_evaluado`.
- Antes de inferir, comprueba si el alumno tiene algún dato real hasta ese bimestre (`nota_bN` y `asist_bN` no valen `0.0` a la vez). Si no lo tiene, no ejecuta el modelo y lo marca `SIN_DATOS_SUFICIENTES` en vez de arriesgar un falso ALTO (ver [Evaluador de Riesgo](referencia/evaluador_riesgo.md#estado-sin_datos_suficientes)).
- Si sí hay datos reales, calcula probabilidad de abandono (clase 1) y asigna nivel de riesgo: ALTO (≥0.70), MEDIO (≥0.40), BAJO (<0.40).
- Genera justificación XAI recorriendo el camino del árbol de decisión y la traduce a una frase en lenguaje natural, para los tres niveles de riesgo (también BAJO) y para `SIN_DATOS_SUFICIENTES`.

### 5. Presentación (`Visualizador`)

- **Consola**: Reporte de texto con conteo por nivel.
- **CSV**: Exportación con timestamp a `output/`.
- **Streamlit**: Interfaz web con tabs de carga, resultados, alertas y visualización de árboles.
  - "Carga y ejecución" incluye el selector de **modo de evaluación** (Automático / Fijar bimestre de corte) que se pasa a `EvaluadorRiesgo.ejecutar_evaluacion` al pulsar "Ejecutar cálculos"; la elección queda en `st.session_state.bimestre_corte_activo` para el resto de pestañas.
  - "Alertas" separa a los alumnos PRONÓSTICO en tres bloques por nivel de riesgo (ALTO, MEDIO, BAJO), cada uno con su propio mensaje, listado expandible por alumno y descarga CSV, más un bloque aparte para `SIN_DATOS_SUFICIENTES` (no es una alerta de riesgo, es ausencia de información) — así los tres niveles quedan siempre visibles aunque alguno tenga muy pocos o ningún alumno.
  - "Árbol de decisión" renderiza un único árbol en vivo (sin botón) por bimestre: la ruta solo se colorea si el bimestre elegido coincide con el `bimestre_evaluado` real del alumno, y el panel de justificación/métricas solo se muestra en ese mismo caso — nunca coexisten dos árboles ni un mensaje que describa un bimestre distinto al mostrado.

---

## Decisiones de Diseño

### Modelos por bimestre (no global)

En lugar de un único modelo global, se entrenan **6 modelos independientes** (uno por bimestre). Esto permite:

- **Explicabilidad temporal**: Cada modelo refleja solo la información disponible en su hito.
- **Evitar data leakage**: El modelo de B1 no ve notas de B2-B6.
- **Adaptación a la realidad**: Un alumno en B2 tiene más información que uno en B1.

### Cost-Sensitive Learning

El parámetro `class_weight="balanced"` ajusta automáticamente los pesos para compensar el desequilibrio de clases (normalmente hay más alumnos que continúan que alumnos que abandonan). Ver [Limitaciones Conocidas](#hojas-pequenas-y-class_weight-que-tan-fiable-es-la-probabilidad-de-una-hoja-con-pocas-muestras) para un análisis de cuánto amplifica esta reponderación la probabilidad de las hojas pequeñas (frecuentes dado el tamaño del histórico) y si eso llega a cambiar el nivel de riesgo asignado.

### Explicabilidad Local (XAI)

Cada predicción de riesgo (ALTO, MEDIO o BAJO) incluye una justificación en lenguaje natural (no la salida técnica cruda del árbol) que describe las condiciones que llevaron a esa predicción, facilitando la interpretación por parte de los usuarios no técnicos.

### Exclusión de fuga de datos ("Data Leakage")

`PreparadorDatos.COLUMNAS_EXCLUIDAS_ML` es la fuente única de verdad de qué columnas no deben llegar al modelo (identificadores, desenlace ya conocido, medias globales `nota`/`asist`/`sd_nota`/`sd_asist` que resumen *todos* los bimestres a la vez, columnas `Unnamed: N` = basura de Excel, variables de alta cardinalidad). La reutilizan tanto `preparar_dataset_ml` (entrenamiento) como `preprocesar_fila_alumno` (inferencia) a través de `excluir_columnas_ml`, que compara nombres de columna sin distinguir mayúsculas/minúsculas — necesario porque `ejecutar_preparacion` normaliza las columnas del Excel de origen a minúsculas, y una entrada de exclusión con otra capitalización dejaría de coincidir en silencio (caso real detectado: `"Unnamed: 30"` colaba como variable de split real en el árbol de Bimestre 1 antes de esta corrección).

### Ambigüedad del cero relleno (`SIN_DATOS_SUFICIENTES`)

`ejecutar_preparacion` rellena con `0.0` `nota_bN`/`asist_bN` cuando un alumno no tiene registro real en `LSE_Notas_Estadistica_Bimestre.csv` para ese bimestre — la cobertura real de ese fichero es baja (~9% del alumnado activo en Bimestre 1 en los datos del proyecto). Un `0.0` así relleno es indistinguible, a nivel de dato, de un `0.0` real (alumno que efectivamente rindió con nota cero), pero ambos casos tienen implicaciones opuestas. `EvaluadorRiesgo` resuelve la ambigüedad verificando si `nota_bN` **y** `asist_bN` valen `0.0` a la vez para el bimestre detectado del alumno: ese doble-cero conjunto es la señal empíricamente fiable de "sin registro" (ver [Evaluador de Riesgo](referencia/evaluador_riesgo.md#estado-sin_datos_suficientes)), y en ese caso no se ejecuta el modelo sobre esa fila.

---

## Limitaciones Conocidas

### Hojas del árbol respaldadas por muy pocas muestras (sobreajuste) — corregida

Causa estructural: el histórico de alumnos con desenlace conocido tiene solo 97 registros en total (~77 en train, ~20 en test por bimestre tras el split 80/20). `DecisionTreeClassifier(max_depth=5, class_weight="balanced")` no fijaba `min_samples_leaf`/`min_samples_split`, así que con tan pocas muestras el árbol podía aislar alumnos individuales en hojas propias: se verificó que, sin esa restricción, entre el 8% y el 58% de las hojas de los 6 árboles estaban respaldadas por 1-2 muestras — una hoja así predice con 100%/0% de "confianza" sin respaldo estadístico real. Caso detectado: un alumno con historial claramente bueno (notas > 7 y asistencia > 75% en los cuatro bimestres) fue clasificado ALTO con probabilidad 1.0 por caer en una hoja de 1 sola muestra del árbol de Bimestre 4.

Tras un análisis de sensibilidad se fijó `min_samples_leaf=3` (el valor mínimo que garantiza estructuralmente 0 hojas de ≤2 muestras; valores mayores empeoraban visiblemente la Accuracy dado el tamaño del histórico). Efecto verificado tras reentrenar: 0.0% de hojas con ≤2 muestras en los 6 árboles (antes, 8%-58%); Accuracy media 0.717→0.633 (coste moderado) y Recall medio 0.382→0.405 (mejora ligera); el alumno del caso detectado pasa de una hoja de 1 muestra (ALTO, prob. 1.0) a una de 25 muestras (BAJO, prob. 0.0). Detalle completo, tabla por bimestre y test de regresión en [Entrenador de Modelos](referencia/entrenador_modelos.md#limitacion-conocida-hojas-de-1-2-muestras-corregida-con-min_samples_leaf3).

`min_samples_leaf=3` es el mínimo que resuelve el problema, no un óptimo definitivo: con solo 97 alumnos etiquetados debería reevaluarse en cursos futuros a medida que el histórico crezca (más datos permiten previsiblemente un valor mayor, más robusto, sin penalizar tanto la Accuracy).

### Codificación de variables categóricas en la inferencia por fila individual — corregida

Al verificar el caso del alumno con hoja de 1 muestra (arriba) se detectó un segundo bug, independiente: `preprocesar_fila_alumno` (usado por `EvaluadorRiesgo.ejecutar_evaluacion` para inferir el riesgo de cada alumno PRONÓSTICO uno a uno) codificaba con One-Hot Encoding **una fila aislada** con `drop_first=True`. Explicado en simple: una fila sola solo puede tener un único valor por variable categórica (p.ej. una sola provincia), y `drop_first=True` siempre descarta esa única categoría presente — generando 0 columnas dummy para esa variable. El relleno posterior de columnas faltantes con 0 terminaba poniendo a 0 la categoría **real** del alumno exactamente igual que cualquier otra categoría que no tuviera: el modelo no podía distinguir "es de Córdoba" de "no es de ningún sitio en particular". Esto no afectaba a `preparar_dataset_ml` (entrenamiento), que sí opera sobre el dataset completo con múltiples categorías presentes, donde `drop_first=True` es correcto.

Impacto verificado sobre los datos reales del proyecto: de los 8 alumnos PRONÓSTICO con inferencia real (excluyendo `SIN_DATOS_SUFICIENTES`), **2 (25%) cambiaban de nivel de riesgo** por este bug — ver tabla completa en [Evaluador de Riesgo](referencia/evaluador_riesgo.md#codificacion-de-categoricas-en-inferencia-por-fila-corregida). Corregido cambiando a `drop_first=False` en `preprocesar_fila_alumno` (el alineado posterior a `columnas_entrenamiento` ya descarta cualquier dummy sobrante, así que el comportamiento resultante es correcto en todos los casos). **Las justificaciones XAI generadas en ejecuciones anteriores a este fix pueden haber descrito incorrectamente variables categóricas (país, provincia, ocupación, tipo de estudio) y no deben tomarse como referencia histórica fiable en ese aspecto.**

### Hojas pequeñas y `class_weight`: ¿qué tan fiable es la probabilidad de una hoja con pocas muestras?

Aun con `min_samples_leaf=3` y sin fuga de datos ni bug de codificación, las hojas donde caen los alumnos PRONÓSTICO actuales siguen siendo pequeñas: entre 4 y 24 muestras de entrenamiento, según se auditó sobre los 8 alumnos con inferencia real — consecuencia directa, otra vez, de que el histórico completo son solo 97 alumnos. Caso analizado en detalle: un alumno con buen historial (Ingeniero, nota B4 > 7.6, asistencia B3 > 71%) fue clasificado MEDIO (prob. 0.552) porque cae en una hoja con 5 muestras históricas, de las cuales 2 (40%) acabaron en abandono — un dato real, no una hoja de 1 muestra con falsa certeza.

Se investigó si `class_weight="balanced"` (que reescala las clases para compensar que hay más "continúa" que "abandono" en el histórico, y así mejorar el Recall) estaba **inflando artificialmente** esa probabilidad por encima de un umbral de nivel de riesgo. Verificado sobre los 8 alumnos con inferencia real: **en ninguno de los 8 casos el nivel de riesgo cambia** al comparar la probabilidad ponderada por `class_weight` contra la proporción bruta (sin ponderar) de abandono en la misma hoja — en el caso analizado, el 40% bruto ya estaba por encima del umbral de MEDIO (≥0.40) antes de cualquier reponderación; `class_weight` amplifica el número (a 0.552) pero no cambia la categoría asignada. Por tanto, `class_weight="balanced"` no está distorsionando las clasificaciones actuales, pero el tamaño pequeño de estas hojas (4-24 muestras) sigue siendo un límite estructural del histórico disponible: interpretar "riesgo MEDIO" o "riesgo ALTO" como una probabilidad estadísticamente sólida, en vez de como una señal orientativa basada en un puñado de casos históricos similares, sería sobre-interpretar la precisión del modelo.

**Un caso más extremo, encontrado en modo simulación** (no en el modo Automático, que es el que usa la interfaz por defecto): al fijar el bimestre de corte en B3, el alumno E1430/MSE — cuya evaluación real en modo Automático (autodetectado en su hito real, Bimestre 5) es MEDIO (prob. 0.552) — se clasifica **ALTO (prob. 0.847)** con el árbol de B3, pese a tener una nota de 8.3 en ese mismo bimestre; el árbol combina esa nota con otras tres condiciones (nota B1 = 0.0, programa distinto de CESE, provincia desconocida). La hoja del árbol de B3 donde cae tiene solo **4 muestras de entrenamiento**, de las cuales 3 abandonaron (75% bruto, sin ponderar por `class_weight`; con `class_weight="balanced"` sube al 84.7% mostrado). Es el ejemplo más frágil detectado hasta ahora: ni siquiera hace falta invocar el efecto de `class_weight` para llegar a ALTO, ya que el 75% bruto por sí solo ya supera el umbral (≥0.70) — la fragilidad viene enteramente del tamaño de la muestra (n=4), no de la reponderación. Refuerza la recomendación general y añade una específica del modo simulación: cuanto más temprano el bimestre de corte elegido, menos alumnos históricos comparables hay disponibles para ese árbol y menos fiable es la probabilidad resultante — una probabilidad de "simulación" en un bimestre temprano debe leerse con más cautela todavía que una evaluación en modo Automático.
