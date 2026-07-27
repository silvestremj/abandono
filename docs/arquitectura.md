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
- Calcula probabilidad de abandono (clase 1).
- Asigna nivel de riesgo: ALTO (≥0.70), MEDIO (≥0.40), BAJO (<0.40).
- Genera justificación XAI recorriendo el camino del árbol de decisión y la traduce a una frase en lenguaje natural, para los tres niveles de riesgo (también BAJO).

### 5. Presentación (`Visualizador`)

- **Consola**: Reporte de texto con conteo por nivel.
- **CSV**: Exportación con timestamp a `output/`.
- **Streamlit**: Interfaz web con tabs de carga, resultados, alertas y visualización de árboles.
  - "Carga y ejecución" incluye el selector de **modo de evaluación** (Automático / Fijar bimestre de corte) que se pasa a `EvaluadorRiesgo.ejecutar_evaluacion` al pulsar "Ejecutar cálculos"; la elección queda en `st.session_state.bimestre_corte_activo` para el resto de pestañas.
  - "Árbol de decisión" renderiza un único árbol en vivo (sin botón) por bimestre: la ruta solo se colorea si el bimestre elegido coincide con el `bimestre_evaluado` real del alumno, y el panel de justificación/métricas solo se muestra en ese mismo caso — nunca coexisten dos árboles ni un mensaje que describa un bimestre distinto al mostrado.

---

## Decisiones de Diseño

### Modelos por bimestre (no global)

En lugar de un único modelo global, se entrenan **6 modelos independientes** (uno por bimestre). Esto permite:

- **Explicabilidad temporal**: Cada modelo refleja solo la información disponible en su hito.
- **Evitar data leakage**: El modelo de B1 no ve notas de B2-B6.
- **Adaptación a la realidad**: Un alumno en B2 tiene más información que uno en B1.

### Cost-Sensitive Learning

El parámetro `class_weight="balanced"` ajusta automáticamente los pesos para compensar el desequilibrio de clases (normalmente hay más alumnos que continúan que alumnos que abandonan).

### Explicabilidad Local (XAI)

Cada predicción de riesgo (ALTO, MEDIO o BAJO) incluye una justificación en lenguaje natural (no la salida técnica cruda del árbol) que describe las condiciones que llevaron a esa predicción, facilitando la interpretación por parte de los usuarios no técnicos.
