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

### 2. Preparación (`PreparadorDatos`)

Pipeline de preprocesamiento aplicado:

1. **Normalización de texto**: Strip, mayúsculas, eliminación de tildes (NFKD + ASCII).
2. **Deduplicación**: Por `(n_siu, estudio)`, prevalece el último registro.
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
- Calcula probabilidad de abandono (clase 1).
- Asigna nivel de riesgo: ALTO (≥0.70), MEDIO (≥0.40), BAJO (<0.40).
- Genera justificación XAI recorriendo el camino del árbol de decisión.

### 5. Presentación (`Visualizador`)

- **Consola**: Reporte de texto con conteo por nivel.
- **CSV**: Exportación con timestamp a `output/`.
- **Streamlit**: Interfaz web con tabs de carga, resultados, alertas y visualización de árboles.

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

Cada predicción de riesgo ALTO o MEDIO incluye una justificación en texto plano que describe las reglas del árbol de decisión que llevaron a esa predicción, facilitando la interpretación por parte de los usuarios no técnicos.
