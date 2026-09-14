# Predicción de Riesgo de Abandono Académico

## Objetivo

Desarrollar un sistema predictivo que identifique el riesgo de abandono de estudiantes universitarios, utilizando modelos de árbol de decisión entrenados con datos históricos académicos.

## Contexto de Privacidad y Protección de Datos

Este proyecto trabaja con datos académicos provenientes de la Universidad de Buenos Aires (UBA), en cumplimiento con la normativa de protección de datos aplicable:

- **Seudonimización del identificador principal**: El campo `n_siu` (Número de Sistema de Información Universitaria) que ingresa al sistema ya se encuentra **previamente seudonimizado**. Como desarrollador, nunca se ha tenido acceso a la tabla de correspondencia original que vincula este número con la identidad real del estudiante.
- **Principio de minimización de datos**: Solo se procesan las variables estrictamente necesarias para el modelo predictivo (notas, asistencia, estado académico, datos sociodemográficos generales).
- **Cumplimiento normativo**: Se respeta el RGPD (Reglamento General de Protección de Datos) y la Ley Nacional Argentina Nº 25.326 de Protección de Datos Personales al trabajar exclusivamente con datos anonimizados o seudonimizados proporcionados por la institución educativa.

## Formato de los Ficheros de Entrada

Los datos se organizan en la carpeta `data/` y pueden provenir de dos formatos distintos, cada uno con un propósito específico:

### Ficheros CSV — Datos históricos y transaccionales

Los archivos CSV almacenan datos históricos y transaccionales de naturaleza multimodal: inscripciones con datos sociodemográficos y calificaciones detalladas por bimestre. Requieren control estricto de codificación (`latin1`) y un separador personalizado (`;`) para preservar la integridad de los datos.

| Fichero | Propósito | Separador | Encoding |
|---|---|---|---|
| `LSE_Inscrip_Baja_Recibido.csv` | Desenlaces cerrados (Recibido / Abandono) con los datos de inscripción de cada caso. Se ingesta y se persiste como `raw_inscripciones`, pero no interviene en la Tabla Maestra | `;` | `latin1` |
| `LSE_Notas_Estadistica_Bimestre.csv` | Calificaciones y asistencia desagregadas por bimestre | `;` | `latin1` |

#### Estructura de columnas — `LSE_Inscrip_Baja_Recibido.csv`

| Columna | Tipo | Descripción | ¿Clave de merge? |
|---|---|---|---|
| `n_siu` | Texto | Identificador seudonimizado del estudiante | **Sí** |
| `estudio` | Texto | Carrera o programa académico | **Sí** |
| `cohorte` | Numérico | Año decohorte del estudiante | No |
| `year` | Numérico | Año del registro | No |
| `fecha` | Texto | Fecha del registro (formato MM/DD/YYYY HH:MM:SS) | No |
| `edad` | Numérico | Edad del estudiante | No |
| `fecha_nacimiento` | Texto | Fecha de nacimiento | No |
| `pais` | Texto | País de origen | No |
| `provincia` | Texto | Provincia de residencia | No |
| `ciudad` | Texto | Ciudad de residencia | No |
| `max_grado` | Texto | Nivel máximo de estudios alcanzados | No |
| `postgrado` | Numérico | Indicador de postgrado (0/1) | No |
| `ocupacion` | Texto | Descripción de la ocupación actual | No |
| `tipo_ocupacion` | Numérico | Código de la ocupación (0-8) | No |
| `año_estado` | Numérico | Año en que se registró el estado | No |
| `estado_actual` | Texto | Estado académico del estudiante | No (no se usa: la etiqueta se deriva del `estado_actual` del Excel) |

#### Estructura de columnas — `LSE_Notas_Estadistica_Bimestre.csv`

| Columna | Tipo | Descripción | ¿Clave de merge? |
|---|---|---|---|
| `n_siu` | Texto | Identificador seudonimizado del estudiante | **Sí** |
| `Estudio` | Texto | Carrera o programa académico | **Sí** |
| `nota_m` | Numérico | Nota promedio del bimestre | No |
| `nota_sd` | Numérico | Desviación estándar de la nota | No |
| `asist_m` | Numérico | Porcentaje de asistencia promedio | No |
| `asist_sd` | Numérico | Desviación estándar de la asistencia | No |
| `Comentario` | Texto | Identificador del bimestre (ej: "Bimestre: 1.0") | No |

### Fichero Excel — Estado consolidado actual

El archivo Excel proporciona el estado académico **consolidado y actual** del estudiante, incluyendo inscripción, notas y baja en un único registro por alumno.

| Fichero | Propósito | Formato |
|---|---|---|
| `LSE_Notas_Inscrip_Baja_Actual.xlsx` | Estado académico actual consolidado (inscripción + notas + baja) | Excel (`.xlsx`) |

#### Estructura de columnas principales — `LSE_Notas_Inscrip_Baja_Actual.xlsx`

| Columna | Tipo | Descripción | ¿Clave de merge? |
|---|---|---|---|
| `n_siu` | Texto | Identificador seudonimizado del estudiante | **Sí** |
| `estudio` | Texto | Carrera o programa académico | **Sí** |
| `estado_actual` | Texto | Estado académico actual del estudiante | No (genera el target) |
| `pais` | Texto | País de origen | No |
| `provincia` | Texto | Provincia de residencia | No |
| `tipo_ocupacion` | Numérico | Código de ocupación (se mapea a nombre descriptivo) | No |

> **Nota**: El Excel contiene columnas adicionales que se eliminan automáticamente durante el preprocesamiento (ver sección "Variables Excluidas").

### Claves de merge y Normalización

El sistema cruza los datos de los tres ficheros utilizando **dos campos clave**: `n_siu` y `estudio`. Para garantizar que el merge funcione correctamente independientemente de variaciones en el formato de los datos, se aplica un proceso de normalización:

```python
# Normalización aplicada (preparador_datos.py:16-34)
# 1. Convertir a string
# 2. Eliminar espacios en blanco a los lados (strip)
# 3. Convertir a mayúsculas
# 4. Eliminar tildes y caracteres diacríticos
```

**Ejemplo de fila mínima para merge exitoso:**

```
# En LSE_Notas_Inscrip_Baja_Actual.xlsx:
n_siu: "2974"  |  estudio: "MSE"

# En LSE_Notas_Estadistica_Bimestre.csv:
n_siu: "2273001"  |  Estudio: "CEIoT"
```

Ambas combinaciones (`n_siu` + `estudio`) deben existir en los ficheros para que el cruce se realize correctamente.

### Subida de ficheros desde la interfaz web

El pipeline solo lee los **tres ficheros declarados en `config.yaml`**, y los busca por su **nombre exacto** dentro de `data/`. Por eso, la pestaña *Carga y ejecución* no acepta cualquier fichero: lo que sube la persona usuaria se trata siempre como una **versión nueva de uno de esos tres**, nunca como un fichero adicional.

Cada fichero subido pasa por tres controles antes de sustituir al que ya está en `data/`:

| Control | Qué comprueba | Si no se cumple |
|---|---|---|
| **Nombre declarado** | Que el nombre coincida exactamente con uno de los `datasets[*].nombre` de `config.yaml` | El fichero **no se escribe**. Se muestra un error con el nombre recibido y la lista de nombres admitidos, y se registra en el log |
| **Estructura** | Que el fichero se pueda leer con los parámetros de su clave (tipo, separador, encoding, BOM UTF-8) y que traiga las columnas mínimas de esa fuente | **No se sustituye el original**. Se muestra el motivo concreto (qué columna falta o qué formato no cuadra) y se registra el rechazo |
| **Respaldo** | Que no se pierda la versión anterior | Antes de sobrescribir, el fichero actual se copia a `data/_backup/<nombre>.<YYYYmmdd_HHMMSS>` y la copia se registra en el log |

La validación se hace sobre una copia temporal, de modo que un fichero inválido **nunca llega a tocar** el original: el pipeline sigue ejecutándose sobre la versión buena que ya había.

Columnas mínimas exigidas a cada fuente (constante `COLUMNAS_MINIMAS_DATASET` en `src/visualizador.py`):

| Fichero | Columnas mínimas | Comprobación adicional |
|---|---|---|
| `LSE_Notas_Estadistica_Bimestre.csv` | `n_siu`, `Estudio`, `nota_m`, `asist_m`, `Comentario` | Al menos una fila de `Comentario` debe indicar el bimestre (patrón `Bimestre: <n>`), que es de donde se extrae el número de bimestre |
| `LSE_Notas_Inscrip_Baja_Actual.xlsx` | `n_siu`, `estudio`, `estado_actual` | — |
| `LSE_Inscrip_Baja_Recibido.csv` | `n_siu`, `estudio` | — |

La cabecera de la sección *1. Cargar archivos de datos* lista los tres nombres admitidos e indica, para cada uno, si está presente en `data/` y su fecha de modificación. **Si no se sube ningún fichero, los cálculos se ejecutan sobre lo que ya haya en `data/`**, sin cambios.

## Flujo de Memoria y Tabla Maestra

Los datos abandonan su formato físico (CSV/Excel) y se cargan en memoria como un diccionario de DataFrames (`datasets_dict`) que preserva la información sin alterar. Posteriormente, mediante operaciones de reestructuración tabular en Pandas, se consolida la **Tabla Maestra** (`df_master`), que es la base de todo el análisis.

### Pipeline de procesamiento

```
1. INGESTA (ingestor_datos.py)
   CSV + Excel → datasets_dict (diccionario de DataFrames crudos)

2. PREPARACIÓN (preparador_datos.py)
   datasets_dict → Normalización → Deduplicación → Merge
   → Tabla Maestra (df_master)

3. ANÁLISIS (analizador_datos.py)
   df_master → Estadísticas descriptivas

4. MACHINE LEARNING (preparador_datos.py → entrenador_modelos.py)
   df_master → Variables numéricas (One-Hot Encoding)
   → Entrenamiento de árboles de decisión por bimestre

5. EVALUACIÓN (evaluador_riesgo.py)
   df_master + Modelos guardados → Predicción de riesgo
   → Reporte final con nivel de riesgo y justificación
```

### Transformaciones clave en la preparación

- **Deduplicación**: Se eliminan registros duplicados por (`n_siu`, `estudio`), prevaleciendo el registro más informativo (con `estado_actual` y/o nota reales) y, entre varios igual de informativos, el último.
- **Tratamiento de nulos**: Los valores faltantes en columnas de bimestres se rellenan con `0.0`.
- **Mapeo geográfico**: Los países se agrupan en "ARGENTINA" / "OTRO PAIS". Las provincias con valor "OTRO" o nulas se etiquetan como "DESCONOCIDA".
- **Mapeo de ocupación**: Los códigos numéricos de `tipo_ocupacion` (0-8) se transforman a nombres descriptivos (ej: `2` → "Desarrollador/Programador").

## Variables Excluidas para Machine Learning

Antes de entrenar los modelos, el sistema elimina automáticamente las variables que no son predictivas o que provocan **fuga de datos** (data leakage). Esto garantiza que el modelo solo use información disponible al momento de la predicción.

### Variables eliminadas por fuga de datos (Data Leakage)

| Variable | Motivo de exclusión |
|---|---|
| `comentario_baja` | Contiene información del resultado final (fuga de datos) |
| `fecha_estado` | Revela la fecha en que se determinó el estado (fuga de datos) |
| `fecha_baja` | Contiene información del momento de la baja (fuga de datos) |
| `estado_baja` | Revela el estado final de la baja (fuga de datos) |
| `causa_baja` | Contiene la causa de la baja (fuga de datos) |
| `target` | Es el target interno de reglas de negocio (no debe usarse para ML) |
| `tf_nota`, `tf_asist`, `tf_recibido` | Evaluación del trabajo final (TF): solo existen para el alumnado que se ha recibido, por lo que revelan el desenlace (fuga de datos) |

### Variables eliminadas por no ser predictivas

| Variable | Motivo de exclusión |
|---|---|
| `n_siu` | Identificador único (no aporta información predictiva) |
| `fecha` | Fecha del registro (no es predictiva) |
| `fecha_nacimiento` | Se usa para calcular edad, pero no es predictiva directamente |
| `estado_actual` | Se usa para generar el target_ml, pero no puede usarse como predictor |
| `año_estado` | Año del registro (no es predictiva) |
| `comentario` | Comentario libre (no estructurado) |
| `cohorte` | Año de cohorte (no es predictiva directamente) |
| `year` | Año del registro (no es predictiva) |
| `cant` | Campo no accionable |

### Variables eliminadas por alta cardinalidad

| Variable | Motivo de exclusión |
|---|---|
| `ciudad` | Demasiados valores únicos (alta cardinalidad), no generaliza bien |
| `ocupacion` | Texto libre con muchos valores únicos (alta cardinalidad) |

### Variables de basura del Excel

| Variable | Motivo de exclusión |
|---|---|
| `unnamed: 28`, `unnamed: 29`, `unnamed: 30`, `unnamed: 31` | Columnas vacías o sin sentido generadas por Excel |

## Instrucciones de Uso

1. Instalar las dependencias del proyecto:
   ```
   pip install -r requirements.txt
   ```
2. Guardar los ficheros de datos en la carpeta `data/` en la raíz del proyecto, con los nombres exactos declarados en `config.yaml`.
3. Ejecutar desde la terminal (en la carpeta del proyecto):
   ```
   streamlit run src/visualizador.py
   ```
4. Se abrirá la interfaz web en el navegador. Desde allí se pueden subir versiones nuevas de los tres ficheros declarados (ver [Subida de ficheros desde la interfaz web](#subida-de-ficheros-desde-la-interfaz-web)), ejecutar los cálculos y visualizar los resultados. Si no se sube nada, se usa lo que ya hay en `data/`.
5. Los resultados también se exportan a la carpeta `output/` en formato CSV.

> **Nota**: El modo de ejecución puede cambiar en futuras versiones para que la persona usuaria no necesite usar la terminal.

## Estructura del Proyecto

```
Abandono/
├── data/                          # Ficheros de datos de entrada
│   ├── LSE_Inscrip_Baja_Recibido.csv
│   ├── LSE_Notas_Estadistica_Bimestre.csv
│   ├── LSE_Notas_Inscrip_Baja_Actual.xlsx
│   └── _backup/                   # Versiones anteriores, archivadas al subir una nueva
├── modelos/                       # Modelos entrenados guardados (.pkl)
├── output/                        # Resultados exportados
├── src/                           # Código fuente
│   ├── ingestor_datos.py          # Módulo 1: Carga de datos
│   ├── preparador_datos.py        # Módulo 2: Integración y limpieza
│   ├── analizador_datos.py        # Módulo 3: Análisis descriptivo
│   ├── entrenador_modelos.py      # Entrenamiento de árboles de decisión
│   ├── evaluador_riesgo.py        # Módulo 4: Evaluación de riesgo
│   ├── visualizador.py            # Interfaz Streamlit
│   ├── gestor_base_datos.py       # Gestión de SQLite
│   ├── gestor_logs.py             # Sistema de logs
│   └── config.py                  # Carga de configuración YAML
├── config.yaml                    # Configuración del proyecto
├── main.py                        # Punto de entrada principal
└── README.md                      # Este archivo
```
