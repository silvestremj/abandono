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
| `LSE_Inscrip_Baja_Recibido.csv` | Inscripciones históricas con estados académicos (incluye dados de baja y recibidos) | `;` | `latin1` |
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
| `estado_actual` | Texto | Estado académico del estudiante | No (se usa para generar el target) |

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

El archivo Excel proporciona el estado académico **consolidado y actual** del estudiante. Se utiliza este formato porque representa una instantánea oficial del estado de cada alumno, con datos más limpios y estructurados que los archivos CSV históricos.

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

- **Deduplicación**: Se eliminan registros duplicados por (`n_siu`, `estudio`), prevaleciendo el último registro.
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
| `tf_nota`, `tf_asist`, `tf_recibido` | Variables de transformación (no predictivas) |

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

1. Guardar los ficheros de datos en la carpeta `data/` en la raíz del proyecto.
2. Ejecutar desde la terminal (en la carpeta del proyecto):
   ```
   streamlit run src/visualizador.py
   ```
3. Se abrirá la interfaz web en el navegador. Desde allí se pueden cargar los datos, ejecutar los cálculos y visualizar los resultados.
4. Los resultados también se exportan a la carpeta `output/` en formato CSV.

> **Nota**: El modo de ejecución puede cambiar en futuras versiones para que la persona usuaria no necesite usar la terminal.

## Estructura del Proyecto

```
Abandono/
├── data/                          # Ficheros de datos de entrada
│   ├── LSE_Inscrip_Baja_Recibido.csv
│   ├── LSE_Notas_Estadistica_Bimestre.csv
│   └── LSE_Notas_Inscrip_Baja_Actual.xlsx
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
