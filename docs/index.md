# Predictor de Abandono Escolar

Pipeline de datos y modelo de evaluación de riesgo para el Trabajo de Fin de Máster (TFM).

---

## Descripción del Proyecto

Este sistema predice el **riesgo de abandono escolar** de estudiantes universitarios mediante árboles de decisión entrenados por hitos bimestrales. El pipeline cubre desde la ingesta de datos crudos hasta la inferencia de riesgo con explicabilidad local (XAI).

### Fases del sistema

| Fase | Descripción | Módulo |
|------|-------------|--------|
| Ingesta | Carga de CSV y Excel | `IngestorDatos` |
| Preparación | Limpieza, deduplicación, normalización | `PreparadorDatos` |
| Análisis | Estadísticas descriptivas | `Analizador` |
| Entrenamiento | Árboles de decisión por bimestre | `EntrenadorModelos` |
| Evaluación | Inferencia de riesgo + XAI | `EvaluadorRiesgo` |
| Presentación | Consola, CSV, interfaz web | `Visualizador` |

---

## Requisitos previos

* Python 3.10 o superior.
* Sistema operativo compatible (Windows, Linux, macOS).

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

### Pipeline por consola

```bash
python main.py
```

### Interfaz web (Streamlit)

```bash
streamlit run src/visualizador.py
```

### Tests

```bash
pytest tests/ -v
```

---

## Estructura del Proyecto

```
Abandono/
├── main.py                 # Orquestador del pipeline
├── config.yaml             # Configuración centralizada
├── requirements.txt        # Dependencias
├── estudiantes.db          # Base de datos SQLite
├── src/
│   ├── config.py           # Cargador de configuración
│   ├── gestor_base_datos.py
│   ├── gestor_logs.py
│   ├── ingestor_datos.py
│   ├── preparador_datos.py
│   ├── analizador_datos.py
│   ├── entrenador_modelos.py
│   ├── evaluador_riesgo.py
│   └── visualizador.py
├── data/                   # Datos de entrada
├── modelos/                # Modelos entrenados (.pkl)
├── output/                 # Exportaciones CSV
├── logs/                   # Ficheros de log diarios
├── tests/                  # Tests unitarios
├── notebooks/              # Notebooks de experimentación
├── docs/                   # Documentación Zensical
└── zensical.toml           # Configuración de Zensical
```
