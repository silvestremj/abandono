# Configuración

Módulo de configuración centralizada. Lee `config.yaml` y distribuye los parámetros del proyecto.

::: src.config.ConfigLoader
    options:
      show_source: false
      members:
        - datasets
        - paths
        - reglas_riesgo
        - machine_learning

---

## Ejemplo de uso

```python
from src.config import config

# Rutas del proyecto
data_dir = config.paths.get("data_dir", "data")

# Configuración de datasets
for clave, conf in config.datasets.items():
    print(f"{clave}: {conf['nombre']} ({conf['tipo']})")

# Reglas de negocio
reglas = config.reglas_riesgo
print(reglas["palabras_alto_riesgo"])  # ['ABANDONO', 'LIBRE', 'BAJA']
```

---

## Archivo de configuración

El archivo `config.yaml` define cuatro secciones principales:

| Sección | Contenido |
|---------|-----------|
| `paths` | Rutas de directorios (data, output, db) |
| `datasets` | Archivos de entrada (nombre, tipo, separador, encoding) |
| `reglas_riesgo` | Palabras clave para clasificar estados de riesgo |
| `machine_learning` | Parámetros de entrenamiento (forzar_entrenamiento, max_bimestre) |
