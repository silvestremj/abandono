# Ingestor de Datos

Módulo 1: Carga y validación de archivos de entrada (CSV y Excel).

::: src.ingestor_datos.IngestorDatos
    options:
      show_source: false
      members:
        - leer_datos

---

## Ejemplo de uso

```python
from src.ingestor_datos import IngestorDatos

ingestor = IngestorDatos()
datasets = ingestor.leer_datos()

print(datasets.keys())  # dict_keys(['inscripciones', 'notas_bimestre', 'actual'])
```

---

## Notas de implementación

- Los archivos CSV con encoding `utf-8` se leen automáticamente con `utf-8-sig` para eliminar el BOM de Excel.
- Si un archivo no se encuentra, se imprime un error pero la ejecución continúa.
