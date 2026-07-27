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
- **Detección de BOM independiente del encoding declarado**: si un CSV está configurado con un encoding distinto de `utf-8` (p.ej. `latin1`) pero en realidad empieza por el BOM de UTF-8 (`EF BB BF`), `leer_datos` lo detecta (`IngestorDatos._tiene_bom_utf8`) y fuerza `utf-8-sig` igualmente. Leer un CSV con BOM como `latin1` decodifica esos 3 bytes como caracteres sueltos que se anteponen al primer nombre de columna (p.ej. `n_siu` → `ï»¿n_siu`), rompiendo cualquier normalización o cruce posterior sobre esa columna — se detectó exactamente este caso en `LSE_Inscrip_Baja_Recibido.csv` (configurado como `latin1` en `config.yaml` pese a tener BOM UTF-8 real).
- Si un archivo no se encuentra, se imprime un error pero la ejecución continúa.
