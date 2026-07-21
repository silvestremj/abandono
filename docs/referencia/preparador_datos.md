# Preparador de Datos

Módulo de integración, limpieza y preparación de datos para machine learning.

::: src.preparador_datos.PreparadorDatos
    options:
      show_source: false
      members:
        - ejecutar_preparacion
        - preparar_dataset_ml
        - filtrar_columnas_por_bimestre

---

## Ejemplo de uso

```python
from src.preparador_datos import PreparadorDatos

preparador = PreparadorDatos(datasets)

# Tabla maestra
df_master = preparador.ejecutar_preparacion()

# Dataset para ML (One-Hot Encoding + target binario)
df_ml = preparador.preparar_dataset_ml(df_master)

# Filtrar por bimestre (para entrenamiento temporal)
df_b2 = preparador.filtrar_columnas_por_bimestre(df_ml, bimestre_corte=2)
```

---

## Tareas de preprocesamiento

| Tarea | Descripción |
|-------|-------------|
| Normalización | Strip, mayúsculas, eliminación de tildes (NFKD + ASCII) |
| Deduplicación | Por `(n_siu, estudio)`, prevalece último registro |
| Nulos | Estados vacíos → `"SIN REGISTRO"`, notas → `0` |
| Type Casting | Conversión de coma decimal a punto |
| One-Hot Encoding | Variables categóricas → numéricas (`drop_first=True`) |
| Filtrado temporal | Elimina columnas de bimestres futuros |
