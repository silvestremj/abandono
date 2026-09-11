# Analizador de Datos

Módulo de análisis descriptivo y estadístico de la tabla maestra.

::: src.analizador_datos.Analizador
    options:
      show_source: false
      members:
        - calcular_estadisticas_basicas

---

## Ejemplo de uso

```python
from src.analizador_datos import Analizador

analizador = Analizador()
stats = analizador.calcular_estadisticas_basicas(df_master)

print(stats)
# {'columnas_analizadas': ['nota_b1', 'nota_b2', 'nota_b3', 'nota_b4', 'nota_b5', 'nota_b6'],
#  'media_aritmetica': 3.45,
#  'medias_por_bimestre': {'nota_b1': 3.62, 'nota_b2': 3.51, ...},
#  'total_alumnos': 646}
```
