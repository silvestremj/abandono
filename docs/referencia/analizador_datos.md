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
# {'columna_analizada': 'nota_b1', 'media_aritmetica': 3.45, 'total_alumnos': 646}
```
