# Evaluador de Riesgo

Módulo de inferencia de riesgo de abandono mediante modelos por hitos bimestrales con explicabilidad local (XAI).

::: src.evaluador_riesgo.EvaluadorRiesgo
    options:
      show_source: false
      members:
        - ejecutar_evaluacion

---

## Ejemplo de uso

```python
from src.evaluador_riesgo import EvaluadorRiesgo

evaluador = EvaluadorRiesgo()
df_final = evaluador.ejecutar_evaluacion(df_master)

# Ver resultados
df_alto = df_final[df_final["nivel_riesgo"] == "ALTO"]
print(df_alto[["n_siu", "estudio", "probabilidad_abandono", "justificacion_riesgo"]])
```

---

## Clasificación de alumnos

| Tipo | Condición | Evaluación |
|------|-----------|------------|
| HISTÓRICO | Estado es RECIBIDO, ABANDONO o BAJA | No se evalúa (target conocido) |
| PRONÓSTICO | Estado es CURSO o PAUSA | Se infiere riesgo con ML |
| SIN REGISTRO | Sin estado válido | No se puede calcular |

---

## Umbrales de riesgo

| Nivel | Probabilidad | Acción recomendada |
|-------|-------------|-------------------|
| ALTO | ≥ 0.70 | Seguimiento inmediato |
| MEDIO | ≥ 0.40 | Monitoreo periódico |
| BAJO | < 0.40 | Sin acción específica |
