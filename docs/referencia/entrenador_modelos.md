# Entrenador de Modelos

Módulo de entrenamiento de árboles de decisión por hitos bimestrales.

::: src.entrenador_modelos.EntrenadorModelos
    options:
      show_source: false
      members:
        - entrenar_y_guardar

---

## Ejemplo de uso

```python
from src.entrenador_modelos import EntrenadorModelos
from src.preparador_datos import PreparadorDatos

preparador = PreparadorDatos(datasets)
df_master = preparador.ejecutar_preparacion()
df_ml = preparador.preparar_dataset_ml(df_master)

entrenador = EntrenadorModelos()
df_metricas = entrenador.entrenar_y_guardar(df_ml, preparador)
print(df_metricas[["Hito", "Accuracy", "Recall (Abandono)"]])
```

---

## Artefactos generados

El entrenador guarda en la carpeta `modelos/` los siguientes archivos:

| Archivo | Contenido |
|---------|-----------|
| `arbol_b1.pkl` ... `arbol_b6.pkl` | Modelo DecisionTreeClassifier por bimestre |
| `columnas_b1.pkl` ... `columnas_b6.pkl` | Lista ordenada de columnas del modelo |
| `metricas.pkl` | DataFrame con métricas comparativas |
