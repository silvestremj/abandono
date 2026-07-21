# Visualizador

Módulo de presentación de resultados: consola, CSV y interfaz web (Streamlit).

::: src.visualizador.Visualizador
    options:
      show_source: false
      members:
        - mostrar_en_consola
        - exportar_csv
        - graficar_arbol
        - generar_interfaz_web

---

## Ejemplo de uso en consola

```python
from src.visualizador import Visualizador

vista = Visualizador()
vista.mostrar_en_consola(df_final, stats)
ruta = vista.exportar_csv(df_final)
print(f"Exportado a: {ruta}")
```

## Ejemplo de uso en Streamlit

```bash
streamlit run src/visualizador.py
```

---

## Interfaz web

La interfaz Streamlit dispone de cuatro pestañas:

| Pestaña | Función |
|---------|---------|
| Carga y ejecución | Subida de archivos y ejecución del pipeline |
| Resultados | Métricas de entrenamiento, validación y pronóstico |
| Alertas | Listado de alumnos en riesgo alto con justificación XAI |
| Árbol de decisión | Visualización gráfica del árbol por bimestre |
