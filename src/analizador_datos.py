"""
Módulo de análisis descriptivo y estadístico.

Calcula estadísticas básicas sobre la tabla maestra para caracterizar
la población de estudiantes del dataset.

Uso::

    from src.analizador_datos import Analizador
    analizador = Analizador()
    stats = analizador.calcular_estadisticas_basicas(df_master)
"""

from typing import Any, Dict

import pandas as pd


class Analizador:
    """Análisis descriptivo y temporal de la tabla maestra.

    Proporciona estadísticas resumen que ayudan a entender la distribución
    y calidad de los datos antes del entrenamiento de modelos.
    """

    def calcular_estadisticas_basicas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estadísticas descriptivas de la tabla maestra.

        Args:
            df: Tabla maestra de estudiantes.

        Returns:
            Diccionario con ``columna_analizada``, ``media_aritmetica``
            y ``total_alumnos``, o ``error``/``warning`` en caso de fallo.
        """
        # Validación defensiva
        if df is None or df.empty:
            return {"error": "El DataFrame está vacío"}

        # Cálculo simple (Bala Trazadora)
        try:
            # Asumiendo que existen columnas que empiezan por 'nota_'
            cols_notas = [c for c in df.columns if isinstance(c, str) and "nota_" in c]
            if not cols_notas:
                return {"warning": "No hay columnas de notas para analizar"}

            # Media de la primera columna de notas encontrada
            col_ref = cols_notas[0]
            media = df[col_ref].mean()
            return {
                "columna_analizada": col_ref,
                "media_aritmetica": round(media, 2),
                "total_alumnos": len(df),
            }
        except Exception as e:
            return {"error": str(e)}
