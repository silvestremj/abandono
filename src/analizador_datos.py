"""
Módulo de análisis descriptivo y estadístico.

Calcula estadísticas básicas sobre la tabla maestra para caracterizar
la población de estudiantes del dataset.

Uso::

    from src.analizador_datos import Analizador
    analizador = Analizador()
    stats = analizador.calcular_estadisticas_basicas(df_master)
"""

import re
from typing import Any, Dict, List

import pandas as pd

from src.config import config

# Patrón anclado para las columnas de calificación por bimestre ('nota_b1',
# 'nota_b2', ...). El anclaje evita capturar por subcadena otras columnas
# futuras que contengan 'nota_' sin ser notas de bimestre.
PATRON_NOTA_BIMESTRE = re.compile(r"^nota_b(\d+)$")


class Analizador:
    """Análisis descriptivo y temporal de la tabla maestra.

    Proporciona estadísticas resumen que ayudan a entender la distribución
    y calidad de los datos antes del entrenamiento de modelos. La media
    aritmética que calcula es global: agrega en un único promedio todas las
    calificaciones de todos los bimestres modelados por el sistema
    (hasta ``machine_learning.max_bimestre`` en ``config.yaml``), de modo
    que el dato sea coherente con el entrenamiento y con la inferencia.
    """

    def calcular_estadisticas_basicas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estadísticas descriptivas globales de la tabla maestra.

        Selecciona las columnas de calificación por bimestre (``nota_b1``,
        ``nota_b2``, ...) limitadas al último bimestre modelado
        (``config.machine_learning['max_bimestre']``) y promedia en conjunto
        todos sus valores, ignorando los nulos. Los bimestres posteriores
        (7 en adelante) se conservan en la tabla maestra por trazabilidad,
        pero quedan fuera del modelo y, por tanto, fuera de este cálculo.

        Args:
            df: Tabla maestra de estudiantes.

        Returns:
            Diccionario con ``columnas_analizadas`` (lista de columnas
            empleadas), ``media_aritmetica`` (media global redondeada a 2
            decimales), ``medias_por_bimestre`` (``{columna: media}``) y
            ``total_alumnos``, o ``error``/``warning`` en caso de fallo.
        """
        # Validación defensiva
        if df is None or df.empty:
            return {"error": "El DataFrame está vacío"}

        try:
            max_bimestre: int = config.machine_learning.get("max_bimestre", 6)

            # Columnas de nota por bimestre dentro del horizonte modelado,
            # ordenadas por número de bimestre (no por el orden del DataFrame).
            bimestres: List[tuple] = []
            for c in df.columns:
                if not isinstance(c, str):
                    continue
                coincidencia = PATRON_NOTA_BIMESTRE.match(c)
                if coincidencia and int(coincidencia.group(1)) <= max_bimestre:
                    bimestres.append((int(coincidencia.group(1)), c))

            cols_notas: List[str] = [c for _, c in sorted(bimestres)]
            if not cols_notas:
                return {"warning": "No hay columnas de notas para analizar"}

            # Media global: todos los valores de todas las columnas tomados en
            # conjunto. pandas ignora los nulos en 'mean()'.
            valores = pd.concat([df[c] for c in cols_notas], ignore_index=True)
            media_global = valores.mean()

            medias_por_bimestre: Dict[str, float] = {
                c: round(float(df[c].mean()), 2) for c in cols_notas
            }

            return {
                "columnas_analizadas": cols_notas,
                "media_aritmetica": round(float(media_global), 2),
                "medias_por_bimestre": medias_por_bimestre,
                "total_alumnos": len(df),
            }
        except Exception as e:
            return {"error": str(e)}
