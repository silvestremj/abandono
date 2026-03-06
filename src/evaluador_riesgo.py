from typing import Any, Dict

import pandas as pd

# Importa la configuración global
from src.config import config


class EvaluadorRiesgo:
    """Módulo 4: Cálculo de riesgo (Reglas de negocio y heurística académica)."""

    def __init__(self) -> None:
        # Carga las reglas del YAML al instanciar la clase
        self.reglas: Dict[str, Any] = config.reglas_riesgo

    def _asignar_riesgo_administrativo(self, estado: str) -> str:
        estado_str = str(estado).upper()

        # 1. Chequea si el riesgo es ALTO (Abandono, Baja, etc.)
        if any(p in estado_str for p in self.reglas.get("palabras_alto_riesgo", [])):
            return "ALTO"

        # 2. Chequea si el riesgo es BAJO (Recibido, etc.)
        if any(p in estado_str for p in self.reglas.get("palabras_bajo_riesgo", [])):
            return "BAJO"

        # 3. Chequea si el riesgo es MEDIO (En curso, En pausa, etc.)
        if any(p in estado_str for p in self.reglas.get("palabras_medio_riesgo", [])):
            return "MEDIO"

        # 4. Si es "SIN REGISTRO" u otro estado desconocido
        return "NO CALCULABLE"

    def _calcular_medias_academicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula la nota media y asistencia media de los bimestres"""
        # Busca dinámicamente las columnas que empiecen por nota_ o asist_
        cols_notas = [c for c in df.columns if str(c).startswith("nota_")]
        cols_asist = [c for c in df.columns if str(c).startswith("asist_")]

        # Reemplazamos 0.0 por NA temporalmente para que no afecte el cálculo de la media
        # axis=1 para calcular la media por fila (estudiante)
        if cols_notas:
            df["nota_media"] = (
                df[cols_notas].replace(0.0, pd.NA).mean(axis=1).fillna(0.0)
            )
        else:
            df["nota_media"] = 0.0  # Si no hay columnas de notas, asigna 0.0

        if cols_asist:
            df["asistencia_media"] = (
                df[cols_asist].replace(0.0, pd.NA).mean(axis=1).fillna(0.0)
            )
        else:
            df["asistencia_media"] = 0.0  # Si no hay columnas de asistencia, asigna 0.0

        return df

    def _aplicar_heuristica(self, row: pd.Series) -> str:
        """Ajusta el riesgo administrativo en base a las notas y asistencias."""
        riesgo_base = row["riesgo_admin"]

        # 1. Si ya sabemos que es un éxito o un abandono, no lo tocamos.
        if riesgo_base in ["ALTO", "BAJO"]:
            return riesgo_base

        nota = row.get("nota_media", 0.0)
        asist = row.get("asistencia_media", 0.0)

        # 2. Si no hay datos académicos reales (0 absoluto en todo)
        # Se queda con su estado original (MEDIO si acaba de empezar, NO CALCULABLE si no hay info).
        if nota == 0.0 and asist == 0.0:
            return riesgo_base

        # 3. Si hay datos académicos, estos mandan
        # Penalizacion: Suspende o falta mucho a clase
        if nota < 5.0 or asist < 0.50:
            return "ALTO"

        # Bonificación: Va muy bien en clase
        if nota >= 7.0 and asist >= 0.80:
            return "BAJO"

        # Caso intermedio: Aprueba pero no llega a la excelencia
        # Se considera automáticamente un alumno "en curso" activo.
        return "MEDIO"

    def ejecutar_evaluacion(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or "estado_actual" not in df.columns:
            raise ValueError("El DataFrame no tiene la columna 'estado_actual'")

        df_riesgo = df.copy()

        # 1. Obtenemos riesgo administrativo (sin ajustar)
        df_riesgo["riesgo_admin"] = df_riesgo["estado_actual"].apply(
            self._asignar_riesgo_administrativo
        )

        # 2. Calculamos las medias académicas
        df_riesgo = self._calcular_medias_academicas(df_riesgo)

        # 3. Aplicamos la lógica combinada para obtener el nivel de riesgo final
        df_riesgo["nivel_riesgo"] = df_riesgo.apply(self._aplicar_heuristica, axis=1)

        return df_riesgo
