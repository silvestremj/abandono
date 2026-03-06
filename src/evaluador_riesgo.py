from typing import Any, Dict

import pandas as pd

# Importa la configuración global
from src.config import config


class EvaluadorRiesgo:
    """Módulo 4: Cálculo de riesgo (Reglas de negocio)."""

    def __init__(self) -> None:
        # Carga las reglas del YAML al instanciar la clase
        self.reglas: Dict[str, Any] = config.reglas_riesgo

    def asignar_riesgo(self, estado: str) -> str:
        # Lógica Hardcoded inicial
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

    def ejecutar_evaluacion(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or "estado_actual" not in df.columns:
            raise ValueError("El DataFrame no tiene la columna 'estado_actual'")

        df_riesgo = df.copy()
        df_riesgo["nivel_riesgo"] = df_riesgo["estado_actual"].apply(
            self.asignar_riesgo
        )
        return df_riesgo
