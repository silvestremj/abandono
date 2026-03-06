from typing import Any, Dict

import numpy as np
import pandas as pd

# Importa la configuración global
from src.config import config


class PreparadorDatos:
    """Módulo 2: Integración y Limpieza."""

    def __init__(self, datasets_dict: Dict[str, pd.DataFrame]) -> None:
        self.datasets: Dict[str, pd.DataFrame] = datasets_dict

    def _normalizar(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        PREPROCESAMIENTO: Tarea 2: Normalización de Texto
        ---------------------------------------------------------------
        1. Convierte a string.
        2. Elimina espacios en blanco a los lados (strip).
        3. Convierte a mayúsculas.
        4. Elimina tildes y caracteres diacríticos.
        ----------------------------------------------------------------
        """
        return (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.normalize("NFKD")  # Descompone caracteres con tilde (ej: Á -> A + ´)
            .str.encode("ascii", errors="ignore")  # Descarta la tilde por no ser ASCII
            .str.decode("utf-8")  # Vuelve a convertir el resultado a texto limpio
        )

    def ejecutar_preparacion(self) -> pd.DataFrame:
        # 1. Base y Target (del archivo Excel)
        df_base = self.datasets["actual"].copy()

        # Normalizamos nombres de columnas por si acaso el Excel varía
        df_base.columns = [c.lower().strip() for c in df_base.columns]

        df_base["n_siu"] = self._normalizar(df_base, "n_siu")
        df_base["estudio"] = self._normalizar(df_base, "estudio")

        # PREPROCESAMIENTO: Tarea 3: Deduplicación
        # --------------------------------------------------------------
        # Elimina alumnado duplicado en el mismo estudio. Prevalece el último registro (asumiendo que es el más actualizado).
        filas_antes = len(df_base)
        df_base = df_base.drop_duplicates(subset=["n_siu", "estudio"], keep="last")
        filas_despues = len(df_base)

        if filas_antes != filas_despues:
            print(
                f"🧹Preprocesamiento: Se eliminaron {filas_antes - filas_despues} registros duplicados en la base."
            )
        # --------------------------------------------------------------

        # PREPROCESAMIENTO: Tarea 5: Tratamiento de anomalías
        # --------------------------------------------------------------
        # Etiqueta alumnos sin estado para que aportan en el reporte final pero no distorsionen el cálculo.
        df_base["estado_actual"] = df_base["estado_actual"].fillna("SIN REGISTRO")
        # --------------------------------------------------------------

        # Extraer las reglas del YAML una sola vez
        reglas = config.reglas_riesgo

        def mapear_riesgo(estado: Any) -> float:
            estado = str(estado).upper()

            if any(p in estado for p in reglas.get("palabras_bajo_riesgo", [])):
                return 0.0
            if any(p in estado for p in reglas.get("palabras_medio_riesgo", [])):
                return 1.0
            if any(p in estado for p in reglas.get("palabras_alto_riesgo", [])):
                return 2.0

            return np.nan

        df_base["target"] = df_base["estado_actual"].apply(mapear_riesgo)

        # 2. Notas por Bimestre (del archivo CSV)
        df_notas = self.datasets["notas_bimestre"].copy()
        df_notas.rename(columns={"Estudio": "estudio"}, inplace=True)

        # Normalizamos también en la tabla de notas para asegurar el cruce correcto.
        df_notas["n_siu"] = self._normalizar(df_notas, "n_siu")
        df_notas["estudio"] = self._normalizar(df_notas, "estudio")

        df_notas["bimestre_n"] = (
            df_notas["Comentario"].str.extract(r"(\d+)").fillna(0).astype(int)
        )
        df_notas = df_notas[df_notas["bimestre_n"] > 0]

        df_pivot = df_notas.pivot_table(
            index=["n_siu", "estudio"],
            columns="bimestre_n",
            values=["nota_m", "asist_m"],
            aggfunc="first",
        )
        df_pivot.columns = [
            f"{c[0].split('_')[0]}_b{int(c[1])}" for c in df_pivot.columns
        ]
        df_pivot.reset_index(inplace=True)

        # 3. Join Final
        tabla_maestra = pd.merge(df_base, df_pivot, on=["n_siu", "estudio"], how="left")

        # PREPROCESAMIENTO - Tarea 1: Tratamiento de Nulos
        # --------------------------------------------------------------
        # Identifica las columnas que corresponden a los bimestres (contienen "_b")
        # Ejemplo: "nota_b1", "asist_b2".
        cols_bimestres = [c for c in tabla_maestra.columns if "_b" in str(c)]

        # Rellenar los nulos (NaN) de esas columnas con 0, asumiendo que la ausencia de nota o asistencia implica 0.
        tabla_maestra[cols_bimestres] = tabla_maestra[cols_bimestres].fillna(0.0)
        # --------------------------------------------------------------

        # PREPROCESAMIENTO - Tarea 4: Estandarización de Tipos (Type Casting)
        # ---------------------------------------------------------------
        # Aseguramos que las columnas de bimestres sean numéricas (float).
        for col in cols_bimestres:
            # errors='coerce' fuerza la conversión y, si hay algún texto raro (ej. "N/A"), lo pasa a NaN, que luego rellenamos con 0.0.
            tabla_maestra[col] = pd.to_numeric(
                tabla_maestra[col], errors="coerce"
            ).fillna(0.0)

        # Aseguramos que la columna target sea numérica (float).
        if "target" in tabla_maestra.columns:
            tabla_maestra["target"] = tabla_maestra["target"].astype(float)
        # ---------------------------------------------------------------

        print(
            f"Tabla Maestra generada con {len(tabla_maestra)} registros y nulos tratados."
        )
        return tabla_maestra
