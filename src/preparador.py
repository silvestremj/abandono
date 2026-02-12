import numpy as np
import pandas as pd


class PreparadorDatos:
    """Módulo 2: Integración y Limpieza."""

    def __init__(self, datasets_dict):
        self.datasets = datasets_dict

    def _normalizar(self, df, col):
        return df[col].astype(str).str.strip().str.upper()

    def ejecutar_preparacion(self):
        # 1. Base y Target (del archivo Excel)
        df_base = self.datasets["actual"].copy()

        # Normalizamos nombres de columnas por si acaso el Excel varía
        df_base.columns = [c.lower().strip() for c in df_base.columns]

        df_base["n_siu"] = self._normalizar(df_base, "n_siu")
        df_base["estudio"] = self._normalizar(df_base, "estudio")

        def mapear_riesgo(estado):
            estado = str(estado).upper()
            if "RECIBIDO" in estado:
                return 0
            if "CURSO" in estado or "PAUSA" in estado:
                return 1
            if "ABANDONO" in estado or "LIBRE" in estado or "BAJA" in estado:
                return 2
            return np.nan

        df_base["target"] = df_base["estado_actual"].apply(mapear_riesgo)

        # 2. Notas por Bimestre (del archivo CSV)
        df_notas = self.datasets["notas_bimestre"].copy()
        df_notas.rename(columns={"Estudio": "estudio"}, inplace=True)
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
        print(f"Tabla Maestra generada con {len(tabla_maestra)} registros.")
        return tabla_maestra
