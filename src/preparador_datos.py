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

        # PREPROCESAMIENTO - Tratamiento de variables geográficas (Criterio Escalada)
        # ---------------------------------------------------------------
        if "pais" in tabla_maestra.columns:
            tabla_maestra["pais"] = tabla_maestra["pais"].apply(
                lambda x: "ARGENTINA" if "ARGENTINA" in str(x).upper() else "OTRO PAIS"
            )

        if "provincia" in tabla_maestra.columns:
            # Reemplazar nulos matemáticos y cadenas vacías/falsas
            tabla_maestra["provincia"] = tabla_maestra["provincia"].fillna(
                "DESCONOCIDA"
            )
            tabla_maestra["provincia"] = tabla_maestra["provincia"].replace(
                ["", "NAN", "NONE", "NaN", "nan"], "DESCONOCIDA"
            )
            # Convertimos a mayúsculas para que detecte "Otro", "OTRO", "otro", etc.
            tabla_maestra["provincia"] = tabla_maestra["provincia"].apply(
                lambda x: "DESCONOCIDA" if "OTRO" in str(x).upper() else x
            )

        # ---------------------------------------------------------------
        # PREPROCESAMIENTO - Tarea 4: Estandarización de Tipos (Type Casting)
        # ---------------------------------------------------------------

        # Arreglar columnas numéricas que traen coma en lugar de punto
        cols_con_comas = ["nota", "sd_nota", "asist", "sd_asist"]
        for col in cols_con_comas:
            if col in tabla_maestra.columns:
                tabla_maestra[col] = (
                    tabla_maestra[col].astype(str).str.replace(",", ".")
                )
                tabla_maestra[col] = pd.to_numeric(
                    tabla_maestra[col], errors="coerce"
                ).fillna(0.0)

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

        # Mapeo de los códigos numéricos de ocupación a sus nombres descriptivos.
        mapping_ocupacion = {
            0: "No",
            1: "Estudiante/Becario",
            2: "Desarrollador/Programador",
            3: "Docente/Profesor",
            4: "Ingeniero",
            5: "Analista",
            6: "Trabajador a cuenta propia / Emprendedor",
            7: "Gestor/Director/Jefe",
            8: "Otros",
        }

        if "tipo_ocupacion" in tabla_maestra.columns:
            # Convertimos a entero primero por si acaso viene como float/string
            tabla_maestra["tipo_ocupacion"] = (
                pd.to_numeric(tabla_maestra["tipo_ocupacion"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
            # Aplicamos el cambio de nombre
            tabla_maestra["tipo_ocupacion"] = (
                tabla_maestra["tipo_ocupacion"]
                .map(mapping_ocupacion)
                .fillna("Otra/Desconocida")  # Para códigos fuera del rango esperado
            )

        print(
            f"Tabla Maestra generada con {len(tabla_maestra)} registros y nulos tratados."
        )
        return tabla_maestra

    def preparar_dataset_ml(self, df_maestro: pd.DataFrame) -> pd.DataFrame:
        """
        Genera un DataFrame numérico apto para entrenar o predecir con scikit-learn.
        Aplica One-Hot Encoding y define el target binario.
        """
        df_ml = df_maestro.copy()
        reglas = config.reglas_riesgo

        # 1. Creación del Target (1 = Abandono, 0 = Recibido, NaN = En curso/pausa)
        def asignar_target(estado: str) -> float:
            estado_str = str(estado).upper()
            if any(p in estado_str for p in reglas.get("palabras_alto_riesgo", [])):
                return 1.0
            elif any(p in estado_str for p in reglas.get("palabras_bajo_riesgo", [])):
                return 0.0
            else:
                return np.nan

        if "estado_actual" in df_ml.columns:
            df_ml["target_ml"] = df_ml["estado_actual"].apply(asignar_target)

        # 2. Eliminación de variables no predictivas o que generan "Data Leakage"
        cols_excluir = [
            "n_siu",
            "fecha",
            "fecha_nacimiento",
            "estado_actual",
            "año_estado",
            "comentario",
            "target",
            "fecha_baja",
            "estado_baja",
            "causa_baja",
            "comentario_baja",  # Fuga de datos
            "fecha_estado",  # Fuga de datos
            "cohorte",
            "year",
            "tf_nota",
            "tf_asist",
            "tf_recibido",
            "cant",  # No accionables
            "unnamed: 28",
            "unnamed: 29",
            "unnamed: 30",
            "unnamed: 31",  # Basura del excel
            "ocupacion",  # No es predictiva y tiene muchos valores únicos (alta cardinalidad)
            "ciudad",  # Alta cardinalidad
        ]
        cols_excluir_existentes = [c for c in cols_excluir if c in df_ml.columns]
        df_ml.drop(columns=cols_excluir_existentes, inplace=True, errors="ignore")

        # 3. Identificación de variables categóricas para One-Hot Encoding
        # Seleccionamos las columnas de tipo 'object' (strings) que quedan
        cols_categoricas = df_ml.select_dtypes(include=["object"]).columns.tolist()

        # 4. Aplicación de One-Hot Encoding
        # drop_first=True evita la multicolinealidad perfecta (dummy variable trap)
        df_ml = pd.get_dummies(
            df_ml, columns=cols_categoricas, dummy_na=False, drop_first=True
        )

        return df_ml
