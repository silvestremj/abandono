"""
Módulo de integración, limpieza y preparación de datos.

Este módulo implementa las tareas de preprocesamiento esenciales:
normalización de texto, deduplicación, tratamiento de nulos, mapeo de
reglas de negocio, filtrado temporal por bimestre y codificación
One-Hot para modelos de machine learning.

Uso::

    from src.preparador_datos import PreparadorDatos
    preparador = PreparadorDatos(datasets)
    df_master = preparador.ejecutar_preparacion()
    df_ml = preparador.preparar_dataset_ml(df_master)
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import config

# ======================================================================
# Columnas no predictivas o con fuga de datos ("data leakage") que deben
# excluirse antes de entrenar/inferir con el modelo. Es la fuente única de
# verdad: tanto la preparación del dataset de entrenamiento
# (:meth:`PreparadorDatos.preparar_dataset_ml`) como el preprocesamiento de
# una fila individual en inferencia
# (:func:`src.evaluador_riesgo.preprocesar_fila_alumno`) la reutilizan, para
# evitar que ambas listas se desincronicen con el tiempo.
# ======================================================================
COLUMNAS_EXCLUIDAS_ML = [
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
    "nota",  # Media global (data leakage para modelos por bimestre)
    "asist",  # Media global (data leakage para modelos por bimestre)
    "tf_nota",
    "tf_asist",
    "tf_recibido",
    "cant",  # No accionables
    "Unnamed: 28",
    "Unnamed: 29",
    "Unnamed: 30",
    "Unnamed: 31",  # Basura del excel
    "ocupacion",  # No es predictiva y tiene muchos valores únicos (alta cardinalidad)
    "ciudad",  # Alta cardinalidad
]


class PreparadorDatos:
    """Módulo 2: Integración y Limpieza de datos.

    Attributes:
        datasets: Diccionario de DataFrames de entrada indexados por clave
            (``actual``, ``notas_bimestre``, etc.).
    """

    def __init__(self, datasets_dict: Dict[str, pd.DataFrame]) -> None:
        """Inicializa el preparador con los DataFrames de entrada.

        Args:
            datasets_dict: Diccionario ``{clave: DataFrame}`` con los datos crudos.
        """
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
        """Genera la tabla maestra de estudiantes.

        Aplica normalización, deduplicación, tratamiento de nulos, mapeo de
        reglas de negocio, unificación de notas por bimestre y estandarización
        de tipos.

        Returns:
            DataFrame con la tabla maestra lista para análisis y ML.
        """
        # 1. Base y Target (del archivo Excel)
        df_base = self.datasets["actual"].copy()

        # Normalizamos nombres de columnas por si acaso el Excel varía
        df_base.columns = [c.lower().strip() for c in df_base.columns]

        df_base["n_siu"] = self._normalizar(df_base, "n_siu")
        df_base["estudio"] = self._normalizar(df_base, "estudio")

        # PREPROCESAMIENTO: Tarea 3: Deduplicación
        # --------------------------------------------------------------
        # Elimina alumnado duplicado en el mismo estudio. Prevalece el
        # registro más informativo (con estado_actual y/o nota reales) y,
        # entre varios igualmente informativos, el último (asumiendo que es
        # el más actualizado).
        #
        # Antes se aplicaba "último registro" a secas, lo cual descarta
        # silenciosamente un desenlace real (p.ej. "Abandono"/"Recibido" con
        # su nota) cuando el fichero de origen trae, para el mismo alumno, una
        # fila en blanco POSTERIOR a la fila con los datos reales — se ha
        # detectado empíricamente en los datos de origen (~13% de los grupos
        # duplicados). Se ordena primero por "informatividad" (estable, así
        # que entre filas igual de informativas se conserva el orden
        # original) para que drop_duplicates(keep="last") nunca prefiera una
        # fila vacía sobre una con datos reales.
        if "nota" in df_base.columns:
            nota_dedup = pd.to_numeric(
                df_base["nota"].astype(str).str.replace(",", "."), errors="coerce"
            )
            tiene_dato = df_base["estado_actual"].notna() | nota_dedup.notna()
        else:
            tiene_dato = df_base["estado_actual"].notna()

        df_base = (
            df_base.assign(_tiene_dato=tiene_dato)
            .sort_values("_tiene_dato", kind="stable")
            .drop(columns="_tiene_dato")
        )

        filas_antes = len(df_base)
        df_base = df_base.drop_duplicates(subset=["n_siu", "estudio"], keep="last")
        filas_despues = len(df_base)

        if filas_antes != filas_despues:
            print(
                f"Preprocesamiento: Se eliminaron {filas_antes - filas_despues} registros duplicados en la base."
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
        """Genera un DataFrame numérico apto para scikit-learn.

        Aplica One-Hot Encoding, crea ``target_ml`` binario y elimina
        variables no predictivas y con data leakage.

        Args:
            df_maestro: Tabla maestra generada por :meth:`ejecutar_preparacion`.

        Returns:
            DataFrame preprocesado para entrenamiento de modelos.
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
        cols_excluir_existentes = [
            c for c in COLUMNAS_EXCLUIDAS_ML if c in df_ml.columns
        ]
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

    @staticmethod
    def filtrar_columnas_por_bimestre(
        df: pd.DataFrame, bimestre_corte: int
    ) -> pd.DataFrame:
        """
        Elimina dinámicamente del DataFrame las columnas de notas y asistencias
        que pertenecen a bimestres posteriores al hito temporal de corte.

        Es un método estático (no depende de ``self.datasets``) para poder
        reutilizarlo también fuera del entrenamiento — por ejemplo desde
        :meth:`src.evaluador_riesgo.EvaluadorRiesgo.ejecutar_evaluacion` para
        recortar el techo temporal visible al autodetectar el bimestre de un
        alumno — sin necesidad de instanciar :class:`PreparadorDatos` con un
        ``datasets_dict`` que no pintaría nada aquí.

        Args:
            df (pd.DataFrame): Dataset original con todas las variables temporales.
            bimestre_corte (int): Hito temporal actual (de 1 a 6).

        Returns:
            pd.DataFrame: Un nuevo DataFrame sin los bimestres futuros.
        """
        import re

        df_filtrado = df.copy()

        # Detectar todas las columnas que siguen el patrón nota_b{N} o asist_b{N}
        # y eliminar aquellas con N > bimestre_corte (funciona para cualquier N)
        columnas_a_eliminar = []
        patron = re.compile(r"^(nota|asist)_b(\d+)$")

        for col in df_filtrado.columns:
            m = patron.match(str(col))
            if m and int(m.group(2)) > bimestre_corte:
                columnas_a_eliminar.append(col)

        if columnas_a_eliminar:
            df_filtrado = df_filtrado.drop(columns=columnas_a_eliminar)

        return df_filtrado
