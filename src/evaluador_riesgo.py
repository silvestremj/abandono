"""
Módulo de evaluación de riesgo mediante inferencia ML.

Carga los modelos bimestrales previamente entrenados, detecta el hito
temporal de cada alumno, alinea las columnas al esquema de entrenamiento
y calcula la probabilidad de abandono con explicabilidad local (XAI).

Uso::

    from src.evaluador_riesgo import EvaluadorRiesgo
    evaluador = EvaluadorRiesgo()
    df_final = evaluador.ejecutar_evaluacion(df_master)
"""

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import config
from src.preparador_datos import COLUMNAS_EXCLUIDAS_ML

# ======================================================================
# Columnas que se excluyen del dataset antes de la inferencia.
#
# Reutiliza la misma lista de exclusión que el entrenamiento
# (:data:`src.preparador_datos.COLUMNAS_EXCLUIDAS_ML`) para que ambas fases
# del pipeline nunca puedan desincronizarse, y añade las columnas que solo
# existen en tiempo de inferencia (el target de entrenamiento y las columnas
# de resultado que genera :meth:`EvaluadorRiesgo.ejecutar_evaluacion`).
# ======================================================================
COLS_EXCLUIR = COLUMNAS_EXCLUIDAS_ML + [
    "target_ml",
    # Columnas añadidas por ejecutar_evaluacion — nunca deben entrar al modelo
    "tipo_prediccion",
    "nivel_riesgo",
    "probabilidad_abandono",
    "justificacion_riesgo",
]


def preprocesar_fila_alumno(
    fila: pd.Series,
    columnas_entrenamiento: list,
) -> Optional[pd.DataFrame]:
    """Preprocesa una fila individual de alumno para obtener el vector de
    características listo para inferencia con el modelo.

    Aplica el mismo pipeline que :meth:`EvaluadorRiesgo.ejecutar_evaluacion`:
    exclusión de columnas no relevantes, codificación *one-hot* de categóricas
    y alineación exacta al esquema de columnas usado durante el entrenamiento.

    Args:
        fila: Serie de pandas con los datos brutos del alumno (una fila del
            DataFrame **master**).
        columnas_entrenamiento: Lista de nombres de columnas que espera el
            modelo entrenado (cargadas desde ``columnas_b*.pkl``).

    Returns:
        DataFrame de una sola fila listo para ``model.predict_proba`` o
        ``model.decision_path``, o ``None`` si ocurre algún error.
    """
    try:
        df = pd.DataFrame([fila])

        df_ml = df.drop(
            columns=[c for c in COLS_EXCLUIR if c in df.columns],
            errors="ignore",
        )

        cols_categoricas = df_ml.select_dtypes(include=["object"]).columns.tolist()
        df_ml = pd.get_dummies(
            df_ml, columns=cols_categoricas, dummy_na=False, drop_first=True
        )

        columnas_faltantes = [
            col for col in columnas_entrenamiento if col not in df_ml.columns
        ]
        if columnas_faltantes:
            df_faltantes = pd.DataFrame(
                0, index=df_ml.index, columns=columnas_faltantes
            )
            df_ml = pd.concat([df_ml, df_faltantes], axis=1)

        return df_ml[columnas_entrenamiento]

    except Exception:
        return None


class EvaluadorRiesgo:
    """Evaluación de riesgo por hitos bimestrales con explicabilidad.

    Attributes:
        reglas: Diccionario con las reglas de negocio para clasificación
            de estados (bajo, medio, alto riesgo).
        model_dir: Ruta al directorio de modelos guardados.
        modelos_cache: Caché de modelos cargados por bimestre.
        columnas_cache: Caché de listas de columnas por bimestre.
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        """Inicializa el evaluador cargando reglas y configurando rutas.

        Args:
            model_dir: Directorio de modelos. Si es ``None``, se usa
                ``modelos/`` en la raíz del proyecto.
        """
        self.reglas: Dict[str, Any] = config.reglas_riesgo
        self.max_bimestre: int = config.machine_learning.get("max_bimestre", 6)

        if model_dir is None:
            # Crea una carpeta 'modelos' en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.model_dir = os.path.join(base_dir, "modelos")
        else:
            self.model_dir = model_dir

        # Los diccionarios almacenarán en caché los modelos a medida que se necesiten
        self.modelos_cache: Dict[int, Any] = {}
        self.columnas_cache: Dict[int, list] = {}

    def _obtener_modelo_bimestre(self, bimestre: int) -> tuple:
        """Carga bajo demanda (Lazy Loading) el modelo y las columnas de un bimestre.

        Args:
            bimestre: Número de bimestre (1-6).

        Returns:
            Tupla ``(modelo, columnas)`` o ``(None, None)`` si no existe.
        """
        if bimestre not in self.modelos_cache:
            ruta_modelo = os.path.join(self.model_dir, f"arbol_b{bimestre}.pkl")
            ruta_columnas = os.path.join(self.model_dir, f"columnas_b{bimestre}.pkl")

            if os.path.exists(ruta_modelo) and os.path.exists(ruta_columnas):
                self.modelos_cache[bimestre] = joblib.load(ruta_modelo)
                self.columnas_cache[bimestre] = joblib.load(ruta_columnas)
            else:
                return None, None

        return self.modelos_cache[bimestre], self.columnas_cache[bimestre]

    def _detectar_bimestre_alumno(self, fila_alumno: pd.Series) -> int:
        """Determina el hito temporal actual del alumno buscando desde el último
        bimestre configurado hacia B1.

        Args:
            fila_alumno: Fila del DataFrame con los datos del alumno.

        Returns:
            Número de bimestre detectado (1-``self.max_bimestre``). Por defecto 1.
        """
        # Buscamos de atrás hacia adelante (del último bimestre al 1) cuál es el primero con datos válidos
        for b in range(self.max_bimestre, 0, -1):
            col_nota = f"nota_b{b}"
            if col_nota in fila_alumno.index:
                valor_nota = fila_alumno[col_nota]
                # Si la nota no es nula, ni vacía, ni cero puro sin asistencia, asumimos que está en este hito
                if pd.notna(valor_nota) and valor_nota > 0:
                    return b
        return 1  # Por defecto, si no hay notas registradas aún, se evalúa con el modelo del Bimestre 1

    def _extraer_justificacion(
        self, df_muestra: pd.DataFrame, modelo: Any, columnas_entrenamiento: list
    ) -> str:
        """Extrae la justificación XAI recorriendo el camino del árbol de decisión.

        Args:
            df_muestra: Fila de datos del alumno (1 registro).
            modelo: Árbol de decisión entrenado.
            columnas_entrenamiento: Lista de columnas del modelo.

        Returns:
            cadena de texto con las reglas del camino (``AND`` separadas).
        """
        nodo_indicador = modelo.decision_path(df_muestra)
        nodos_ids = nodo_indicador.indices

        reglas = []
        valores = df_muestra.values

        for nodo_id in nodos_ids:
            if (
                modelo.tree_.children_left[nodo_id]
                == modelo.tree_.children_right[nodo_id]
            ):
                continue

            caracteristica_id = modelo.tree_.feature[nodo_id]
            umbral = modelo.tree_.threshold[nodo_id]
            nombre_caracteristica = columnas_entrenamiento[caracteristica_id]
            valor = valores[0, caracteristica_id]

            if valor <= umbral:
                reglas.append(f"{nombre_caracteristica} <= {umbral:.2f}")
            else:
                reglas.append(f"{nombre_caracteristica} > {umbral:.2f}")

        return " AND ".join(reglas)

    def ejecutar_evaluacion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta la evaluación de riesgo completa sobre la tabla maestra.

        Clasifica a cada alumno como HISTÓRICO, PRONÓSTICO o SIN REGISTRO.
        Para los PRONÓSTICOS, infiere probabilidad de abandono usando el modelo
        del bimestre correspondiente, asigna nivel de riesgo y genera justificación XAI.

        Args:
            df: Tabla maestra de estudiantes.

        Returns:
            DataFrame con columnas añadidas: ``tipo_prediccion``,
            ``probabilidad_abandono``, ``nivel_riesgo``, ``justificacion_riesgo``.
        """
        df_riesgo = df.copy()

        # 1. Clasificar alumnos: HISTÓRICO (tienen target conocido) vs PRONÓSTICO (activos sin etiqueta)
        def es_etiquetado(estado: str) -> bool:
            est = str(estado).upper()
            return any(p in est for p in self.reglas.get("palabras_bajo_riesgo", []) + self.reglas.get("palabras_alto_riesgo", []))

        def es_evaluable(estado: str) -> bool:
            est = str(estado).upper()
            return any(p in est for p in self.reglas.get("palabras_medio_riesgo", []))

        mask_etiquetados = df_riesgo["estado_actual"].apply(es_etiquetado)
        mask_evaluables = df_riesgo["estado_actual"].apply(es_evaluable)

        df_riesgo["tipo_prediccion"] = "SIN REGISTRO"
        df_riesgo.loc[mask_etiquetados, "tipo_prediccion"] = "HISTÓRICO"
        df_riesgo.loc[mask_evaluables, "tipo_prediccion"] = "PRONÓSTICO"

        # Inicializamos columnas de resultados
        df_riesgo["probabilidad_abandono"] = np.nan
        df_riesgo["nivel_riesgo"] = "NO_CALCULABLE"
        df_riesgo["justificacion_riesgo"] = "N/A"

        df_evaluar = df_riesgo[mask_evaluables].copy()

        if df_evaluar.empty:
            return df_riesgo

        # 2. Inferencia individualizada por hito temporal
        for idx in df_evaluar.index:
            fila_original = df_evaluar.loc[idx]

            # Detectar en qué hito temporal (bimestre) se encuentra este alumno concreto
            b_alumno = self._detectar_bimestre_alumno(fila_original)

            # Recuperar el modelo adaptado a su realidad temporal
            modelo_b, columnas_b = self._obtener_modelo_bimestre(b_alumno)

            if modelo_b is None or columnas_b is None:
                # Si el modelo de ese hito no existe, dejamos al alumno como no calculable por seguridad
                continue

            # Preprocesar la fila del alumno alineándola al esquema del modelo del bimestre actual
            X_inferencia = preprocesar_fila_alumno(fila_original, columnas_b)
            if X_inferencia is None:
                continue

            # Calcular probabilidad (Clase 1 = Abandono)
            prob = float(modelo_b.predict_proba(X_inferencia)[0, 1])
            df_riesgo.loc[idx, "probabilidad_abandono"] = prob

            # Aplicar reglas de negocio para los umbrales
            if prob >= 0.70:
                nivel = "ALTO"
            elif prob >= 0.40:
                nivel = "MEDIO"
            else:
                nivel = "BAJO"

            df_riesgo.loc[idx, "nivel_riesgo"] = nivel

            # 4. Explicabilidad Local (XAI) con el árbol correcto
            if nivel in ["ALTO", "MEDIO"]:
                justificacion = self._extraer_justificacion(
                    X_inferencia, modelo_b, columnas_b
                )
                df_riesgo.loc[idx, "justificacion_riesgo"] = (
                    f"[Hito B{b_alumno}] {justificacion}"
                )

        return df_riesgo
