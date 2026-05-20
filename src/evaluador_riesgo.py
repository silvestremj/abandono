import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

# Importa la configuración global
from src.config import config


class EvaluadorRiesgo:
    """Módulo 4: Cálculo de riesgo mediante ML por Puntos de Control y Explicabilidad."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        # Carga las reglas del YAML al instanciar la clase
        self.reglas: Dict[str, Any] = config.reglas_riesgo

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
        """Carga bajo demanda (Lazy Loading) el modelo y las columnas de un bimestre específico."""
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
        """Determina el hito temporal actual del alumno basándose en sus notas registradas."""
        # Buscamos de atrás hacia adelante (del bimestre 6 al 1) cuál es el primero con datos válidos
        for b in range(6, 0, -1):
            col_nota = f"nota_b{b}"
            if col_nota in fila_alumno.index:
                valor_nota = fila_alumno[col_nota]
                # Si la nota no es nula, ni vacía, ni cero puro sin asistencia, asumimos que está en este hito
                if pd.notna(valor_nota) and valor_nota > 0:
                    return b
        return 1  # Por defecto, si no hay notas registradas aún, se evalúa con el modelo del Bimestre 1

    def _alinear_columnas(
        self, df_ml: pd.DataFrame, columnas_entrenamiento: list
    ) -> pd.DataFrame:
        """Asegura que el dataset de inferencia tenga la misma estructura que el de entrenamiento."""
        columnas_faltantes = [
            col for col in columnas_entrenamiento if col not in df_ml.columns
        ]

        if columnas_faltantes:
            df_faltantes = pd.DataFrame(
                0, index=df_ml.index, columns=columnas_faltantes
            )
            df_ml = pd.concat([df_ml, df_faltantes], axis=1)

        return df_ml[columnas_entrenamiento]

    def _extraer_justificacion(
        self, df_muestra: pd.DataFrame, modelo: Any, columnas_entrenamiento: list
    ) -> str:
        """Genera la regla en texto plano recorriendo el camino del árbol de decisión."""
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
        df_riesgo = df.copy()

        # 1. Filtramos alumnos activos (los únicos que necesitan predicción)
        def es_evaluable(estado: str) -> bool:
            est = str(estado).upper()
            return any(p in est for p in self.reglas.get("palabras_medio_riesgo", []))

        mask_evaluables = df_riesgo["estado_actual"].apply(es_evaluable)

        # Inicializamos columnas de resultados
        df_riesgo["probabilidad_abandono"] = np.nan
        df_riesgo["nivel_riesgo"] = "HISTORICO/NO_CALCULABLE"
        df_riesgo["justificacion_riesgo"] = "N/A"

        df_evaluar = df_riesgo[mask_evaluables].copy()

        if df_evaluar.empty:
            return df_riesgo

        # 2. Preprocesamiento base (Eliminación de IDs y codificación de categorías)
        cols_excluir = [
            "n_siu",
            "fecha",
            "fecha_nacimiento",
            "estado_actual",
            "año_estado",
            "comentario",
            "target",
            "target_ml",
            "nota",
            "asist",
        ]

        df_ml_base = df_evaluar.drop(
            columns=[c for c in cols_excluir if c in df_evaluar.columns],
            errors="ignore",
        )

        cols_categoricas = df_ml_base.select_dtypes(include=["object"]).columns.tolist()
        df_ml_base = pd.get_dummies(
            df_ml_base, columns=cols_categoricas, dummy_na=False, drop_first=True
        )

        # 3. Inferencia individualizada por hito temporal
        for idx in df_evaluar.index:
            fila_original = df_evaluar.loc[idx]

            # Detectar en qué hito temporal (bimestre) se encuentra este alumno concreto
            b_alumno = self._detectar_bimestre_alumno(fila_original)

            # Recuperar el modelo adaptado a su realidad temporal
            modelo_b, columnas_b = self._obtener_modelo_bimestre(b_alumno)

            if modelo_b is None or columnas_b is None:
                # Si el modelo de ese hito no existe, dejamos al alumno como no calculable por seguridad
                continue

            # Extraemos la fila preprocesada del alumno y la adaptamos al molde de su modelo
            df_muestra_ml = df_ml_base.loc[[idx]]
            X_inferencia = self._alinear_columnas(df_muestra_ml, columnas_b)

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
