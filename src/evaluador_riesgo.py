import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

# Importa la configuración global
from src.config import config


class EvaluadorRiesgo:
    """Módulo 4: Cálculo de riesgo mediante ML (Caja Blanca) y Explicabilidad."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        # Carga las reglas del YAML al instanciar la clase
        self.reglas: Dict[str, Any] = config.reglas_riesgo

        if model_dir is None:
            # Crea una carpeta 'modelos' en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.model_dir = os.path.join(base_dir, "modelos")
        else:
            self.model_dir = model_dir

        self.ruta_modelo = os.path.join(self.model_dir, "arbol_decision.pkl")
        self.ruta_columnas = os.path.join(self.model_dir, "columnas_entrenamiento.pkl")

        self.modelo = None
        self.columnas_entrenamiento = None

        # Carga silenciosa del modelo si ya ha sido entrenado
        if os.path.exists(self.ruta_modelo) and os.path.exists(self.ruta_columnas):
            self.modelo = joblib.load(self.ruta_modelo)
            self.columnas_entrenamiento = joblib.load(self.ruta_columnas)

    def _alinear_columnas(
        self, df_ml: pd.DataFrame, columnas_entrenamiento: list
    ) -> pd.DataFrame:
        """Asegura que el dataset de inferencia tenga la misma estructura que el de entrenamiento."""

        # 1. Identificamos qué columnas faltan en una sola pasada
        columnas_faltantes = [
            col for col in columnas_entrenamiento if col not in df_ml.columns
        ]

        # 2. Si faltan columnas, creamos un bloque de ceros y lo concatenamos de golpe
        if columnas_faltantes:
            df_faltantes = pd.DataFrame(
                0, index=df_ml.index, columns=columnas_faltantes
            )
            df_ml = pd.concat([df_ml, df_faltantes], axis=1)

        # 3. Devolvemos el DataFrame filtrado y ordenado
        return df_ml[columnas_entrenamiento]

    def _extraer_justificacion(
        self, df_muestra: pd.DataFrame, modelo: Any, columnas_entrenamiento: list
    ) -> str:
        """Genera la regla en texto plano recorriendo el camino del árbol de decisión."""
        # Le pasamos el DataFrame completo con sus nombres de columna
        nodo_indicador = modelo.decision_path(df_muestra)
        nodos_ids = nodo_indicador.indices

        reglas = []
        # Extraemos los valores puros solo para la comprobación interna rápida
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
        if self.modelo is None or self.columnas_entrenamiento is None:
            print(
                "Advertencia: El modelo ML no está entrenado. Ejecuta el entrenamiento primero."
            )
            return df

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

        # 2. Si hay alumnos activos, preparamos sus datos para la inferencia
        df_evaluar = df_riesgo[mask_evaluables].copy()

        if not df_evaluar.empty:
            # Reproducimos el preprocesamiento de ML (Eliminación de IDs y One-Hot Encoding)
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
            df_ml = df_evaluar.drop(
                columns=[c for c in cols_excluir if c in df_evaluar.columns],
                errors="ignore",
            )

            cols_categoricas = df_ml.select_dtypes(include=["object"]).columns.tolist()
            df_ml = pd.get_dummies(
                df_ml, columns=cols_categoricas, dummy_na=False, drop_first=True
            )

            # Pasamos los datos por el molde
            X_inferencia = self._alinear_columnas(df_ml, self.columnas_entrenamiento)

            # 3. Calculamos la probabilidad matemática (Clase 1 = Abandono)
            probabilidades = self.modelo.predict_proba(X_inferencia)[:, 1]

            for i, idx in enumerate(df_evaluar.index):
                prob = probabilidades[i]
                df_riesgo.loc[idx, "probabilidad_abandono"] = prob

                # Traducción de la probabilidad a lógicas de negocio (Umbrales)
                if prob >= 0.70:
                    nivel = "ALTO"
                elif prob >= 0.40:
                    nivel = "MEDIO"
                else:
                    nivel = "BAJO"

                df_riesgo.loc[idx, "nivel_riesgo"] = nivel

                # 4. Explicabilidad: Extraemos las reglas si el riesgo es relevante
                if nivel in ["ALTO", "MEDIO"]:
                    # Pasamos la fila como DataFrame (sin usar .to_numpy())
                    df_muestra = X_inferencia.iloc[[i]]
                    justificacion = self._extraer_justificacion(
                        df_muestra, self.modelo, self.columnas_entrenamiento
                    )
                    df_riesgo.loc[idx, "justificacion_riesgo"] = justificacion
        return df_riesgo
