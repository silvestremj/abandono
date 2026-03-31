import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class EntrenadorModelos:
    """Módulo auxiliar: Entrenamiento del árbol de decisión (Caja Blanca)."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            # Crea una carpeta 'modelos' en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.output_dir = os.path.join(base_dir, "modelos")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.ruta_modelo = os.path.join(self.output_dir, "arbol_decision.pkl")
        self.ruta_columnas = os.path.join(self.output_dir, "columnas_entrenamiento.pkl")

    def entrenar_y_guardar(self, df_ml: pd.DataFrame) -> None:
        """Entrena el modelo de ML y lo guarda en disco."""

        # 1. Filtrar solo alumnos con estado histórico cerrado (Target 0.0 o 1.0)
        # Se descartan los NaN (alumnos en curso)
        df_entrenamiento = df_ml.dropna(subset=["target_ml"])

        if df_entrenamiento.empty:
            print("Error: No hay datos históricos para entrenar el modelo.")
            return

        # 2. Separar características (X) y objetivo (y)
        X = df_entrenamiento.drop(columns=["target_ml"])
        y = df_entrenamiento["target_ml"]

        # 3. División en conjunto de entrenamiento y prueba (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Configurar y entrenar el modelo de caja blanca
        # Limitamos la profundidad (max_depth=5) para que las reglas extraíbles sean legibles
        # class_weight="balanced" ayuda si hay desproporción entre abandonos y recibidos
        modelo = DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight="balanced"
        )
        modelo.fit(X_train, y_train)

        # 5. Evaluación básica por consola
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modelo entrenado. Exactitud en test: {acc:.2f}")
        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # 6. Extracción de la Importancia Global de Variables (XAI)
        importancias = pd.DataFrame(
            {"Variable": X.columns, "Importancia": modelo.feature_importances_}
        ).sort_values("Importancia", ascending=False)

        print("\n--- XAI: TOP 5 VARIABLES GLOBALES MÁS IMPORTANTES ---")
        print(
            importancias[importancias["Importancia"] > 0].head(5).to_string(index=False)
        )
        print("---------------------------------------------------\n")

        # 7. Guardar el modelo y la lista de columnas (Crucial para la fase de inferencia)
        joblib.dump(modelo, self.ruta_modelo)
        joblib.dump(X.columns.tolist(), self.ruta_columnas)
        print(f"Modelo guardado en: {self.ruta_modelo}")
