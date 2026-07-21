"""
Módulo de entrenamiento de modelos por hitos bimestrales.

Entrena un árbol de decisión independiente para cada bimestre (1-6),
almacena los modelos y sus columnas en disco, y genera un historial
de métricas comparativas entre hitos temporales.

Uso::

    from src.entrenador_modelos import EntrenadorModelos
    entrenador = EntrenadorModelos()
    df_metricas = entrenador.entrenar_y_guardar(df_ml, preparador)
"""

import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class EntrenadorModelos:
    """Entrenamiento del árbol de decisión por puntos de control bimestrales.

    Attributes:
        output_dir: Ruta absoluta al directorio donde se guardan los modelos
            y métricas (``modelos/``).
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """Inicializa el entrenador y crea el directorio de salida si no existe.

        Args:
            output_dir: Ruta al directorio de modelos. Si es ``None``, se crea
                la carpeta ``modelos/`` en la raíz del proyecto.
        """
        if output_dir is None:
            # Crea una carpeta 'modelos' en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.output_dir = os.path.join(base_dir, "modelos")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        # La ruta base ya no es un único archivo, la gestionamos dinámicamente en el bucle

    def entrenar_y_guardar(self, df_ml: pd.DataFrame, preparador: "PreparadorDatos") -> pd.DataFrame:
        """
        Entrena múltiples modelos independientes por cada hito bimestral
        y almacena sus métricas para permitir la comparación temporal.

        Args:
            df_ml (pd.DataFrame): Dataset maestro con todas las variables.
            preparador: Instancia del preparador de datos para usar el filtro temporal.

        Returns:
            pd.DataFrame: Historial de métricas comparativas de los modelos.
        """
        # 1. Filtrar solo alumnos con estado histórico cerrado (Target 0.0 o 1.0)
        df_entrenamiento_maestro = df_ml.dropna(subset=["target_ml"])

        if df_entrenamiento_maestro.empty:
            print("Error: No hay datos históricos para entrenar el modelo.")
            return pd.DataFrame()

        historial_metricas = []

        print("\n" + "=" * 55)
        print(" INICIANDO ENTRENAMIENTO POR PUNTOS DE CONTROL (XAI)")
        print("=" * 55)

        # 2. Bucle secuencial del bimestre 1 al 6 (o los que defina el sistema)
        for b in range(1, 7):
            # 3. Aplicar el filtro dinámico de la Fase 3
            df_bimestre = preparador.filtrar_columnas_por_bimestre(
                df_entrenamiento_maestro, b
            )

            # 4. Separar características (X) y objetivo (y)
            X = df_bimestre.drop(columns=["target_ml"])
            y = df_bimestre["target_ml"]

            # 5. División en conjunto de entrenamiento y prueba (80% / 20%) manteniendo tu semilla
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            total_etiquetados = len(y)
            total_train = len(y_train)
            total_test = len(y_test)
            abandono_train = int(y_train.sum())
            continua_train = total_train - abandono_train

            # 6. Configurar y entrenar el modelo de caja blanca con Cost-Sensitive Learning
            modelo = DecisionTreeClassifier(
                max_depth=5, random_state=42, class_weight="balanced"
            )
            modelo.fit(X_train, y_train)

            # 7. Evaluación de métricas clave (Accuracy y Recall de abandono)
            y_pred = modelo.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))

            # Forzamos a que si no hay datos de abandono en test, devuelva 0.0 en lugar de fallar
            try:
                rec = float(
                    recall_score(
                        y_test.astype(int),
                        y_pred.astype(int),
                        pos_label=1,
                        zero_division=0,
                    )
                )
            except Exception:
                rec = 0.0

            # 8. Extraer la variable más importante (Top 1 Global Importance)
            importancias = modelo.feature_importances_
            indices = importancias.argsort()[::-1]
            top_variable = (
                X.columns[indices[0]]
                if len(indices) > 0 and importancias[indices[0]] > 0
                else "Ninguna"
            )
            peso_variable = (
                float(importancias[indices[0]])
                if len(indices) > 0 and importancias[indices[0]] > 0
                else 0.0
            )

            # 9. Guardar modelo y columnas correspondientes a este hito temporal específico
            ruta_modelo_b = os.path.join(self.output_dir, f"arbol_b{b}.pkl")
            ruta_columnas_b = os.path.join(self.output_dir, f"columnas_b{b}.pkl")

            joblib.dump(modelo, ruta_modelo_b)
            joblib.dump(X.columns.tolist(), ruta_columnas_b)

            # 10. Almacenar resultados asegurando el redondeo limpio
            historial_metricas.append(
                {
                    "Hito": f"Bimestre {b}",
                    "Registros etiquetados": total_etiquetados,
                    "Train (80%)": total_train,
                    "Test (20%)": total_test,
                    "Abandono en train": abandono_train,
                    "Continúa en train": continua_train,
                    "Accuracy": round(acc, 2),
                    "Recall (Abandono)": round(rec, 2),
                    "Variable Clave": top_variable,
                    "Importancia": round(peso_variable, 2),
                }
            )

        print("Proceso de entrenamiento completado. Modelos guardados en disco.")
        print("=" * 55 + "\n")

        df_metricas = pd.DataFrame(historial_metricas)

        # Guardar métricas junto a los modelos para usarlas en ejecuciones posteriores
        ruta_metricas = os.path.join(self.output_dir, "metricas.pkl")
        joblib.dump(df_metricas, ruta_metricas)

        return df_metricas
