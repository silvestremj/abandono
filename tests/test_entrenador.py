import joblib
import pandas as pd

from src.entrenador_modelos import EntrenadorModelos
from src.preparador_datos import PreparadorDatos


class TestEntrenadorModelos:
    """Tests del entrenamiento por hitos bimestrales, centrados en el split
    train/test estratificado y su fallback ante clases minoritarias muy pequeñas."""

    def setup_method(self):
        # No necesitamos datos crudos: filtrar_columnas_por_bimestre no usa self.datasets.
        self.preparador = PreparadorDatos(datasets_dict={})

    def _dataset(self, n_positivos: int, n_negativos: int) -> pd.DataFrame:
        filas = [
            {"nota_b1": 3.0, "asist_b1": 0.30, "target_ml": 1.0}
            for _ in range(n_positivos)
        ] + [
            {"nota_b1": 8.0, "asist_b1": 0.90, "target_ml": 0.0}
            for _ in range(n_negativos)
        ]
        return pd.DataFrame(filas)

    def test_entrena_con_split_estratificado(self, tmp_path):
        """Con muestras suficientes de ambas clases, el entrenamiento debe
        completarse usando el split estratificado (sin necesidad de fallback)."""
        df_ml = self._dataset(n_positivos=10, n_negativos=10)
        entrenador = EntrenadorModelos(output_dir=str(tmp_path))

        df_metricas = entrenador.entrenar_y_guardar(df_ml, self.preparador)

        assert not df_metricas.empty
        assert len(df_metricas) == 6  # un modelo por bimestre (1 a 6, config por defecto)
        assert (tmp_path / "arbol_b1.pkl").exists()
        assert (tmp_path / "columnas_b1.pkl").exists()

    def test_fallback_sin_estratificar_con_clase_minoritaria_insuficiente(
        self, tmp_path, capsys
    ):
        """Con una única muestra de la clase minoritaria, stratify=y no es viable
        (train_test_split exige al menos 2 muestras por clase). El entrenamiento
        debe completarse igualmente mediante el fallback sin lanzar excepción."""
        df_ml = self._dataset(n_positivos=1, n_negativos=20)
        entrenador = EntrenadorModelos(output_dir=str(tmp_path))

        df_metricas = entrenador.entrenar_y_guardar(df_ml, self.preparador)

        assert not df_metricas.empty
        salida = capsys.readouterr().out
        assert "no se pudo estratificar" in salida.lower()

    def _dataset_con_outliers(self) -> pd.DataFrame:
        """Dataset sintético donde la mayoría de filas son perfectamente
        separables por nota_b1/asist_b1, pero se añaden unas pocas filas
        "outlier" con combinaciones de características únicas y etiqueta
        contraria a la de su vecindario. Sin min_samples_leaf, un árbol de
        max_depth=5 tiende a aislar estos outliers en hojas de 1-2 muestras
        -- el mismo patrón de sobreajuste detectado empíricamente en los
        modelos reales del proyecto (ver docs/referencia/entrenador_modelos.md,
        "Limitación conocida: hojas de 1-2 muestras")."""
        filas = [
            {
                "nota_b1": 2.0 + (i % 5) * 0.3,
                "asist_b1": 0.15 + (i % 5) * 0.03,
                "target_ml": 1.0,
            }
            for i in range(35)
        ] + [
            {
                "nota_b1": 7.0 + (i % 5) * 0.3,
                "asist_b1": 0.80 + (i % 5) * 0.03,
                "target_ml": 0.0,
            }
            for i in range(35)
        ]
        # Outliers: valores únicos de características con etiqueta contraria
        # al rango en el que caen (fuerzan al árbol, sin restricción, a
        # aislarlos en hojas minúsculas).
        filas += [
            {"nota_b1": nota, "asist_b1": 0.85 + i * 0.01, "target_ml": 1.0}
            for i, nota in enumerate([8.5, 8.6, 8.7, 8.8, 8.9])
        ]
        filas += [
            {"nota_b1": nota, "asist_b1": 0.20 + i * 0.01, "target_ml": 0.0}
            for i, nota in enumerate([2.5, 2.6, 2.7, 2.8, 2.9])
        ]
        return pd.DataFrame(filas)

    def test_ninguna_hoja_tiene_menos_de_min_samples_leaf(self, tmp_path):
        """Regresión: se detectó empíricamente en los modelos reales del
        proyecto que, sin min_samples_leaf, hasta el 58% de las hojas de un
        árbol quedaban respaldadas por 1-2 muestras de entrenamiento -- una
        "confianza" del 100%/0% sin respaldo estadístico real (caso más
        grave: un alumno cuya predicción ALTO dependía de una única muestra
        de entrenamiento, ver docs/arquitectura.md "Limitaciones Conocidas").
        Este test protege el fix (min_samples_leaf=3) ante futuros
        reentrenamientos con datos distintos: ningún nodo hoja de ningún
        árbol bimestral debe tener menos de 3 muestras de entrenamiento."""
        df_ml = self._dataset_con_outliers()
        entrenador = EntrenadorModelos(output_dir=str(tmp_path))

        df_metricas = entrenador.entrenar_y_guardar(df_ml, self.preparador)
        assert not df_metricas.empty

        for b in range(1, 7):
            modelo = joblib.load(tmp_path / f"arbol_b{b}.pkl")
            assert modelo.get_params()["min_samples_leaf"] == 3, (
                f"El árbol del Bimestre {b} no tiene min_samples_leaf=3 "
                "configurado en DecisionTreeClassifier."
            )

            tree_ = modelo.tree_
            hojas_insuficientes = [
                (nid, int(tree_.n_node_samples[nid]))
                for nid in range(tree_.node_count)
                if tree_.children_left[nid] == tree_.children_right[nid]
                and tree_.n_node_samples[nid] < 3
            ]
            assert not hojas_insuficientes, (
                f"El árbol del Bimestre {b} tiene hojas con menos de 3 "
                f"muestras de entrenamiento (nodo, muestras): "
                f"{hojas_insuficientes}. Esto reproduce el sobreajuste ya "
                "documentado en docs/referencia/entrenador_modelos.md -- "
                "revisa que min_samples_leaf siga fijado en "
                "DecisionTreeClassifier."
            )
