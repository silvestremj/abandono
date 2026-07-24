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
