import pandas as pd

from src.analizador_datos import Analizador
from src.config import config


class TestAnalizador:
    """Tests del análisis descriptivo sobre la tabla maestra."""

    def setup_method(self):
        """Se ejecuta antes de cada test. Prepara el entorno."""
        self.analizador = Analizador()

    def test_media_global_sobre_todos_los_bimestres(self):
        """La media debe agregar todas las notas de todos los bimestres modelados."""
        df_test = pd.DataFrame(
            {
                "id_alumno": [1, 2],
                "nota_b1": [10.0, 6.0],
                "nota_b2": [4.0, 8.0],
            }
        )

        stats = self.analizador.calcular_estadisticas_basicas(df_test)

        # Media global = (10 + 6 + 4 + 8) / 4 = 7.0, no la media de 'nota_b1' (8.0)
        assert stats["media_aritmetica"] == 7.0
        assert stats["columnas_analizadas"] == ["nota_b1", "nota_b2"]
        assert stats["medias_por_bimestre"] == {"nota_b1": 8.0, "nota_b2": 6.0}
        assert stats["total_alumnos"] == 2

    def test_bimestres_fuera_del_modelo_quedan_excluidos(self):
        """Los bimestres posteriores a 'max_bimestre' se conservan pero no se analizan."""
        max_bimestre = config.machine_learning.get("max_bimestre", 6)
        df_test = pd.DataFrame(
            {
                "nota_b1": [10.0, 6.0],
                "nota_b2": [4.0, 8.0],
                # Bimestres conservados por trazabilidad, fuera del modelo
                "nota_b7": [0.0, 0.0],
                "nota_b8": [0.0, 0.0],
                "nota_b10": [0.0, 0.0],
            }
        )

        stats = self.analizador.calcular_estadisticas_basicas(df_test)

        assert max_bimestre == 6
        assert stats["columnas_analizadas"] == ["nota_b1", "nota_b2"]
        for col in ("nota_b7", "nota_b8", "nota_b10"):
            assert col not in stats["medias_por_bimestre"]
        # Si los ceros de b7 en adelante entraran, la media caería a 3.5
        assert stats["media_aritmetica"] == 7.0

    def test_columnas_que_no_son_notas_de_bimestre_se_ignoran(self):
        """El patrón anclado descarta columnas que solo contienen 'nota_' como subcadena."""
        df_test = pd.DataFrame(
            {
                "nota_b1": [10.0, 6.0],
                "nota_media_final": [1.0, 1.0],
                "media_nota_b1": [1.0, 1.0],
            }
        )

        stats = self.analizador.calcular_estadisticas_basicas(df_test)

        assert stats["columnas_analizadas"] == ["nota_b1"]
        assert stats["media_aritmetica"] == 8.0

    def test_nulos_ignorados_en_la_media(self):
        """Los valores nulos no deben contar en la media global ni en las parciales."""
        df_test = pd.DataFrame(
            {
                "nota_b1": [10.0, None],
                "nota_b2": [4.0, 7.0],
            }
        )

        stats = self.analizador.calcular_estadisticas_basicas(df_test)

        # (10 + 4 + 7) / 3 = 7.0
        assert stats["media_aritmetica"] == 7.0
        assert stats["medias_por_bimestre"] == {"nota_b1": 10.0, "nota_b2": 5.5}
        assert stats["total_alumnos"] == 2

    def test_dataframe_vacio_devuelve_error(self):
        """Salida defensiva: DataFrame vacío o None."""
        assert "error" in self.analizador.calcular_estadisticas_basicas(pd.DataFrame())
        assert "error" in self.analizador.calcular_estadisticas_basicas(None)

    def test_sin_columnas_de_notas_devuelve_warning(self):
        """Salida defensiva: la tabla no trae ninguna nota por bimestre."""
        df_test = pd.DataFrame({"id_alumno": [1, 2], "estado_actual": ["En curso"] * 2})

        stats = self.analizador.calcular_estadisticas_basicas(df_test)

        assert stats["warning"] == "No hay columnas de notas para analizar"
