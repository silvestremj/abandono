import pandas as pd

from src.evaluador_riesgo import EvaluadorRiesgo


class TestEvaluadorRiesgo:
    """Batería de tests unitarios para la lógica de riesgo (Espiral 2)."""

    def setup_method(self):
        """Se ejecuta antes de cada test. Prepara el entorno."""
        self.evaluador = EvaluadorRiesgo()

    def test_casos_cerrados_no_se_modifican(self):
        # Simulamos un alumno que ya está RECIBIDO (BAJO)
        row_bajo = pd.Series(
            {"riesgo_admin": "BAJO", "nota_media": 2.0, "asistencia_media": 0.10}
        )
        # Simulamos un alumno que ya ABANDONÓ (ALTO)
        row_alto = pd.Series(
            {"riesgo_admin": "ALTO", "nota_media": 9.0, "asistencia_media": 0.95}
        )

        assert self.evaluador._aplicar_heuristica(row_bajo) == "BAJO"
        assert self.evaluador._aplicar_heuristica(row_alto) == "ALTO"

    def test_alumnos_fantasma_mantienen_estado(self):
        # Simulamos un alumno con notas y asistencias a 0 absoluto
        row_fantasma = pd.Series(
            {"riesgo_admin": "MEDIO", "nota_media": 0.0, "asistencia_media": 0.0}
        )

        assert self.evaluador._aplicar_heuristica(row_fantasma) == "MEDIO"

    def test_penalizacion_por_mal_rendimiento(self):
        # Alumno en curso pero que suspende o falta mucho
        row_mala_nota = pd.Series(
            {"riesgo_admin": "MEDIO", "nota_media": 4.0, "asistencia_media": 0.90}
        )
        row_mala_asist = pd.Series(
            {"riesgo_admin": "MEDIO", "nota_media": 8.0, "asistencia_media": 0.20}
        )

        assert self.evaluador._aplicar_heuristica(row_mala_nota) == "ALTO"
        assert self.evaluador._aplicar_heuristica(row_mala_asist) == "ALTO"

    def test_bonificacion_por_excelencia(self):
        # Alumno en curso con notas excelentes
        row_excelente = pd.Series(
            {"riesgo_admin": "MEDIO", "nota_media": 8.5, "asistencia_media": 0.95}
        )

        assert self.evaluador._aplicar_heuristica(row_excelente) == "BAJO"

    def test_alumno_intermedio(self):
        # Alumno aprueba pero sin destacar, debería quedarse en MEDIO
        row_normal = pd.Series(
            {
                "riesgo_admin": "NO CALCULABLE",
                "nota_media": 6.0,
                "asistencia_media": 0.60,
            }
        )

        assert self.evaluador._aplicar_heuristica(row_normal) == "MEDIO"
