import pandas as pd

from src.evaluador_riesgo import EvaluadorRiesgo


class TestEvaluadorRiesgo:
    """Batería de tests unitarios adaptada al motor de Machine Learning (Espiral 3)."""

    def setup_method(self):
        """Se ejecuta antes de cada test. Prepara el entorno."""
        self.evaluador = EvaluadorRiesgo()

    def test_casos_historicos_no_se_calculan(self):
        """Verifica que los alumnos que ya han terminado (Recibido/Abandono) permanezcan intactos."""
        df_test = pd.DataFrame(
            [
                {"estado_actual": "Recibido", "nota_b1": 8.0, "asist_b1": 0.90},
                {"estado_actual": "Abandono", "nota_b1": 2.0, "asist_b1": 0.20},
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        # Deben devolver obligatoriamente "HISTORICO/NO_CALCULABLE" por seguridad metodológica
        assert resultado.loc[0, "nivel_riesgo"] == "HISTORICO/NO_CALCULABLE"
        assert resultado.loc[1, "nivel_riesgo"] == "HISTORICO/NO_CALCULABLE"
        assert pd.isna(resultado.loc[0, "probabilidad_abandono"])

    def test_inferencia_alumno_en_curso_b1(self):
        """Verifica que un alumno activo sea evaluado y reciba una justificación con prefijo de hito."""
        # Creamos un alumno "En curso" que solo tiene datos del Bimestre 1
        df_test = pd.DataFrame(
            [
                {
                    "estado_actual": "En curso",
                    "nota_b1": 3.0,
                    "asist_b1": 0.40,
                    "nota_b2": 0.0,
                    "asist_b2": 0.0,  # Futuro vacío
                    "nota_b3": 0.0,
                    "asist_b3": 0.0,
                }
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        # 1. El nivel de riesgo ya no debe ser el histórico predeterminado
        assert resultado.loc[0, "nivel_riesgo"] in ["ALTO", "MEDIO", "BAJO"]
        # 2. La probabilidad matemática debe haberse calculado
        assert pd.notna(resultado.loc[0, "probabilidad_abandono"])

        # 3. Si el riesgo requiere justificación, debe llevar la etiqueta del hito detectado [Hito B1]
        nivel = resultado.loc[0, "nivel_riesgo"]
        if nivel in ["ALTO", "MEDIO"]:
            # Forzamos el casting a str para que Pylance reconozca el método startswith
            justificacion = str(resultado.loc[0, "justificacion_riesgo"])
            assert justificacion.startswith("[Hito B1]")
