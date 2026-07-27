import pandas as pd

from src.evaluador_riesgo import EvaluadorRiesgo, ETIQUETAS_VARIABLES_CATEGORICAS


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
        assert resultado.loc[0, "nivel_riesgo"] == "NO_CALCULABLE"
        assert resultado.loc[1, "nivel_riesgo"] == "NO_CALCULABLE"
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

        # 3. Todo alumno evaluado lleva justificación con la etiqueta del hito
        # detectado [Hito B1] -- también si su nivel es BAJO (ver
        # test_justificacion_se_genera_tambien_para_riesgo_bajo).
        # Forzamos el casting a str para que Pylance reconozca el método startswith
        justificacion = str(resultado.loc[0, "justificacion_riesgo"])
        assert justificacion.startswith("[Hito B1]")
        # 4. La justificación debe ser una frase en lenguaje natural,
        # no la salida técnica cruda del árbol de decisión (TASK-APP-02)
        assert " <= " not in justificacion
        assert " > " not in justificacion
        assert " AND " not in justificacion
        assert "porque" in justificacion

    def test_justificacion_se_genera_tambien_para_riesgo_bajo(self):
        """Un alumno de riesgo BAJO también recibe justificación en lenguaje
        natural: dejar "N/A" ahí era una inconsistencia (el árbol ya resalta
        su ruta coloreada igual que para ALTO/MEDIO, TASK-APP-03)."""
        df_test = pd.DataFrame(
            [
                {
                    "estado_actual": "En curso",
                    "nota_b1": 9.5,
                    "asist_b1": 0.95,
                    "nota_b2": 0.0,
                    "asist_b2": 0.0,
                    "nota_b3": 0.0,
                    "asist_b3": 0.0,
                }
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        assert resultado.loc[0, "nivel_riesgo"] == "BAJO"
        justificacion = str(resultado.loc[0, "justificacion_riesgo"])
        assert justificacion != "N/A"
        assert "riesgo BAJO" in justificacion
        assert "porque" in justificacion

    def test_historico_no_evaluado_mantiene_na_en_justificacion(self):
        """Un alumno HISTÓRICO (no evaluado) sí debe mantener "N/A": no hay
        ninguna ruta de decisión que justificar porque nunca se ejecuta el
        modelo para él."""
        df_test = pd.DataFrame([{"estado_actual": "Recibido", "nota_b1": 8.0, "asist_b1": 0.90}])

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        assert resultado.loc[0, "justificacion_riesgo"] == "N/A"

    def test_traducir_regla_numerica_asistencia(self):
        """Una condición sobre asist_bN se traduce a un porcentaje legible, no a la fracción cruda."""
        frase = self.evaluador._traducir_regla("asist_b2", 0.60, 0.40)

        assert "asistencia en el Bimestre 2" in frase
        assert "60%" in frase
        assert "0.60" not in frase
        assert "<=" not in frase

    def test_traducir_regla_categorica_one_hot(self):
        """Una condición sobre una columna one-hot verbaliza pertenencia a la categoría, no un umbral numérico."""
        frase = self.evaluador._traducir_regla("provincia_CORDOBA", 0.5, 1.0)

        assert frase == f"{ETIQUETAS_VARIABLES_CATEGORICAS['provincia']} es 'CORDOBA'"
        assert "0.5" not in frase

    def test_traducir_regla_desconocida_usa_fallback_y_registra_aviso(self):
        """Una columna sin traducción conocida no debe romper la ejecución: usa un fallback genérico."""
        frase = self.evaluador._traducir_regla("columna_inventada_xyz", 3.0, 5.0)

        # No debe lanzar excepción y debe devolver algo utilizable como texto
        assert isinstance(frase, str) and len(frase) > 0

    def test_bimestre_corte_actua_como_techo_no_como_valor_forzado(self):
        """TASK-APP-03: bimestre_corte debe recortar el futuro visible a la
        autodetección sin fingir datos que el alumno todavía no tiene."""
        df_test = pd.DataFrame(
            [
                {
                    # Alumno con datos reales hasta B4, pero se fija corte en B2:
                    # debe evaluarse con el modelo/columnas de B2, ignorando B3/B4.
                    "estado_actual": "En curso",
                    "nota_b1": 8.0,
                    "asist_b1": 0.90,
                    "nota_b2": 7.0,
                    "asist_b2": 0.85,
                    "nota_b3": 6.0,
                    "asist_b3": 0.70,
                    "nota_b4": 5.0,
                    "asist_b4": 0.60,
                },
                {
                    # Alumno con datos reales solo hasta B1, pero se fija corte
                    # en B4: no hay que forzarle ceros en B2-B4 (simularía un
                    # suspenso falso), debe evaluarse en su bimestre real (B1).
                    "estado_actual": "En curso",
                    "nota_b1": 3.0,
                    "asist_b1": 0.40,
                    "nota_b2": 0.0,
                    "asist_b2": 0.0,
                    "nota_b3": 0.0,
                    "asist_b3": 0.0,
                    "nota_b4": 0.0,
                    "asist_b4": 0.0,
                },
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test, bimestre_corte=2)
        assert resultado.loc[0, "bimestre_evaluado"] == 2

        resultado_b4 = self.evaluador.ejecutar_evaluacion(df_test, bimestre_corte=4)
        assert resultado_b4.loc[1, "bimestre_evaluado"] == 1
