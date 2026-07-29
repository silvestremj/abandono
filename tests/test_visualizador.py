import os

from streamlit.testing.v1 import AppTest

RUTA_APP = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "src", "visualizador.py"
)

PESTANAS_ESPERADAS = [
    "Carga y ejecución",
    "Resultados",
    "Alertas",
    "Árbol de decisión",
]


class TestVisualizadorAppTest:
    """Pruebas de la interfaz Streamlit con streamlit.testing.v1.AppTest
    (Espiral 4): validan que la app real -- no una réplica de su lógica --
    arranca y responde a interacción sin lanzar excepciones."""

    def test_arranca_sin_excepciones_y_renderiza_las_cuatro_pestanas(self):
        """La app debe cargar sin ninguna excepción no controlada y mostrar
        exactamente las 4 pestañas principales, en el orden esperado."""
        at = AppTest.from_file(RUTA_APP)
        at.run(timeout=60)

        assert not at.exception

        etiquetas_pestanas = [tab.label for tab in at.tabs]
        assert etiquetas_pestanas == PESTANAS_ESPERADAS

    def test_pestana_resultados_y_alertas_muestran_aviso_antes_de_ejecutar(self):
        """Antes de pulsar 'Ejecutar cálculos', Resultados y Alertas deben
        mostrar el aviso de que aún no hay datos, en vez de fallar al
        intentar leer un DataFrame inexistente en session_state."""
        at = AppTest.from_file(RUTA_APP)
        at.run(timeout=60)

        assert not at.exception
        assert any("Ejecuta los cálculos" in info.value for info in at.tabs[1].info)
        assert any("Ejecuta los cálculos" in info.value for info in at.tabs[2].info)

    def test_flujo_completo_ejecutar_pipeline_no_rompe_la_interfaz(self):
        """Pulsa 'Ejecutar cálculos' reutilizando los ficheros ya presentes en
        data/ (como si el usuario no hubiera subido ficheros nuevos) y
        ejecuta el pipeline real completo (ingesta, preparación, entrenamiento
        y evaluación). Verifica que no se lanza ninguna excepción no
        controlada y que la interfaz queda operativa con resultados reales."""
        at = AppTest.from_file(RUTA_APP)
        at.run(timeout=60)

        at.button[0].click().run(timeout=120)

        assert not at.exception
        assert any(
            "Pipeline completado con éxito" in success.value for success in at.success
        )

        # Tras ejecutar el pipeline, Resultados y Alertas dejan de mostrar el
        # aviso de "ejecuta los cálculos primero".
        assert not any("Ejecuta los cálculos" in info.value for info in at.tabs[1].info)
        assert not any("Ejecuta los cálculos" in info.value for info in at.tabs[2].info)

        # La pestaña del árbol sigue respondiendo tras el pipeline: el
        # selector de alumno se puebla con alumnos PRONÓSTICO reales.
        selector_alumno = next(
            sb
            for sb in at.tabs[3].selectbox
            if sb.label == "Resaltar ruta de un alumno (opcional)"
        )
        assert len(selector_alumno.options) > 1
