import os

from streamlit.testing.v1 import AppTest

from src.visualizador import resolver_clave_dataset, validar_estructura_dataset

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


class TestValidacionFicherosSubidos:
    """Pruebas de la validación de ficheros subidos por la interfaz web.

    Ejercitan directamente las funciones de módulo (sin levantar la app con
    AppTest): el filtro por nombre declarado en ``config.yaml`` y la
    comprobación de estructura que evita que un fichero con el nombre correcto
    pero el contenido equivocado sustituya al original."""

    def test_rechaza_fichero_con_nombre_no_declarado(self):
        """Un nombre que no figura en config.datasets no se resuelve a ninguna
        clave, así que nunca llega a escribirse en data/."""
        assert resolver_clave_dataset("notas_2026_v2_final.csv") is None
        assert resolver_clave_dataset("data/../otro.xlsx") is None

        # Los tres nombres declarados sí se resuelven a su clave.
        assert (
            resolver_clave_dataset("LSE_Notas_Estadistica_Bimestre.csv")
            == "notas_bimestre"
        )
        assert (
            resolver_clave_dataset("LSE_Inscrip_Baja_Recibido.csv") == "inscripciones"
        )
        assert resolver_clave_dataset("LSE_Notas_Inscrip_Baja_Actual.xlsx") == "actual"

    def test_rechaza_fichero_con_columnas_faltantes(self, tmp_path):
        """Un CSV con el nombre correcto pero sin las columnas mínimas se
        rechaza indicando cuáles faltan."""
        ruta = tmp_path / "LSE_Notas_Estadistica_Bimestre.csv"
        ruta.write_text(
            "n_siu;Estudio;nota_m;Comentario\n2273001;CEIoT;9.5;Bimestre: 1.0\n",
            encoding="latin1",
        )

        motivo = validar_estructura_dataset("notas_bimestre", str(ruta))

        assert motivo is not None
        assert "asist_m" in motivo

    def test_rechaza_notas_sin_bimestre_en_comentario(self, tmp_path):
        """Con todas las columnas presentes pero sin el número de bimestre en
        'Comentario', el pivotado por bimestre quedaría vacío: se rechaza."""
        ruta = tmp_path / "LSE_Notas_Estadistica_Bimestre.csv"
        ruta.write_text(
            "n_siu;Estudio;nota_m;asist_m;Comentario\n"
            "2273001;CEIoT;9.5;0.93;sin datos\n",
            encoding="latin1",
        )

        motivo = validar_estructura_dataset("notas_bimestre", str(ruta))

        assert motivo is not None
        assert "Comentario" in motivo

    def test_acepta_fichero_con_estructura_correcta(self, tmp_path):
        """Un CSV con el nombre, el separador, el encoding y las columnas
        esperadas se acepta (la función devuelve None: sin motivo de rechazo)."""
        ruta = tmp_path / "LSE_Notas_Estadistica_Bimestre.csv"
        ruta.write_text(
            "n_siu;Estudio;nota_m;asist_m;Comentario\n"
            "2273001;CEIoT;9.5;0.93;Bimestre: 1.0\n",
            encoding="latin1",
        )

        assert validar_estructura_dataset("notas_bimestre", str(ruta)) is None
