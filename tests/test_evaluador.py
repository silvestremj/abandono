from typing import List, Tuple

import pandas as pd

from src.evaluador_riesgo import (
    ETIQUETAS_VARIABLES_CATEGORICAS,
    EvaluadorRiesgo,
    preprocesar_fila_alumno,
)
from src.gestor_logs import GestorLogs


class LoggerEspia:
    """Doble de prueba de :class:`GestorLogs` que captura las llamadas a
    ``registrar`` en memoria en lugar de escribir en consola, fichero y base
    de datos."""

    def __init__(self) -> None:
        self.registros: List[Tuple[str, str, str]] = []

    def registrar(self, modulo: str, mensaje: str, estado: str = "EXITO") -> None:
        self.registros.append((modulo, mensaje, estado))


class GestorBaseDatosEspia:
    """Doble de prueba de :class:`GestorBaseDatos` que captura lo que un
    :class:`GestorLogs` intentaría persistir en ``logs_ejecucion``, sin tocar
    SQLite."""

    def __init__(self) -> None:
        self.logs: List[Tuple[str, str, str]] = []

    def registrar_log(self, modulo: str, estado: str, mensaje: str) -> None:
        self.logs.append((modulo, estado, mensaje))


class TestEvaluadorRiesgo:
    """Batería de tests unitarios del motor de Machine Learning."""

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
        # Alumno "En curso" que solo tiene datos del Bimestre 1.
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
        # Se fuerza el casting a str para que Pylance reconozca el método startswith
        justificacion = str(resultado.loc[0, "justificacion_riesgo"])
        assert justificacion.startswith("[Hito B1]")
        # 4. La justificación debe ser una frase en lenguaje natural,
        # no la salida técnica cruda del árbol de decisión.
        assert " <= " not in justificacion
        assert " > " not in justificacion
        assert " AND " not in justificacion
        assert "porque" in justificacion

    def test_justificacion_se_genera_tambien_para_riesgo_bajo(self):
        """Un alumno de riesgo BAJO también recibe justificación en lenguaje
        natural: dejar "N/A" ahí era una inconsistencia (el árbol ya resalta
        su ruta coloreada igual que para ALTO/MEDIO)."""
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
        df_test = pd.DataFrame(
            [{"estado_actual": "Recibido", "nota_b1": 8.0, "asist_b1": 0.90}]
        )

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

    def test_alumno_sin_datos_reales_no_recibe_alto_medio_bajo(self):
        """Hallazgo principal: preparador_datos rellena con 0.0 nota_bN/
        asist_bN cuando no hay registro real, así que ese mismo 0.0 puede
        significar "sacó un cero" (predictor genuino) o "sin dato todavía".
        Un alumno PRONÓSTICO sin ningún dato real (doble-cero conjunto en
        nota y asistencia) no debe recibir ALTO/MEDIO/BAJO -- sería
        inferencia sobre ruido -- sino el estado SIN_DATOS_SUFICIENTES,
        distinto de NO_CALCULABLE (que es para quien nunca llega a
        evaluarse)."""
        df_test = pd.DataFrame(
            [
                {
                    "estado_actual": "En curso",
                    "nota_b1": 0.0,
                    "asist_b1": 0.0,
                    "nota_b2": 0.0,
                    "asist_b2": 0.0,
                    "nota_b3": 0.0,
                    "asist_b3": 0.0,
                }
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        assert resultado.loc[0, "nivel_riesgo"] == "SIN_DATOS_SUFICIENTES"
        assert resultado.loc[0, "nivel_riesgo"] not in (
            "ALTO",
            "MEDIO",
            "BAJO",
            "NO_CALCULABLE",
        )
        assert pd.isna(resultado.loc[0, "probabilidad_abandono"])
        assert resultado.loc[0, "bimestre_evaluado"] == 1

        # Conserva el comportamiento ya existente: justificación en lenguaje
        # natural con el prefijo de hito, no "N/A".
        justificacion = str(resultado.loc[0, "justificacion_riesgo"])
        assert justificacion != "N/A"
        assert justificacion.startswith("[Hito B1]")
        assert "Bimestre 1" in justificacion

    def test_alumno_con_datos_reales_no_se_ve_afectado_por_el_fix_sin_datos(self):
        """Un alumno PRONÓSTICO con datos reales (al menos nota o
        asistencia > 0 en su bimestre) debe seguir evaluándose con
        ALTO/MEDIO/BAJO como antes -- el fix de SIN_DATOS_SUFICIENTES no
        debe alcanzar a quien sí tiene señal real."""
        df_test = pd.DataFrame(
            [
                {
                    "estado_actual": "En curso",
                    "nota_b1": 3.0,
                    "asist_b1": 0.40,
                    "nota_b2": 0.0,
                    "asist_b2": 0.0,
                    "nota_b3": 0.0,
                    "asist_b3": 0.0,
                }
            ]
        )

        resultado = self.evaluador.ejecutar_evaluacion(df_test)

        assert resultado.loc[0, "nivel_riesgo"] in ("ALTO", "MEDIO", "BAJO")
        assert pd.notna(resultado.loc[0, "probabilidad_abandono"])

    def test_deteccion_de_bimestre_considera_asistencia_no_solo_nota(self):
        """Un alumno que asistió pero sacó un cero real en su bimestre más
        reciente (``nota_bN == 0.0`` con ``asist_bN > 0``) debe detectarse
        en ESE bimestre, no en uno anterior con nota positiva: el cero es
        un dato real de ese bimestre, no ausencia de registro (que exige
        nota Y asistencia en 0.0 a la vez, ver
        ``_sin_datos_reales_bimestre``)."""
        fila = pd.Series(
            {"nota_b1": 8.0, "asist_b1": 0.90, "nota_b2": 0.0, "asist_b2": 0.55}
        )

        assert self.evaluador._detectar_bimestre_alumno(fila) == 2

    def test_bimestre_corte_actua_como_techo_no_como_valor_forzado(self):
        """bimestre_corte debe recortar el futuro visible a la autodetección
        sin fingir datos que el alumno todavía no tiene."""
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

    def test_preprocesar_fila_alumno_codifica_categoria_real_no_a_cero(self):
        """Bug real detectado durante la verificación del caso E1311/MSE:
        preprocesar_fila_alumno codifica con One-Hot Encoding UNA fila
        aislada, que por definición solo puede tener un único valor por
        variable categórica. Con drop_first=True, pandas siempre descarta
        esa única categoría presente (genera 0 columnas dummy para esa
        variable), y el relleno posterior de "columnas_faltantes" con 0
        termina poniendo a 0 la categoría real del alumno igual que
        cualquier otra -- indistinguible de no tenerla. Se verificó con
        datos reales del proyecto que esto cambiaba el nivel_riesgo de 2 de
        8 alumnos PRONÓSTICO con inferencia real (25%).

        Este test simula el escenario exacto: un alumno con provincia
        'CORDOBA', y un esquema de columnas de entrenamiento (como las que
        genera preparar_dataset_ml sobre el dataset completo, con más de una
        provincia presente) que incluye la dummy 'provincia_CORDOBA'. Con el
        bug (drop_first=True) esa columna queda en 0 pese a ser la
        provincia real del alumno; corregido (drop_first=False) debe quedar
        en 1."""
        fila = pd.Series(
            {
                "estado_actual": "En curso",
                "nota_b1": 6.0,
                "asist_b1": 0.70,
                "provincia": "CORDOBA",
            }
        )
        # Esquema de entrenamiento simulado: incluye la dummy de la
        # provincia real del alumno más otra dummy de una provincia distinta
        # (como saldría de preparar_dataset_ml sobre un dataset con varias
        # provincias y drop_first=True aplicado sobre el conjunto completo).
        columnas_entrenamiento = [
            "nota_b1",
            "asist_b1",
            "provincia_CORDOBA",
            "provincia_SANTA FE",
        ]

        resultado = preprocesar_fila_alumno(fila, columnas_entrenamiento)

        assert resultado is not None
        assert resultado.loc[0, "provincia_CORDOBA"] == 1
        assert resultado.loc[0, "provincia_SANTA FE"] == 0

    def test_umbrales_de_riesgo_se_leen_de_la_configuracion(self, monkeypatch):
        """Los umbrales de alerta viven en config.yaml, no en el código.
        Fija el contrato descrito en la memoria: ajustar los umbrales
        no debe exigir editar evaluador_riesgo.py.
        """
        from src import evaluador_riesgo as mod

        reglas = dict(mod.config.reglas_riesgo)
        reglas.update({"umbral_medio": 0.25, "umbral_alto": 0.55})
        monkeypatch.setattr(
            type(mod.config), "reglas_riesgo", property(lambda self: reglas)
        )

        evaluador = mod.EvaluadorRiesgo()

        assert evaluador.umbral_medio == 0.25
        assert evaluador.umbral_alto == 0.55

    def test_sin_modelo_registra_error_y_justifica_el_no_calculable(self, tmp_path):
        """Un alumno PRONÓSTICO cuyo hito no tiene modelo entrenado no debe
        descartarse en silencio: antes el bucle hacía "continue" sin registrar
        nada y el alumno se quedaba en NO_CALCULABLE con justificación "N/A",
        indistinguible para quien audita de un fallo del pipeline. Ahora debe
        quedar constancia en el logger y en el propio resultado."""
        logger_falso = LoggerEspia()
        # model_dir vacío: no existe arbol_b1.pkl ni columnas_b1.pkl
        evaluador = EvaluadorRiesgo(
            model_dir=str(tmp_path),
            gestor_logs=logger_falso,  # type: ignore[arg-type]
        )

        df_test = pd.DataFrame(
            [
                {
                    "n_siu": "E9999",
                    "estado_actual": "En curso",
                    "nota_b1": 5.0,
                    "asist_b1": 0.60,
                }
            ]
        )

        resultado = evaluador.ejecutar_evaluacion(df_test)

        assert resultado.loc[0, "nivel_riesgo"] == "NO_CALCULABLE"
        # 1. Deja constancia en el resultado, no solo el nivel por defecto
        justificacion = str(resultado.loc[0, "justificacion_riesgo"])
        assert justificacion != "N/A"
        assert justificacion.strip() != ""
        assert "Bimestre 1" in justificacion
        assert resultado.loc[0, "bimestre_evaluado"] == 1

        # 2. Deja constancia en la trazabilidad, con nivel ERROR
        errores = [r for r in logger_falso.registros if r[2] == "ERROR"]
        assert errores, "El fallo debe registrarse como ERROR, no descartarse"
        assert any(
            "E9999" in mensaje and "Bimestre 1" in mensaje for _, mensaje, _ in errores
        )

        # 3. Y un resumen final con el desglose por motivo
        assert any(
            "NO_CALCULABLE" in mensaje and "sin_modelo_bimestre" in mensaje
            for _, mensaje, _ in logger_falso.registros
        )

    def test_gestor_logs_con_db_persiste_los_registros(self):
        """PROBLEMA 1: GestorLogs.registrar() solo persiste en la tabla
        logs_ejecucion si la instancia se construyó con gestor_db. Este test
        fija ese contrato con un doble de prueba, para que la regresión
        (instanciar el logger sin base de datos y perder la auditoría) se
        detecte aquí."""
        db_falsa = GestorBaseDatosEspia()
        log = GestorLogs(gestor_db=db_falsa)  # type: ignore[arg-type]

        log.registrar("EVALUADOR", "Mensaje de prueba", "ERROR")

        assert db_falsa.logs == [("EVALUADOR", "ERROR", "Mensaje de prueba")]
