import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ConfigLoader, config
from src.entrenador_modelos import EntrenadorModelos
from src.evaluador_riesgo import EvaluadorRiesgo
from src.gestor_base_datos import GestorBaseDatos
from src.gestor_logs import GestorLogs
from src.ingestor_datos import IngestorDatos
from src.preparador_datos import PreparadorDatos

st.set_page_config(
    page_title="Predicción de Abandono Académico",
    page_icon="🎓",
    layout="wide",
)

if "df_riesgo" not in st.session_state:
    st.session_state.df_riesgo = None
if "pipeline_ok" not in st.session_state:
    st.session_state.pipeline_ok = False


class Visualizador:
    def __init__(
        self,
        configuracion: Optional[ConfigLoader] = None,
        gestor_logs: Optional[GestorLogs] = None,
    ):
        self.config = configuracion or config
        self.logger = gestor_logs or GestorLogs()

    # ======================================================================
    # Compatibilidad con pipeline de consola (main.py)
    # ======================================================================
    def mostrar_en_consola(self, df_riesgo: pd.DataFrame, stats: dict) -> None:
        if df_riesgo.empty:
            print("No hay datos para mostrar.")
            return
        print("\n" + "=" * 60)
        print("  REPORTE DE RIESGO DE ABANDONO ACADEMICO")
        print("=" * 60)
        conteo = df_riesgo["nivel_riesgo"].value_counts()
        for nivel in ["ALTO", "MEDIO", "BAJO", "NO_CALCULABLE"]:
            print(f"  {nivel}: {conteo.get(nivel, 0)}")

    def exportar_csv(self, df_riesgo: pd.DataFrame) -> Optional[str]:
        if df_riesgo.empty:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        nombre = f"resultado_riesgos_{ts}.csv"
        out_dir = self.config.paths.get("output_dir", "output")
        base = os.path.dirname(os.path.dirname(__file__))
        ruta = os.path.join(base, out_dir, nombre)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_riesgo.to_csv(ruta, index=False, sep=";", decimal=",")
        return ruta

    # ======================================================================
    # Pipeline completo
    # ======================================================================
    def _ejecutar_pipeline(self) -> pd.DataFrame:
        db = GestorBaseDatos()
        db.inicializar_tablas_fijas()
        log = GestorLogs(gestor_db=db)

        ingestor = IngestorDatos()
        datasets = ingestor.leer_datos()
        db.guardar_datos_raw(datasets)
        log.registrar("VIZ", "Ingesta completada")

        preparador = PreparadorDatos(datasets)
        df_master = preparador.ejecutar_preparacion()
        db.guardar_datos_master(df_master)
        log.registrar("VIZ", "Preparacion completada")

        modelo_b1 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "modelos", "arbol_b1.pkl"
        )
        if not os.path.exists(modelo_b1):
            df_ml = preparador.preparar_dataset_ml(df_master)
            entrenador = EntrenadorModelos()
            entrenador.entrenar_y_guardar(df_ml, preparador)
            log.registrar("VIZ", "Modelos entrenados")

        evaluador = EvaluadorRiesgo()
        df_riesgo = evaluador.ejecutar_evaluacion(df_master)
        log.registrar("VIZ", "Evaluacion completada")

        return df_riesgo

    # ======================================================================
    # Interfaz Streamlit
    # ======================================================================
    def generar_interfaz_web(self):
        st.title("Sistema Predictivo de Abandono Academico")
        st.markdown("---")

        tab_carga, tab_resultados, tab_alertas = st.tabs(
            ["Carga y Ejecucion", "Resultados", "Alertas"]
        )

        with tab_carga:
            self._render_carga()
        with tab_resultados:
            self._render_resultados()
        with tab_alertas:
            self._render_alertas()

    def _render_carga(self):
        st.header("1. Cargar archivos de datos")
        st.write("Selecciona los archivos CSV o XLSX con los datos academicos.")

        archivos = st.file_uploader(
            "Seleccionar archivos",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if archivos:
            st.success(f"{len(archivos)} archivo(s) seleccionado(s)")
            for f in archivos:
                st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

        st.markdown("---")
        st.header("2. Ejecutar calculos")

        if st.button("Ejecutar calculos", type="primary"):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            if archivos:
                for f in archivos:
                    ruta = os.path.join(data_dir, f.name)
                    with open(ruta, "wb") as fp:
                        fp.write(f.getbuffer())

            with st.spinner("Procesando pipeline de datos..."):
                try:
                    df_riesgo = self._ejecutar_pipeline()
                    st.session_state.df_riesgo = df_riesgo
                    st.session_state.pipeline_ok = True
                    st.success("Pipeline completado con exito.")
                except Exception as e:
                    st.error(f"Error en el pipeline: {e}")
                    self.logger.registrar("VIZ-UI", f"Error: {e}", "ERROR")

    def _render_resultados(self):
        if not st.session_state.pipeline_ok or st.session_state.df_riesgo is None:
            st.info("Ejecuta los calculos en la pestana 'Carga y Ejecucion'.")
            return

        df = st.session_state.df_riesgo

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros procesados", len(df))
        col2.metric("Riesgo Alto", len(df[df["nivel_riesgo"] == "ALTO"]))
        col3.metric("Riesgo Medio", len(df[df["nivel_riesgo"] == "MEDIO"]))
        col4.metric("Riesgo Bajo", len(df[df["nivel_riesgo"] == "BAJO"]))

        st.subheader("Listado de alumnos")

        cols_clave = [
            "n_siu",
            "estudio",
            "nivel_riesgo",
            "probabilidad_abandono",
            "justificacion_riesgo",
        ]
        cols_mostrar = [c for c in cols_clave if c in df.columns]
        st.dataframe(df[cols_mostrar], use_container_width=True, hide_index=True)

    def _render_alertas(self):
        if not st.session_state.pipeline_ok or st.session_state.df_riesgo is None:
            st.info("Ejecuta los calculos para generar alertas.")
            return

        df = st.session_state.df_riesgo
        alumnos_alto = df[df["nivel_riesgo"] == "ALTO"]

        if alumnos_alto.empty:
            st.success("No se detectaron alumnos en riesgo alto de abandono.")
        else:
            st.error(
                f"Atencion: {len(alumnos_alto)} alumno(s) en RIESGO ALTO de abandono."
            )

            for _, alumno in alumnos_alto.iterrows():
                with st.expander(
                    f"Alerta: {alumno.get('n_siu', '?')} - {alumno.get('estudio', '?')}"
                ):
                    st.write(
                        f"**Justificacion:** {alumno.get('justificacion_riesgo', 'N/A')}"
                    )

            csv = alumnos_alto.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
            st.download_button(
                "Descargar alertas (CSV)",
                csv,
                f"alertas_abandono_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )


if __name__ == "__main__":
    app = Visualizador()
    app.generar_interfaz_web()
