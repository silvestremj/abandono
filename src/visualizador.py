import os
import sys
from datetime import datetime
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import plot_tree

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
if "df_metricas" not in st.session_state:
    st.session_state.df_metricas = None
if "total_etiquetados" not in st.session_state:
    st.session_state.total_etiquetados = 0
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
        print("  REPORTE DE RIESGO DE ABANDONO ACADÉMICO")
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
    # Visualización del Árbol de Decisión (XAI)
    # ======================================================================
    def graficar_arbol(
        self, bimestre: int, max_depth: int = 3, guardar_ruta: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        ruta_modelo = os.path.join(base_dir, "modelos", f"arbol_b{bimestre}.pkl")
        ruta_columnas = os.path.join(base_dir, "modelos", f"columnas_b{bimestre}.pkl")

        if not os.path.exists(ruta_modelo) or not os.path.exists(ruta_columnas):
            st.warning(
                f"No se encontró el modelo del bimestre {bimestre}. "
                "Ejecuta los cálculos primero."
            )
            return None

        modelo = joblib.load(ruta_modelo)
        columnas = joblib.load(ruta_columnas)

        fig, ax = plt.subplots(figsize=(14 + 4 * max_depth, 8 + 3 * max_depth))
        plot_tree(
            modelo,
            feature_names=columnas,
            class_names=["Continúa", "Abandono"],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            ax=ax,
            fontsize=10,
        )
        plt.suptitle(
            f"Árbol de Decisión — Bimestre {bimestre} (max_depth={max_depth})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if guardar_ruta:
            os.makedirs(os.path.dirname(guardar_ruta), exist_ok=True)
            fig.savefig(guardar_ruta, dpi=150, bbox_inches="tight")

        return fig

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
        log.registrar("VIZ", "Preparación completada")

        modelo_b1 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "modelos", "arbol_b1.pkl"
        )
        if not os.path.exists(modelo_b1):
            df_ml = preparador.preparar_dataset_ml(df_master)
            entrenador = EntrenadorModelos()
            st.session_state.df_metricas = entrenador.entrenar_y_guardar(df_ml, preparador)
            total_con_target = int(df_ml["target_ml"].dropna().shape[0])
            st.session_state.total_etiquetados = total_con_target
            log.registrar("VIZ", "Modelos entrenados")

        evaluador = EvaluadorRiesgo()
        df_riesgo = evaluador.ejecutar_evaluacion(df_master)
        log.registrar("VIZ", "Evaluación completada")

        # Si no se reentrenó, cargar métricas guardadas en disco
        if st.session_state.df_metricas is None:
            ruta_metricas = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "modelos", "metricas.pkl"
            )
            if os.path.exists(ruta_metricas):
                st.session_state.df_metricas = joblib.load(ruta_metricas)

        return df_riesgo

    # ======================================================================
    # Interfaz Streamlit
    # ======================================================================
    def generar_interfaz_web(self):
        st.title("Sistema predictivo de abandono académico")
        st.markdown("---")

        tab_carga, tab_resultados, tab_alertas, tab_arbol = st.tabs(
            ["Carga y ejecución", "Resultados", "Alertas", "Árbol de decisión"]
        )

        with tab_carga:
            self._render_carga()
        with tab_resultados:
            self._render_resultados()
        with tab_alertas:
            self._render_alertas()
        with tab_arbol:
            self._render_arbol()

    def _render_carga(self):
        st.header("1. Cargar archivos de datos")
        st.write("Selecciona los archivos CSV o XLSX con los datos académicos.")

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
        st.header("2. Ejecutar cálculos")

        if st.button("Ejecutar cálculos", type="primary"):
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
                    st.success("Pipeline completado con éxito.")
                except Exception as e:
                    st.error(f"Error en el pipeline: {e}")
                    self.logger.registrar("VIZ-UI", f"Error: {e}", "ERROR")

    def _render_resultados(self):
        if not st.session_state.pipeline_ok or st.session_state.df_riesgo is None:
            st.info("Ejecuta los cálculos en la pestaña 'Carga y ejecución'.")
            return

        df = st.session_state.df_riesgo
        df_metricas = st.session_state.df_metricas

        # ======================================================================
        # FASE 1 — ENTRENAMIENTO (datos etiquetados)
        # ======================================================================
        with st.expander("**Fase 1 — Entrenamiento** (datos etiquetados)", expanded=True):
            st.markdown(
                "Los **datos etiquetados** corresponden a alumnos históricos cuyo desenlace "
                "ya se conoce: **RECIBIDO** (etiqueta 0, continúa) o **ABANDONO/BAJA** "
                "(etiqueta 1, abandono). Con estos datos el modelo aprende a reconocer "
                "patrones asociados al abandono."
            )

            if df_metricas is not None and not df_metricas.empty:
                total_etiquet = int(df_metricas["Registros etiquetados"].iloc[0])
                total_train = int(df_metricas["Train (80%)"].iloc[0])
                total_test = int(df_metricas["Test (20%)"].iloc[0])
                abandono_train = int(df_metricas["Abandono en train"].iloc[0])
                continua_train = int(df_metricas["Continúa en train"].iloc[0])

                c1, c2, c3 = st.columns(3)
                c1.metric("Registros etiquetados", total_etiquet)
                c2.metric("Train (80%)", total_train)
                c3.metric("Test (20%)", total_test)

                st.progress(0.8, text="80% entrenamiento / 20% prueba")
                st.caption(
                    f"Composición del train: {abandono_train} abandono, "
                    f"{continua_train} continúa"
                )
            else:
                st.info("No hay métricas de entrenamiento disponibles.")

        # ======================================================================
        # FASE 2 — VALIDACIÓN (accuracy y recall)
        # ======================================================================
        with st.expander("**Fase 2 — Validación** (métricas por bimestre)", expanded=True):
            st.markdown(
                "Para cada bimestre se entrena un árbol de decisión independiente. "
                "La validación se realiza sobre el 20% de datos que el modelo **no vio** "
                "durante el entrenamiento. Se prioriza el **Recall (Abandono)** porque "
                "es la métrica de negocio más importante: queremos minimizar los falsos "
                "negativos (alumnos en riesgo que no son detectados)."
            )

            if df_metricas is not None and not df_metricas.empty:
                cols_mostrar_met = [
                    "Hito", "Accuracy", "Recall (Abandono)",
                    "Variable Clave", "Importancia"
                ]
                cols_met = [c for c in cols_mostrar_met if c in df_metricas.columns]
                st.dataframe(
                    df_metricas[cols_met], width="stretch", hide_index=True
                )

                recall_prom = df_metricas["Recall (Abandono)"].mean()
                acc_prom = df_metricas["Accuracy"].mean()
                st.caption(
                    f"Promedio global — Accuracy: {acc_prom:.2f}  |  "
                    f"Recall (Abandono): {recall_prom:.2f}"
                )
            else:
                st.info("No hay métricas de validación disponibles.")

        # ======================================================================
        # FASE 3 — PRONÓSTICO (datos sin etiqueta)
        # ======================================================================
        with st.expander("**Fase 3 — Pronóstico** (datos sin etiqueta)", expanded=True):
            st.markdown(
                "Los **datos sin etiqueta** corresponden a los estudiantes actualmente "
                "activos en estado **CURSO** o **PAUSA**, cuyo desenlace final "
                "(graduación o abandono) aún se desconoce. El modelo entrenado infiere "
                "su probabilidad de abandono basándose en los patrones aprendidos de los "
                "datos históricos etiquetados."
            )

            if "tipo_prediccion" in df.columns:
                df_pronostico = df[df["tipo_prediccion"] == "PRONÓSTICO"]
                n_pronostico = len(df_pronostico)
                n_historico = len(df[df["tipo_prediccion"] == "HISTÓRICO"])

                c1, c2 = st.columns(2)
                c1.metric("Alumnos pronosticados", n_pronostico)
                c2.metric("Alumnos históricos", n_historico)

                if n_pronostico > 0:
                    p_alto = len(df_pronostico[df_pronostico["nivel_riesgo"] == "ALTO"])
                    p_medio = len(df_pronostico[df_pronostico["nivel_riesgo"] == "MEDIO"])
                    p_bajo = len(df_pronostico[df_pronostico["nivel_riesgo"] == "BAJO"])

                    st.subheader("Distribución de riesgo pronosticado")
                    c_a, c_m, c_b = st.columns(3)
                    c_a.metric("Riesgo alto", p_alto)
                    c_m.metric("Riesgo medio", p_medio)
                    c_b.metric("Riesgo bajo", p_bajo)

                    st.caption(
                        "Estos alumnos **no tienen etiqueta conocida**. "
                        "El modelo asigna su nivel de riesgo basándose en su "
                        "rendimiento académico actual."
                    )

        # ======================================================================
        # Listado completo de alumnos
        # ======================================================================
        st.subheader("Listado completo de alumnos")

        cols_clave = [
            "n_siu",
            "estudio",
            "tipo_prediccion",
            "nivel_riesgo",
            "probabilidad_abandono",
            "justificacion_riesgo",
        ]
        cols_mostrar = [c for c in cols_clave if c in df.columns]
        st.dataframe(df[cols_mostrar], width="stretch", hide_index=True)

    def _render_alertas(self):
        if not st.session_state.pipeline_ok or st.session_state.df_riesgo is None:
            st.info("Ejecuta los cálculos para generar alertas.")
            return

        df = st.session_state.df_riesgo
        alumnos_alto = df[df["nivel_riesgo"] == "ALTO"]

        if alumnos_alto.empty:
            st.success("No se detectaron alumnos en riesgo alto de abandono.")
        else:
            st.error(
                f"Atención: {len(alumnos_alto)} alumno(s) en RIESGO ALTO de abandono."
            )

            for _, alumno in alumnos_alto.iterrows():
                with st.expander(
                    f"Alerta: {alumno.get('n_siu', '?')} - {alumno.get('estudio', '?')}"
                ):
                    st.write(
                        f"**Justificación:** {alumno.get('justificacion_riesgo', 'N/A')}"
                    )

            csv = alumnos_alto.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
            st.download_button(
                "Descargar alertas (CSV)",
                csv,
                f"alertas_abandono_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )

    def _render_arbol(self):
        st.header("Árbol de decisión por bimestre")
        st.write(
            "Cada bimestre tiene su propio árbol entrenado con los datos "
            "disponibles hasta ese hito temporal. Selecciona un bimestre "
            "para visualizar su estructura."
        )

        bimestre = st.selectbox(
            "Seleccionar bimestre", range(1, 7), format_func=lambda b: f"Bimestre {b}"
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            max_depth = st.slider(
                "Profundidad visual", min_value=1, max_value=5, value=3
            )

        if st.button("Generar gráfico del árbol", type="primary"):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            assets_dir = os.path.join(base_dir, "assets")
            ruta_png = os.path.join(assets_dir, f"arbol_b{bimestre}.png")

            with st.spinner("Generando árbol de decisión..."):
                fig = self.graficar_arbol(
                    bimestre=bimestre,
                    max_depth=max_depth,
                    guardar_ruta=ruta_png,
                )

            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)

                st.success(f"Gráfico guardado en assets/arbol_b{bimestre}.png")

                with open(ruta_png, "rb") as f:
                    st.download_button(
                        "Descargar imagen PNG",
                        f,
                        f"arbol_b{bimestre}.png",
                        "image/png",
                    )


if __name__ == "__main__":
    app = Visualizador()
    app.generar_interfaz_web()
