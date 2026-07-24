"""
Módulo de visualización y presentación de resultados.

Proporciona dos interfaces de salida:
- **Consola**: reporte de texto y exportación CSV.
- **Streamlit (Web)**: interfaz interactiva con tabs de carga, resultados,
  alertas y visualización de árboles de decisión (XAI).

Uso en consola::

    from src.visualizador import Visualizador
    vista = Visualizador()
    vista.mostrar_en_consola(df_final, stats)
    vista.exportar_csv(df_final)

Uso en Streamlit::

    python -m src.visualizador
"""

import os
import sys
from datetime import datetime
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ConfigLoader, config
from src.entrenador_modelos import EntrenadorModelos
from src.evaluador_riesgo import EvaluadorRiesgo, preprocesar_fila_alumno
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
        """Inicializa el visualizador con configuración y logger opcionales.

        Args:
            configuracion: Instancia de :class:`ConfigLoader`. Si es ``None``
                se usa la instancia global ``config``.
            gestor_logs: Instancia de :class:`GestorLogs`. Si es ``None``
                se crea una nueva.
        """
        self.config = configuracion or config
        self.logger = gestor_logs or GestorLogs()

    # ======================================================================
    # Compatibilidad con pipeline de consola (main.py)
    # ======================================================================
    def mostrar_en_consola(self, df_riesgo: pd.DataFrame, stats: dict) -> None:
        """Muestra un reporte de texto en la consola con el conteo por nivel de riesgo.

        Args:
            df_riesgo: DataFrame con la columna ``nivel_riesgo``.
            stats: Estadísticas calculadas por el :class:`Analizador`.
        """
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
        """Exporta el DataFrame de riesgos a un archivo CSV con timestamp.

        Args:
            df_riesgo: DataFrame con los resultados de evaluación.

        Returns:
            Ruta absoluta del archivo CSV generado, o ``None`` si está vacío.
        """
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
    _COLOR_RIESGO = {
        "ALTO": "#d62728",
        "MEDIO": "#ff7f0e",
        "BAJO": "#2ca02c",
    }
    _COLOR_NO_RUTA = "#F5F5F5"
    _COLOR_BORDE_NO_RUTA = "#D0D0D0"
    _COLOR_TEXTO_NO_RUTA = "#999999"

    def _custom_plot_tree(
        self,
        modelo,
        columnas: list,
        max_depth: int,
        ax,
        path_nodes: Optional[set] = None,
        path_color: Optional[str] = None,
        fontsize: float = 8.0,
    ):
        """Dibuja un árbol de decisión con matplotlib, permitiendo resaltar
        nodos individuales según una ruta de decisión.

        Args:
            modelo: ``DecisionTreeClassifier`` entrenado.
            columnas: Nombres de las características usadas en el modelo.
            max_depth: Profundidad máxima a renderizar.
            ax: Ejes de matplotlib donde se dibuja el árbol.
            path_nodes: Conjunto de IDs de nodos que forman parte de la ruta
                de decisión a resaltar. Si es ``None``, se dibuja el árbol
                completo sin resaltar.
            path_color: Color hex para los nodos/aristas de la ruta activa.
            fontsize: Tamaño de fuente para el texto de los nodos.
        """
        fontsize = max(5.5, min(fontsize, 9.0))
        tree_ = modelo.tree_
        ch_left = tree_.children_left
        ch_right = tree_.children_right
        feature = tree_.feature
        threshold = tree_.threshold
        impurity = tree_.impurity
        value = tree_.value
        n_samples = tree_.n_node_samples

        # 1. Recolectar nodos visibles (dentro de max_depth)
        visible = []  # (node_id, depth)
        stack = [(0, 0)]
        while stack:
            nid, depth = stack.pop()
            if depth > max_depth:
                continue
            visible.append((nid, depth))
            if ch_left[nid] != ch_right[nid]:
                stack.append((ch_right[nid], depth + 1))
                stack.append((ch_left[nid], depth + 1))

        if not visible:
            return
        visible_set = {nid for nid, _ in visible}

        # 2. Calcular posiciones x usando solo nodos terminales visibles
        #
        # Estrategia:
        #   a) Recorrido in-order limitado a max_depth: recoger nodos terminales
        #      (hojas reales o internos truncados en max_depth) en orden.
        #   b) Asignarles x uniformemente espaciados en (0, 1).
        #   c) Nodos internos no truncados: x = punto medio entre hijos (bottom-up).
        #
        # Esto evita que hojas profundas del árbol completo compriman a los nodos
        # visibles en el rango [0,1].

        bottom_in_order = []

        def _collect_bottom(nid, depth):
            if depth > max_depth:
                return
            is_internal = ch_left[nid] != ch_right[nid]
            if is_internal:
                _collect_bottom(ch_left[nid], depth + 1)
            if not is_internal or depth == max_depth:
                bottom_in_order.append(nid)
            if is_internal:
                _collect_bottom(ch_right[nid], depth + 1)

        _collect_bottom(0, 0)

        n_bottom = len(bottom_in_order)
        x_pos = {}
        for i, nid in enumerate(bottom_in_order):
            x_pos[nid] = (i + 1) / (n_bottom + 1)

        # c) Nodos internos no truncados → bottom-up
        for nid, depth in sorted(visible, key=lambda x: -x[1]):
            if nid in x_pos:
                continue
            if ch_left[nid] != ch_right[nid]:
                left, right = ch_left[nid], ch_right[nid]
                if left in x_pos and right in x_pos:
                    x_pos[nid] = (x_pos[left] + x_pos[right]) / 2

        # Asignar y según profundidad
        y_pos = {
            nid: 1.0 - (depth + 0.5) / (max_depth + 1)
            for nid, depth in visible
        }

        # 4. Dibujar aristas (edges)
        #
        # Las aristas se dibujan de centro a centro (zorder=0) y las cajas
        # de los nodos se superponen encima (zorder=2), lo que produce la
        # apariencia estándar de árbol de decisión.  Para la rama derecha
        # con alumno seleccionado, la arista resaltada usa un trazo más
        # grueso que es visible incluso bajo el borde de la caja.
        on_path = path_nodes is not None
        for nid, depth in visible:
            if ch_left[nid] == ch_right[nid]:
                continue
            left, right = ch_left[nid], ch_right[nid]
            if left not in visible_set or right not in visible_set:
                continue
            px, py = x_pos.get(nid), y_pos.get(nid)
            if px is None or py is None:
                continue
            for cid in (left, right):
                cx, cy = x_pos.get(cid), y_pos.get(cid)
                if cx is None or cy is None:
                    continue
                edge_is_path = on_path and nid in path_nodes and cid in path_nodes
                ec = path_color if (edge_is_path and path_color) else "#CCCCCC"
                lw = 2.5 if edge_is_path else 0.8
                ax.plot(
                    [px, cx], [py, cy],
                    color=ec, linewidth=lw, solid_capstyle="round", zorder=0,
                )

        # 5. Dibujar nodos
        def _clase_predicha(nid):
            v = value[nid][0]
            return 0 if v[0] >= v[1] else 1

        for nid, depth in visible:
            x, y = x_pos.get(nid), y_pos.get(nid)
            if x is None or y is None:
                continue
            en_ruta = on_path and nid in path_nodes

            if en_ruta:
                facecolor = path_color or self._COLOR_NO_RUTA
                edgecolor = "#000000"
                linewidth = 2.5
                textcolor = "white"
            else:
                facecolor = self._COLOR_NO_RUTA
                edgecolor = self._COLOR_BORDE_NO_RUTA
                linewidth = 0.8
                textcolor = self._COLOR_TEXTO_NO_RUTA

            is_leaf = ch_left[nid] == ch_right[nid]
            truncated = (depth == max_depth) and not is_leaf

            lines = []
            if not is_leaf and not truncated:
                idx_feat = feature[nid]
                feat_name = (
                    columnas[idx_feat]
                    if idx_feat < len(columnas)
                    else f"X[{idx_feat}]"
                )
                if len(feat_name) > 28:
                    feat_name = feat_name[:25] + "..."
                lines.append(f"{feat_name} <= {threshold[nid]:.2f}")
            elif truncated:
                lines.append("(...)")

            lines.append(f"gini = {impurity[nid]:.3f}")
            lines.append(f"samples = {int(n_samples[nid])}")
            v = value[nid][0]
            lines.append(f"value = [{v[0]:.1f}, {v[1]:.1f}]")
            lines.append(
                f"class = {['Continua', 'Abandono'][_clase_predicha(nid)]}"
            )

            ax.text(
                x, y, "\n".join(lines),
                ha="center", va="center",
                fontsize=fontsize,
                color=textcolor,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                ),
                zorder=2,
                family="monospace",
            )

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

    def graficar_arbol(
        self,
        bimestre: int,
        max_depth: int = 3,
        fila_alumno: Optional[pd.DataFrame] = None,
        nivel_riesgo: Optional[str] = None,
        alumno_id: Optional[str] = None,
        guardar_ruta: Optional[str] = None,
    ) -> Optional[Figure]:
        """Genera la visualización del árbol de decisión de un bimestre,
        con la posibilidad de resaltar la ruta de decisión de un alumno.

        Args:
            bimestre: Número de bimestre (1-6).
            max_depth: Profundidad máxima a mostrar en el árbol.
            fila_alumno: DataFrame de una fila (preprocesado) con los datos
                del alumno. Si se proporciona, se resalta la ruta de decisión.
            nivel_riesgo: Nivel de riesgo del alumno (``ALTO``, ``MEDIO``,
                ``BAJO``). Determina el color de la ruta resaltada.
            alumno_id: Identificador del alumno para incluir en el título.
            guardar_ruta: Si se indica, guarda la imagen PNG en esta ruta.

        Returns:
            Figura de matplotlib o ``None`` si el modelo no existe.
        """
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

        # Extraer ruta de decisión si se proporcionó un alumno
        path_nodes = None
        path_color = None
        sufijo_titulo = ""

        if fila_alumno is not None:
            riesgo_raw = nivel_riesgo.strip() if nivel_riesgo else ""
            riesgo_up = riesgo_raw.upper()
            riesgo_valido = riesgo_up if riesgo_up in ("ALTO", "MEDIO", "BAJO") else None
            try:
                node_indicator = modelo.decision_path(fila_alumno)
                # Solo resaltar la ruta si el riesgo es válido
                if riesgo_valido:
                    path_nodes = set(node_indicator.indices)
                    path_color = self._COLOR_RIESGO[riesgo_valido]
                id_alumno = alumno_id or ""
                if riesgo_valido:
                    sufijo_titulo = (
                        f" — Alumno {id_alumno} (riesgo {riesgo_valido})"
                    )
                elif id_alumno:
                    sufijo_titulo = f" — Alumno {id_alumno}"
            except Exception as e:
                self.logger.registrar(
                    "VIZ", f"Error al extraer ruta de decisión: {e}", "ERROR"
                )
                st.warning(
                    "No se pudo extraer la ruta de decisión del alumno. "
                    "Se muestra el árbol completo."
                )

        # Calcular número de nodos hoja visibles para dimensionar la figura
        tree_ = modelo.tree_
        stack = [(0, 0)]
        leaf_like_count = 0
        while stack:
            nid, d = stack.pop()
            if d > max_depth:
                continue
            is_leaf = tree_.children_left[nid] == tree_.children_right[nid]
            if is_leaf or d == max_depth:
                leaf_like_count += 1
            if not is_leaf:
                stack.append((tree_.children_right[nid], d + 1))
                stack.append((tree_.children_left[nid], d + 1))

        # Tamaño de figura adaptativo: más ancho cuando hay muchas hojas
        ancho = max(20, (leaf_like_count + 1) * 3.5)
        alto = max(10, 4.0 * (max_depth + 1))
        fig, ax = plt.subplots(figsize=(ancho, alto))

        # Fuente adaptativa: más pequeña en árboles profundos/densos
        fontsize = max(5.5, 9.0 - max_depth * 0.5)

        self._custom_plot_tree(
            modelo,
            columnas,
            max_depth,
            ax,
            path_nodes=path_nodes,
            path_color=path_color,
            fontsize=fontsize,
        )

        titulo = f"Árbol de Decisión — Bimestre {bimestre}{sufijo_titulo}"
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
        fig.suptitle(
            titulo,
            fontsize=13,
            fontweight="bold",
            y=0.97,
        )

        if guardar_ruta:
            os.makedirs(os.path.dirname(guardar_ruta), exist_ok=True)
            fig.savefig(guardar_ruta, dpi=150, bbox_inches="tight")

        return fig

    # ======================================================================
    # Pipeline completo
    # ======================================================================
    def _ejecutar_pipeline(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de datos para la interfaz web.

        Returns:
            DataFrame con los resultados de evaluación de riesgo.
        """
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
        if EntrenadorModelos.debe_entrenar(modelo_b1):
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
        """Lanza la interfaz de Streamlit con los cuatro tabs principales."""
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
                    # os.path.basename evita path traversal si el nombre del
                    # fichero subido contiene separadores de ruta (ej. "../").
                    nombre_seguro = os.path.basename(f.name)
                    ruta = os.path.join(data_dir, nombre_seguro)
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
            "y, opcionalmente, un alumno para visualizar la ruta de decisión "
            "que el modelo ha seguido para asignarle su nivel de riesgo."
        )

        max_bimestre = self.config.machine_learning.get("max_bimestre", 6)
        bimestre = st.selectbox(
            "Seleccionar bimestre",
            range(1, max_bimestre + 1),
            format_func=lambda b: f"Bimestre {b}",
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            max_depth = st.slider(
                "Profundidad visual", min_value=1, max_value=5, value=3
            )

        # === Selector de alumno (opcional) ===
        df_riesgo = st.session_state.get("df_riesgo")
        if df_riesgo is not None and not df_riesgo.empty and "nivel_riesgo" in df_riesgo.columns:
            df_riesgo["nivel_riesgo"] = (
                df_riesgo["nivel_riesgo"].astype(str).str.strip()
            )
        alumno_seleccionado = None
        fila_alumno_preprocesada = None
        nivel_riesgo_alumno = None

        if df_riesgo is not None and not df_riesgo.empty:
            alumnos_pronostico = df_riesgo[
                (df_riesgo.get("tipo_prediccion", "") == "PRONÓSTICO")
                & (df_riesgo.get("nivel_riesgo", "").isin(["ALTO", "MEDIO", "BAJO"]))
            ]
            if not alumnos_pronostico.empty:
                opciones = [("", "— Sin alumno específico —")]
                for _, row in alumnos_pronostico.iterrows():
                    riesgo_row = row.get("nivel_riesgo", "?")
                    label = f"{row.get('n_siu', '?')} — {row.get('estudio', '?')} ({riesgo_row})"
                    opciones.append((str(row.get("n_siu", "")), label))

                selected = st.selectbox(
                    "Resaltar ruta de un alumno (opcional)",
                    options=[o[0] for o in opciones],
                    format_func=lambda x: next(
                        (o[1] for o in opciones if o[0] == x), x
                    ),
                )

                if selected:
                    alumno_seleccionado = selected
                    fila_bruta = df_riesgo[
                        (df_riesgo["n_siu"].astype(str) == str(selected))
                        & (df_riesgo.get("tipo_prediccion", "") == "PRONÓSTICO")
                    ]
                    if not fila_bruta.empty:
                        # Usar la misma lógica que el filtro para obtener el riesgo
                        riesgo_raw = fila_bruta.iloc[0]["nivel_riesgo"]
                        riesgo_str = str(riesgo_raw).strip().upper() if pd.notna(riesgo_raw) else ""
                        nivel_riesgo_alumno = riesgo_str if riesgo_str in ("ALTO", "MEDIO", "BAJO") else None
                        base_dir = os.path.dirname(os.path.dirname(__file__))

                        # Usar el bimestre DEL DROPDOWN (elección del usuario)
                        # Preprocesar los datos del alumno para ESE bimestre
                        ruta_col = os.path.join(
                            base_dir, "modelos", f"columnas_b{bimestre}.pkl"
                        )
                        if os.path.exists(ruta_col):
                            cols_b = joblib.load(ruta_col)
                            fila_alumno_preprocesada = preprocesar_fila_alumno(
                                fila_bruta.iloc[0], cols_b
                            )
                        else:
                            st.warning(
                                f"No existe el modelo para el Bimestre {bimestre}. "
                                "No se puede resaltar la ruta de decisión."
                            )

        if st.button("Generar gráfico del árbol", type="primary"):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            assets_dir = os.path.join(base_dir, "assets")

            if alumno_seleccionado:
                nombre_archivo = f"arbol_{alumno_seleccionado}_b{bimestre}.png"
            else:
                nombre_archivo = f"arbol_b{bimestre}.png"
            ruta_png = os.path.join(assets_dir, nombre_archivo)

            ruta_extraida = fila_alumno_preprocesada is not None

            with st.spinner("Generando árbol de decisión..."):
                fig = self.graficar_arbol(
                    bimestre=bimestre,
                    max_depth=max_depth,
                    fila_alumno=fila_alumno_preprocesada,
                    nivel_riesgo=nivel_riesgo_alumno,
                    alumno_id=alumno_seleccionado if ruta_extraida else None,
                    guardar_ruta=ruta_png,
                )

            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)

                if ruta_extraida and df_riesgo is not None:
                    # Obtener el riesgo real del dataframe filtrado
                    fila_info = df_riesgo[
                        (df_riesgo["n_siu"].astype(str) == str(alumno_seleccionado))
                        & (df_riesgo.get("tipo_prediccion", "") == "PRONÓSTICO")
                    ]
                    if not fila_info.empty:
                        riesgo_real = str(fila_info.iloc[0]["nivel_riesgo"]).strip().upper()
                    else:
                        riesgo_real = "NO_CALCULABLE"
                    st.success(
                        f"Ruta resaltada para alumno {alumno_seleccionado} "
                        f"(riesgo {riesgo_real}). "
                        f"Gráfico guardado en assets/{nombre_archivo}"
                    )
                else:
                    st.success(f"Gráfico guardado en assets/{nombre_archivo}")

                with open(ruta_png, "rb") as f:
                    st.download_button(
                        "Descargar imagen PNG",
                        f,
                        nombre_archivo,
                        "image/png",
                    )


if __name__ == "__main__":
    app = Visualizador()
    app.generar_interfaz_web()
