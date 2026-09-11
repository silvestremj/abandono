"""
Módulo de visualización y presentación de resultados.

Proporciona dos interfaces de salida:
- **Consola**: reporte de texto y exportación CSV.
- **Streamlit (Web)**: interfaz interactiva con tabs de carga, resultados,
  alertas y visualización de árboles de decisión (XAI).

Uso en consola::

    from src.visualizador import Visualizador
    vista = Visualizador()
    vista.mostrar_en_consola(df_final)
    vista.exportar_csv(df_final)

Uso en Streamlit::

    python -m src.visualizador
"""

import io
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
if "bimestre_corte_activo" not in st.session_state:
    # Bimestre de corte usado en la última ejecución del pipeline (None = modo
    # Automático). Es la fuente de verdad que consultan el resto de pestañas
    # para saber si están mirando una simulación con techo fijo o no.
    st.session_state.bimestre_corte_activo = None


# ======================================================================
# Validación de ficheros subidos por la interfaz web
# ======================================================================
# Columnas mínimas que debe traer cada fuente para que el pipeline pueda
# procesarla. No es el esquema completo de cada fichero (ver README): es el
# subconjunto sin el cual :class:`PreparadorDatos` fallaría o produciría una
# tabla maestra vacía, por lo que basta para descartar un fichero con el
# nombre correcto pero el contenido equivocado.
COLUMNAS_MINIMAS_DATASET: Dict[str, List[str]] = {
    "notas_bimestre": ["n_siu", "Estudio", "nota_m", "asist_m", "Comentario"],
    "actual": ["n_siu", "estudio", "estado_actual"],
    "inscripciones": ["n_siu", "estudio"],
}

# El número de bimestre no viaja en una columna propia: se extrae de la
# columna de texto ``Comentario`` (p.ej. "Bimestre: 1.0"). Si ninguna fila
# casa con este patrón, el pivotado por bimestre daría columnas vacías.
PATRON_BIMESTRE = re.compile(r"[Bb]imestre\s*:?\s*\d+")

# Subdirectorio de ``data/`` donde se archiva la versión anterior de un
# fichero antes de sustituirlo por el que sube la persona usuaria.
NOMBRE_DIR_BACKUP = "_backup"


def ruta_directorio_datos(configuracion: Optional[ConfigLoader] = None) -> str:
    """Devuelve la ruta absoluta al directorio de datos del proyecto.

    Args:
        configuracion: Instancia de :class:`ConfigLoader`. Si es ``None`` se
            usa la instancia global ``config``.

    Returns:
        Ruta absoluta a ``config.paths["data_dir"]``, la misma carpeta que lee
        :class:`IngestorDatos`.
    """
    cfg = configuracion or config
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, cfg.paths.get("data_dir", "data"))


def nombres_admitidos(configuracion: Optional[ConfigLoader] = None) -> List[str]:
    """Lista los nombres de fichero declarados en ``config.datasets``.

    Args:
        configuracion: Instancia de :class:`ConfigLoader`. Si es ``None`` se
            usa la instancia global ``config``.

    Returns:
        Lista de nombres de fichero admitidos, en el orden del YAML.
    """
    cfg = configuracion or config
    return [conf["nombre"] for conf in cfg.datasets.values()]


def resolver_clave_dataset(
    nombre_fichero: str, configuracion: Optional[ConfigLoader] = None
) -> Optional[str]:
    """Traduce el nombre de un fichero subido a su clave de ``config.datasets``.

    El pipeline solo lee los ficheros declarados en el YAML y los busca por
    nombre exacto, así que un fichero con cualquier otro nombre no se llegaría
    a procesar nunca: se ignoraría en silencio.

    Args:
        nombre_fichero: Nombre del fichero subido (se compara solo el nombre
            base, sin componentes de ruta).
        configuracion: Instancia de :class:`ConfigLoader`. Si es ``None`` se
            usa la instancia global ``config``.

    Returns:
        La clave del dataset (``"inscripciones"``, ``"notas_bimestre"`` o
        ``"actual"``) o ``None`` si el nombre no está declarado.
    """
    cfg = configuracion or config
    base = os.path.basename(nombre_fichero)
    for clave, conf in cfg.datasets.items():
        if base == conf["nombre"]:
            return clave
    return None


def validar_estructura_dataset(
    clave: str, ruta: str, configuracion: Optional[ConfigLoader] = None
) -> Optional[str]:
    """Comprueba que un fichero tiene la estructura esperada para su clave.

    Lee el fichero con los mismos parámetros que usará luego
    :meth:`IngestorDatos.leer_datos` (tipo, separador, encoding y detección de
    BOM UTF-8), mediante :meth:`IngestorDatos.leer_archivo`, y verifica que
    contiene las columnas de :data:`COLUMNAS_MINIMAS_DATASET`. Para
    ``notas_bimestre`` exige además que ``Comentario`` traiga el número de
    bimestre en el formato que espera la preparación.

    Args:
        clave: Clave del dataset en ``config.datasets``.
        ruta: Ruta al fichero a validar.
        configuracion: Instancia de :class:`ConfigLoader`. Si es ``None`` se
            usa la instancia global ``config``.

    Returns:
        ``None`` si el fichero es válido; en caso contrario, el motivo
        concreto del rechazo, listo para mostrar a la persona usuaria.
    """
    cfg = configuracion or config
    conf = cfg.datasets.get(clave)
    if conf is None:
        return f"la clave '{clave}' no está declarada en config.yaml"

    try:
        df = IngestorDatos.leer_archivo(ruta, conf)
    except Exception as exc:  # noqa: BLE001 - el motivo se muestra tal cual
        return (
            f"no se ha podido leer el fichero como {conf['tipo']} "
            f"(separador {conf.get('sep', 'n/a')!r}, "
            f"encoding {conf.get('encoding', 'n/a')!r}): {exc}"
        )

    faltantes = [
        c for c in COLUMNAS_MINIMAS_DATASET.get(clave, []) if c not in df.columns
    ]
    if faltantes:
        return (
            "faltan columnas obligatorias: "
            + ", ".join(faltantes)
            + ". Columnas encontradas: "
            + (", ".join(str(c) for c in df.columns) or "ninguna")
        )

    if clave == "notas_bimestre":
        comentarios = df["Comentario"].dropna().astype(str)
        if not comentarios.str.contains(PATRON_BIMESTRE, regex=True).any():
            return (
                "ninguna fila de la columna 'Comentario' indica el bimestre "
                "con el formato esperado (por ejemplo 'Bimestre: 1')"
            )

    return None


class Visualizador:
    """Presentación de resultados en consola y en la interfaz Streamlit.

    Attributes:
        config: Instancia de :class:`ConfigLoader` con la configuración del proyecto.
        logger: Instancia de :class:`GestorLogs` para trazabilidad.
    """

    def __init__(
        self,
        configuracion: Optional[ConfigLoader] = None,
        gestor_logs: Optional[GestorLogs] = None,
    ) -> None:
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
    def mostrar_en_consola(self, df_riesgo: pd.DataFrame) -> None:
        """Muestra un reporte de texto en la consola con el conteo por nivel de riesgo.

        Args:
            df_riesgo: DataFrame con la columna ``nivel_riesgo``.
        """
        if df_riesgo.empty:
            print("No hay datos para mostrar.")
            return
        print("\n" + "=" * 60)
        print("  REPORTE DE RIESGO DE ABANDONO ACADÉMICO")
        print("=" * 60)
        conteo = df_riesgo["nivel_riesgo"].value_counts()
        for nivel in ["ALTO", "MEDIO", "BAJO", "SIN_DATOS_SUFICIENTES", "NO_CALCULABLE"]:
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

        return fig

    # ======================================================================
    # Pipeline completo
    # ======================================================================
    def _ejecutar_pipeline(self, bimestre_corte: Optional[int] = None) -> pd.DataFrame:
        """Ejecuta el pipeline completo de datos para la interfaz web.

        Args:
            bimestre_corte: Techo temporal a pasar a
                :meth:`EvaluadorRiesgo.ejecutar_evaluacion`. ``None`` (modo
                Automático) evalúa a cada alumno en su propio bimestre real.

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

        evaluador = EvaluadorRiesgo(gestor_logs=log)
        df_riesgo = evaluador.ejecutar_evaluacion(df_master, bimestre_corte=bimestre_corte)
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

    # ------------------------------------------------------------------
    # Carga de ficheros: estado actual, respaldo y validación
    # ------------------------------------------------------------------
    def _lineas_estado_datasets(self) -> List[str]:
        """Describe, para cada fichero declarado, si está en ``data/`` y desde cuándo.

        Returns:
            Lista de líneas Markdown (una por dataset declarado) con el nombre
            admitido, su presencia en el directorio de datos y la fecha de
            modificación del fichero presente.
        """
        data_dir = ruta_directorio_datos(self.config)
        lineas: List[str] = []

        for conf in self.config.datasets.values():
            nombre = conf["nombre"]
            ruta = os.path.join(data_dir, nombre)
            if os.path.exists(ruta):
                fecha = datetime.fromtimestamp(os.path.getmtime(ruta)).strftime(
                    "%d/%m/%Y %H:%M"
                )
                estado = f"presente en `data/`, modificado el {fecha}"
            else:
                estado = "**no está** en `data/`"
            lineas.append(f"- `{nombre}` — {estado}")

        return lineas

    def _respaldar_fichero(self, ruta: str, data_dir: str) -> Optional[str]:
        """Archiva una copia del fichero actual antes de sustituirlo.

        Args:
            ruta: Ruta al fichero que va a ser sobrescrito.
            data_dir: Directorio de datos donde vive el subdirectorio de
                respaldos.

        Returns:
            Ruta de la copia de seguridad, o ``None`` si el fichero no existía
            (nada que respaldar).
        """
        if not os.path.exists(ruta):
            return None

        dir_backup = os.path.join(data_dir, NOMBRE_DIR_BACKUP)
        os.makedirs(dir_backup, exist_ok=True)

        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = os.path.join(dir_backup, f"{os.path.basename(ruta)}.{marca}")
        shutil.copy2(ruta, destino)

        self.logger.registrar(
            "VIZ-UI",
            f"Respaldo de '{os.path.basename(ruta)}' en "
            f"'{NOMBRE_DIR_BACKUP}/{os.path.basename(destino)}'",
        )
        return destino

    def _guardar_archivos_subidos(self, archivos, data_dir: str) -> List[str]:
        """Valida y guarda en ``data/`` los ficheros subidos por la persona usuaria.

        Un fichero solo sustituye al original si (1) su nombre es uno de los
        declarados en ``config.datasets`` y (2) su contenido supera la
        validación de estructura. El original se respalda antes de ser
        sustituido. Los rechazos se muestran con ``st.error`` y se registran en
        el log; los demás ficheros del lote siguen su curso.

        Args:
            archivos: Ficheros devueltos por ``st.file_uploader``.
            data_dir: Directorio de datos donde escribir.

        Returns:
            Lista con los nombres de los ficheros efectivamente sustituidos.
        """
        admitidos = nombres_admitidos(self.config)
        rechazados_por_nombre: List[str] = []
        aceptados: List[Tuple[str, str, Any]] = []

        # 1. Filtro por nombre: lo que no está declarado no llega ni a escribirse.
        for f in archivos:
            # os.path.basename evita path traversal si el nombre del
            # fichero subido contiene separadores de ruta (ej. "../").
            nombre_seguro = os.path.basename(f.name)
            clave = resolver_clave_dataset(nombre_seguro, self.config)
            if clave is None:
                rechazados_por_nombre.append(nombre_seguro)
            else:
                aceptados.append((clave, nombre_seguro, f))

        if rechazados_por_nombre:
            lista_recibidos = ", ".join(f"'{n}'" for n in rechazados_por_nombre)
            lista_admitidos = ", ".join(f"'{n}'" for n in admitidos)
            st.error(
                f"No se han guardado {len(rechazados_por_nombre)} fichero(s) porque "
                f"su nombre no está declarado en config.yaml: {lista_recibidos}. "
                f"Nombres admitidos: {lista_admitidos}."
            )
            for nombre in rechazados_por_nombre:
                self.logger.registrar(
                    "VIZ-UI",
                    f"Fichero '{nombre}' rechazado: nombre no declarado en config.yaml",
                    "ERROR",
                )

        # 2. Validación de estructura sobre una copia temporal, para no tocar
        #    el original hasta saber que el sustituto sirve.
        sustituidos: List[str] = []
        for clave, nombre, f in aceptados:
            with tempfile.TemporaryDirectory() as dir_tmp:
                ruta_tmp = os.path.join(dir_tmp, nombre)
                with open(ruta_tmp, "wb") as fp:
                    fp.write(f.getbuffer())

                motivo = validar_estructura_dataset(clave, ruta_tmp, self.config)
                if motivo is not None:
                    st.error(f"'{nombre}' no sustituye al fichero actual: {motivo}")
                    self.logger.registrar(
                        "VIZ-UI", f"Fichero '{nombre}' rechazado: {motivo}", "ERROR"
                    )
                    continue

                destino = os.path.join(data_dir, nombre)
                self._respaldar_fichero(destino, data_dir)
                shutil.copyfile(ruta_tmp, destino)

            sustituidos.append(nombre)
            self.logger.registrar("VIZ-UI", f"Fichero '{nombre}' actualizado en data/")

        if sustituidos:
            st.success(
                "Ficheros actualizados en `data/`: " + ", ".join(sustituidos) + "."
            )

        return sustituidos

    def _render_carga(self):
        st.header("1. Cargar archivos de datos")
        st.write(
            "Solo se admiten **versiones nuevas de los tres ficheros declarados "
            "en `config.yaml`**: el pipeline los busca por su nombre exacto, así "
            "que cualquier otro nombre se rechaza en vez de guardarse sin usarse. "
            "Si no subes nada, los cálculos se ejecutan sobre lo que ya hay en "
            "`data/`."
        )
        st.markdown("\n".join(self._lineas_estado_datasets()))

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
        st.header("2. Modo de evaluación")

        max_bimestre = self.config.machine_learning.get("max_bimestre", 6)

        modo_evaluacion = st.radio(
            "¿Cómo se debe evaluar a cada alumno?",
            options=[
                "Automático (recomendado)",
                "Fijar bimestre de corte (simulación)",
            ],
            help=(
                "Automático: cada alumno se evalúa en su propio bimestre real "
                "(el último con notas registradas) — es la 'foto de hoy', "
                "cada alumno en su propio punto. Fijar bimestre de corte: "
                "simula la evaluación como si hoy fuera un bimestre concreto, "
                "ignorando datos posteriores a ese hito para todos los "
                "alumnos (un alumno sin datos reales hasta ese hito se sigue "
                "evaluando en su bimestre real, para no simular un suspenso "
                "falso)."
            ),
        )

        bimestre_corte_seleccionado = None
        if modo_evaluacion == "Fijar bimestre de corte (simulación)":
            bimestre_corte_seleccionado = st.selectbox(
                "Bimestre de corte",
                range(1, max_bimestre + 1),
                format_func=lambda b: f"Bimestre {b}",
            )

        st.markdown("---")
        st.header("3. Ejecutar cálculos")

        if st.button("Ejecutar cálculos", type="primary"):
            data_dir = ruta_directorio_datos(self.config)
            os.makedirs(data_dir, exist_ok=True)

            if archivos:
                self._guardar_archivos_subidos(archivos, data_dir)

            with st.spinner("Procesando pipeline de datos..."):
                try:
                    df_riesgo = self._ejecutar_pipeline(
                        bimestre_corte=bimestre_corte_seleccionado
                    )
                    st.session_state.df_riesgo = df_riesgo
                    st.session_state.pipeline_ok = True
                    st.session_state.bimestre_corte_activo = bimestre_corte_seleccionado
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
                    p_sin_datos = len(
                        df_pronostico[df_pronostico["nivel_riesgo"] == "SIN_DATOS_SUFICIENTES"]
                    )

                    st.subheader("Distribución de riesgo pronosticado")
                    c_a, c_m, c_b, c_s = st.columns(4)
                    c_a.metric("Riesgo alto", p_alto)
                    c_m.metric("Riesgo medio", p_medio)
                    c_b.metric("Riesgo bajo", p_bajo)
                    c_s.metric("Sin datos suficientes", p_sin_datos)

                    st.caption(
                        "Estos alumnos **no tienen etiqueta conocida**. "
                        "El modelo asigna su nivel de riesgo basándose en su "
                        "rendimiento académico actual. Los marcados como "
                        "'Sin datos suficientes' todavía no tienen notas ni "
                        "asistencia registradas para su bimestre actual, así "
                        "que no se ejecuta el modelo sobre ellos (evita "
                        "confundir 'sin dato' con 'rindió cero')."
                    )

                # Modo "Fijar bimestre de corte": avisar si algún alumno no
                # tenía aún datos reales hasta el corte elegido y por tanto
                # se evaluó con su bimestre real (inferior al corte).
                bimestre_corte_activo = st.session_state.get("bimestre_corte_activo")
                if bimestre_corte_activo is not None and "bimestre_evaluado" in df.columns:
                    mask_desajuste = (
                        df_pronostico["bimestre_evaluado"] < bimestre_corte_activo
                    )
                    n_desajuste = int(mask_desajuste.fillna(False).sum())
                    if n_desajuste > 0:
                        st.warning(
                            f"Modo simulación (corte fijado en Bimestre "
                            f"{bimestre_corte_activo}): {n_desajuste} alumno(s) "
                            "todavía no tenían datos reales hasta ese hito y se "
                            "evaluaron con su bimestre real (inferior al corte), "
                            "para evitar simular un suspenso falso. Revisa la "
                            "columna 'bimestre_evaluado' en el listado."
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
            "bimestre_evaluado",
            "justificacion_riesgo",
        ]
        cols_mostrar = [c for c in cols_clave if c in df.columns]
        st.dataframe(df[cols_mostrar], width="stretch", hide_index=True)

    def _render_seccion_riesgo(
        self,
        nivel: str,
        alumnos: pd.DataFrame,
        titulo: str,
        fn_vacio,
        fn_alerta,
        mensaje_vacio: str,
        mensaje_alerta: str,
    ) -> None:
        """Renderiza el bloque de alertas de un nivel de riesgo (ALTO/MEDIO/BAJO).

        Antes solo se listaban los alumnos en ALTO: con muy pocos casos en
        ese nivel (p.ej. 2), la pestaña quedaba casi vacía pese a haber
        decenas de alumnos en MEDIO/BAJO cuya justificación también es útil
        revisar. Se factoriza en un único método porque las tres secciones
        comparten exactamente la misma estructura (mensaje de cabecera,
        expander por alumno con su justificación, descarga CSV) y solo
        cambia el tono visual y el texto.

        Args:
            nivel: Nombre del nivel de riesgo (``"ALTO"``, ``"MEDIO"`` o
                ``"BAJO"``), usado para la clave del widget y el nombre del CSV.
            alumnos: Subconjunto de alumnos en ese nivel de riesgo.
            titulo: Encabezado de la sección.
            fn_vacio: Función de Streamlit a usar cuando no hay alumnos
                (p.ej. ``st.success``, ``st.info``).
            fn_alerta: Función de Streamlit a usar cuando sí hay alumnos
                (p.ej. ``st.error``, ``st.warning``, ``st.info``).
            mensaje_vacio: Texto a mostrar cuando ``alumnos`` está vacío.
            mensaje_alerta: Texto a mostrar cuando ``alumnos`` no está vacío.
        """
        st.subheader(titulo)
        if alumnos.empty:
            fn_vacio(mensaje_vacio)
            return

        fn_alerta(mensaje_alerta)
        for _, alumno in alumnos.iterrows():
            with st.expander(
                f"{alumno.get('n_siu', '?')} - {alumno.get('estudio', '?')}"
            ):
                st.write(
                    f"**Justificación:** {alumno.get('justificacion_riesgo', 'N/A')}"
                )

        csv = alumnos.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        st.download_button(
            f"Descargar riesgo {nivel.lower()} (CSV)",
            csv,
            f"riesgo_{nivel.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            key=f"descarga_csv_riesgo_{nivel.lower()}",
        )

    def _render_alertas(self):
        if not st.session_state.pipeline_ok or st.session_state.df_riesgo is None:
            st.info("Ejecuta los cálculos para generar alertas.")
            return

        df = st.session_state.df_riesgo
        alumnos_alto = df[df["nivel_riesgo"] == "ALTO"]
        alumnos_medio = df[df["nivel_riesgo"] == "MEDIO"]
        alumnos_bajo = df[df["nivel_riesgo"] == "BAJO"]
        alumnos_sin_datos = df[df["nivel_riesgo"] == "SIN_DATOS_SUFICIENTES"]

        self._render_seccion_riesgo(
            "ALTO",
            alumnos_alto,
            "Riesgo alto",
            st.success,
            st.error,
            "No se detectaron alumnos en riesgo alto de abandono.",
            f"Atención: {len(alumnos_alto)} alumno(s) en RIESGO ALTO de abandono.",
        )

        st.markdown("---")
        self._render_seccion_riesgo(
            "MEDIO",
            alumnos_medio,
            "Riesgo medio",
            st.info,
            st.warning,
            "No se detectaron alumnos en riesgo medio de abandono.",
            f"{len(alumnos_medio)} alumno(s) en riesgo MEDIO de abandono.",
        )

        st.markdown("---")
        self._render_seccion_riesgo(
            "BAJO",
            alumnos_bajo,
            "Riesgo bajo",
            st.info,
            st.info,
            "No se detectaron alumnos en riesgo bajo de abandono.",
            f"{len(alumnos_bajo)} alumno(s) en riesgo BAJO de abandono.",
        )

        # Bloque aparte, no mezclado con las alertas de riesgo: estos alumnos
        # no tienen ningún dato real todavía (ver EvaluadorRiesgo.
        # _sin_datos_reales_bimestre), así que no representan una alerta de
        # abandono sino una ausencia de señal a la que hacer seguimiento.
        if not alumnos_sin_datos.empty:
            st.markdown("---")
            st.warning(
                f"{len(alumnos_sin_datos)} alumno(s) sin datos suficientes "
                "para evaluar: no tienen notas ni asistencia registradas "
                "hasta su bimestre actual. No es una alerta de riesgo, es "
                "una ausencia de información."
            )
            with st.expander("Alumnos sin datos suficientes para evaluar"):
                cols_sin_datos = [
                    c
                    for c in [
                        "n_siu",
                        "estudio",
                        "bimestre_evaluado",
                        "justificacion_riesgo",
                    ]
                    if c in alumnos_sin_datos.columns
                ]
                st.dataframe(
                    alumnos_sin_datos[cols_sin_datos],
                    width="stretch",
                    hide_index=True,
                )

    def _render_arbol(self):
        st.header("Árbol de decisión por bimestre")
        st.write(
            "Cada bimestre tiene su propio árbol entrenado con los datos "
            "disponibles hasta ese hito temporal. El árbol se actualiza al "
            "instante según el bimestre y el alumno elegidos: solo se "
            "muestra **un** árbol en pantalla, y su color y la información "
            "que lo acompaña corresponden siempre a lo que se está viendo "
            "— nunca a un bimestre distinto."
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
        fila_bruta = None
        bimestre_evaluado_alumno = None
        riesgo_oficial = None

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
                        riesgo_oficial = riesgo_str if riesgo_str in ("ALTO", "MEDIO", "BAJO") else None

                        bimestre_raw = fila_bruta.iloc[0].get("bimestre_evaluado")
                        bimestre_evaluado_alumno = (
                            int(bimestre_raw) if pd.notna(bimestre_raw) else None
                        )

        # ======================================================================
        # TASK-APP-03 (corrección de coherencia): se renderiza SIEMPRE un único
        # árbol, en vivo (sin botón), para que el mensaje mostrado y el árbol
        # en pantalla nunca puedan desincronizarse. Antes existían dos árboles
        # (uno "oficial" automático y otro exploratorio tras pulsar un botón)
        # que podían quedar visualmente contradictorios entre sí; ahora solo
        # hay un punto de renderizado. El panel de justificación/métricas solo
        # se muestra cuando ESE árbol es, además, la evaluación oficial del
        # alumno (mismo bimestre), para no describir un hito que no es el que
        # se ve en pantalla.
        # ======================================================================
        base_dir = os.path.dirname(os.path.dirname(__file__))
        fila_alumno_preprocesada = None
        nivel_riesgo_color = None
        es_vista_oficial = False

        if alumno_seleccionado and fila_bruta is not None and not fila_bruta.empty:
            ruta_col = os.path.join(base_dir, "modelos", f"columnas_b{bimestre}.pkl")
            if os.path.exists(ruta_col):
                cols_b = joblib.load(ruta_col)
                fila_alumno_preprocesada = preprocesar_fila_alumno(
                    fila_bruta.iloc[0], cols_b
                )
                if bimestre_evaluado_alumno is not None and bimestre != bimestre_evaluado_alumno:
                    st.warning(
                        f"Estás viendo el Bimestre {bimestre} (vista "
                        "exploratoria, ruta sin colorear); este alumno fue "
                        f"evaluado oficialmente en el Bimestre "
                        f"{bimestre_evaluado_alumno}. Cambia el selector de "
                        "arriba a ese bimestre para ver su ruta de riesgo "
                        "real junto con su justificación y métricas."
                    )
                else:
                    nivel_riesgo_color = riesgo_oficial
                    es_vista_oficial = bimestre_evaluado_alumno is not None
            else:
                st.warning(
                    f"No existe el modelo para el Bimestre {bimestre}. "
                    "No se puede resaltar la ruta de decisión."
                )

        alumno_id_titulo = (
            alumno_seleccionado if fila_alumno_preprocesada is not None else None
        )

        with st.spinner("Generando árbol de decisión..."):
            fig = self.graficar_arbol(
                bimestre=bimestre,
                max_depth=max_depth,
                fila_alumno=fila_alumno_preprocesada,
                nivel_riesgo=nivel_riesgo_color,
                alumno_id=alumno_id_titulo,
            )

        if fig is None:
            return

        st.pyplot(fig)

        # Panel de justificación + bimestre + métricas: solo si el árbol que
        # se acaba de mostrar ES la evaluación oficial del alumno.
        if es_vista_oficial and fila_bruta is not None:
            fila_oficial = fila_bruta.iloc[0]
            justificacion = fila_oficial.get("justificacion_riesgo", "N/A")
            st.markdown(f"**Justificación:** {justificacion}")

            bimestre_corte_activo = st.session_state.get("bimestre_corte_activo")
            if bimestre_corte_activo is not None:
                if bimestre_evaluado_alumno == bimestre_corte_activo:
                    st.caption(
                        f"Bimestre evaluado: **{bimestre_evaluado_alumno}** "
                        "(coincide con el corte de simulación fijado)."
                    )
                else:
                    st.caption(
                        f"Bimestre evaluado: **{bimestre_evaluado_alumno}** — el "
                        f"corte de simulación fijado era el Bimestre "
                        f"{bimestre_corte_activo}; este alumno todavía no tenía "
                        "datos reales hasta ese hito."
                    )
            else:
                st.caption(
                    f"Bimestre evaluado: **{bimestre_evaluado_alumno}** "
                    "(modo automático)."
                )

            df_metricas = st.session_state.get("df_metricas")
            if df_metricas is not None and not df_metricas.empty:
                hito_label = f"Bimestre {bimestre_evaluado_alumno}"
                fila_metricas = df_metricas[df_metricas["Hito"] == hito_label]
                if not fila_metricas.empty:
                    fm = fila_metricas.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{fm['Accuracy']:.2f}")
                    c2.metric("Recall (Abandono)", f"{fm['Recall (Abandono)']:.2f}")
                    c3.metric("Variable clave", str(fm.get("Variable Clave", "—")))

        # Descarga: un único clic entrega el PNG al navegador (carpeta de
        # descargas del usuario), sin escribir nada en el servidor — reutiliza
        # la MISMA figura que está en pantalla, nunca regenera un árbol
        # distinto para el archivo descargado.
        nombre_archivo = (
            f"arbol_{alumno_seleccionado}_b{bimestre}.png"
            if alumno_seleccionado
            else f"arbol_b{bimestre}.png"
        )
        buffer_png = io.BytesIO()
        fig.savefig(buffer_png, format="png", dpi=150, bbox_inches="tight")
        buffer_png.seek(0)
        st.download_button(
            "Descargar imagen PNG",
            data=buffer_png,
            file_name=nombre_archivo,
            mime="image/png",
        )

        plt.close(fig)


if __name__ == "__main__":
    # Se construye aquí el gestor de logs con base de datos (igual que en
    # main.py) para que todo lo que registre self.logger durante la ejecución
    # con Streamlit quede también persistido en la tabla logs_ejecucion y no
    # solo en consola y fichero.
    db_app = GestorBaseDatos()
    db_app.inicializar_tablas_fijas()
    log_app = GestorLogs(gestor_db=db_app)

    app = Visualizador(gestor_logs=log_app)
    app.generar_interfaz_web()
