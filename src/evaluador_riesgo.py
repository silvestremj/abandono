"""
Módulo de evaluación de riesgo mediante inferencia ML.

Carga los modelos bimestrales previamente entrenados, detecta el hito
temporal de cada alumno, alinea las columnas al esquema de entrenamiento
y calcula la probabilidad de abandono con explicabilidad local (XAI).

Uso::

    from src.evaluador_riesgo import EvaluadorRiesgo
    evaluador = EvaluadorRiesgo()
    df_final = evaluador.ejecutar_evaluacion(df_master)
"""

import os
import re
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import config
from src.gestor_logs import GestorLogs
from src.preparador_datos import (
    COLUMNAS_EXCLUIDAS_ML,
    PreparadorDatos,
    excluir_columnas_ml,
)

# ======================================================================
# Columnas que se excluyen del dataset antes de la inferencia.
#
# Reutiliza la misma lista de exclusión que el entrenamiento
# (:data:`src.preparador_datos.COLUMNAS_EXCLUIDAS_ML`) para que ambas fases
# del pipeline nunca puedan desincronizarse, y añade las columnas que solo
# existen en tiempo de inferencia (el target de entrenamiento y las columnas
# de resultado que genera :meth:`EvaluadorRiesgo.ejecutar_evaluacion`).
# ======================================================================
COLS_EXCLUIR = COLUMNAS_EXCLUIDAS_ML + [
    "target_ml",
    # Columnas añadidas por ejecutar_evaluacion — nunca deben entrar al modelo
    "tipo_prediccion",
    "nivel_riesgo",
    "probabilidad_abandono",
    "justificacion_riesgo",
    "bimestre_evaluado",
]

# ======================================================================
# Etiquetas legibles para traducir a lenguaje natural las variables
# categóricas codificadas con one-hot (``pd.get_dummies`` en
# :mod:`src.preparador_datos`, columnas ``<variable_base>_<categoria>``).
#
# Es una tabla puramente de presentación (cómo se le nombra una variable a
# un usuario no técnico en la justificación XAI), no una regla de negocio,
# por lo que se mantiene como constante de código junto a quien la consume
# — a diferencia de ``reglas_riesgo`` en ``config.yaml``/``ConfigLoader``,
# que sí parametriza decisiones de negocio (umbrales y palabras clave de
# clasificación de riesgo).
# ======================================================================
ETIQUETAS_VARIABLES_CATEGORICAS: Dict[str, str] = {
    "pais": "su país de residencia",
    "provincia": "su provincia de residencia",
    "max_grado": "su nivel máximo de estudios",
    "tipo_ocupacion": "su ocupación actual",
    "estudio": "el programa académico en el que está inscrito",
}


def preprocesar_fila_alumno(
    fila: pd.Series,
    columnas_entrenamiento: list,
    gestor_logs: Optional[GestorLogs] = None,
) -> Optional[pd.DataFrame]:
    """Preprocesa una fila individual de alumno para obtener el vector de
    características listo para inferencia con el modelo.

    Aplica el mismo pipeline que :meth:`EvaluadorRiesgo.ejecutar_evaluacion`:
    exclusión de columnas no relevantes, codificación *one-hot* de categóricas
    y alineación exacta al esquema de columnas usado durante el entrenamiento.

    Args:
        fila: Serie de pandas con los datos brutos del alumno (una fila del
            DataFrame **master**).
        columnas_entrenamiento: Lista de nombres de columnas que espera el
            modelo entrenado (cargadas desde ``columnas_b*.pkl``).
        gestor_logs: Instancia opcional de :class:`GestorLogs`. Si se recibe,
            un fallo de preprocesado deja constancia del tipo y el mensaje de
            la excepción original en lugar de descartarse en silencio. Va al
            final y con valor por defecto para no romper a los llamadores que
            no disponen de un logger.

    Returns:
        DataFrame de una sola fila listo para ``model.predict_proba`` o
        ``model.decision_path``, o ``None`` si ocurre algún error.
    """
    try:
        df = pd.DataFrame([fila])

        df_ml = excluir_columnas_ml(df, COLS_EXCLUIR)

        # drop_first=False (a diferencia de preparar_dataset_ml, que sí usa
        # drop_first=True sobre el dataset completo de entrenamiento, donde es
        # correcto). Aquí se codifica una única fila, que por definición solo
        # puede tener UN valor por variable categórica: con drop_first=True,
        # pandas siempre descarta esa única categoría presente y genera 0
        # columnas dummy para esa variable, así que el relleno posterior de
        # "columnas_faltantes" con 0 termina poniendo a 0 la categoría real
        # del alumno igual que las demás -- indistinguible de no tenerla. Con
        # drop_first=False si la categoría del alumno coincide con una
        # columna de columnas_entrenamiento queda correctamente en 1; si es
        # la categoría de referencia que el entrenamiento descartó, queda en
        # 0 en todas las dummies de esa variable (comportamiento correcto,
        # igual que en entrenamiento). El alineado final a
        # columnas_entrenamiento ya descarta cualquier dummy sobrante.
        cols_categoricas = df_ml.select_dtypes(include=["object"]).columns.tolist()
        df_ml = pd.get_dummies(
            df_ml, columns=cols_categoricas, dummy_na=False, drop_first=False
        )

        columnas_faltantes = [
            col for col in columnas_entrenamiento if col not in df_ml.columns
        ]
        if columnas_faltantes:
            df_faltantes = pd.DataFrame(
                0, index=df_ml.index, columns=columnas_faltantes
            )
            df_ml = pd.concat([df_ml, df_faltantes], axis=1)

        return df_ml[columnas_entrenamiento]

    except Exception as error:
        if gestor_logs is not None:
            identificador_fila = getattr(fila, "name", None)
            referencia_fila = (
                f" (fila {identificador_fila})"
                if identificador_fila is not None
                else ""
            )
            gestor_logs.registrar(
                "EVALUADOR",
                f"Error al preprocesar la fila del alumno{referencia_fila}: "
                f"{type(error).__name__}: {error}",
                "ERROR",
            )
        return None


class EvaluadorRiesgo:
    """Evaluación de riesgo por hitos bimestrales con explicabilidad.

    Attributes:
        reglas: Diccionario con las reglas de negocio para clasificación
            de estados (bajo, medio, alto riesgo).
        max_bimestre: Último bimestre configurado (``config.yaml``,
            ``machine_learning.max_bimestre``).
        umbral_medio: Probabilidad mínima para clasificar la alerta como MEDIO
            (``config.yaml``, ``reglas_riesgo.umbral_medio``).
        umbral_alto: Probabilidad mínima para clasificar la alerta como ALTO
            (``config.yaml``, ``reglas_riesgo.umbral_alto``).
        logger: Instancia de :class:`GestorLogs` para trazabilidad.
        model_dir: Ruta al directorio de modelos guardados.
        modelos_cache: Caché de modelos cargados por bimestre.
        columnas_cache: Caché de listas de columnas por bimestre.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        gestor_logs: Optional[GestorLogs] = None,
    ) -> None:
        """Inicializa el evaluador cargando reglas y configurando rutas.

        Args:
            model_dir: Directorio de modelos. Si es ``None``, se usa
                ``modelos/`` en la raíz del proyecto.
            gestor_logs: Instancia de :class:`GestorLogs`. Si es ``None``
                se crea una por defecto.
        """
        self.reglas: Dict[str, Any] = config.reglas_riesgo
        self.max_bimestre: int = config.machine_learning.get("max_bimestre", 6)
        self.umbral_medio: float = float(self.reglas.get("umbral_medio", 0.40))
        self.umbral_alto: float = float(self.reglas.get("umbral_alto", 0.70))
        self.logger = gestor_logs or GestorLogs()

        if model_dir is None:
            # Crea una carpeta 'modelos' en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.model_dir = os.path.join(base_dir, "modelos")
        else:
            self.model_dir = model_dir

        # Los diccionarios almacenarán en caché los modelos a medida que se necesiten
        self.modelos_cache: Dict[int, Any] = {}
        self.columnas_cache: Dict[int, list] = {}

    def _obtener_modelo_bimestre(self, bimestre: int) -> tuple:
        """Carga bajo demanda (Lazy Loading) el modelo y las columnas de un bimestre.

        Args:
            bimestre: Número de bimestre (1-6).

        Returns:
            Tupla ``(modelo, columnas)`` o ``(None, None)`` si no existe.
        """
        if bimestre not in self.modelos_cache:
            ruta_modelo = os.path.join(self.model_dir, f"arbol_b{bimestre}.pkl")
            ruta_columnas = os.path.join(self.model_dir, f"columnas_b{bimestre}.pkl")

            if os.path.exists(ruta_modelo) and os.path.exists(ruta_columnas):
                self.modelos_cache[bimestre] = joblib.load(ruta_modelo)
                self.columnas_cache[bimestre] = joblib.load(ruta_columnas)
            else:
                return None, None

        return self.modelos_cache[bimestre], self.columnas_cache[bimestre]

    def _detectar_bimestre_alumno(self, fila_alumno: pd.Series) -> int:
        """Determina el hito temporal actual del alumno buscando desde el último
        bimestre configurado hacia B1.

        Un bimestre cuenta como "el actual" si tiene nota o asistencia
        real (mayor que 0): un alumno puede asistir y sacar un cero real en
        su bimestre más reciente, y ese cero sigue siendo un dato real de
        ese bimestre, no ausencia de registro. Comprobar solo la nota
        detectaría a ese alumno en un bimestre anterior, ignorando su
        registro real más reciente. La ambigüedad opuesta (nota y
        asistencia en 0.0 a la vez, sin ningún dato real) la resuelve
        :meth:`_sin_datos_reales_bimestre` para el bimestre ya detectado
        aquí.

        Args:
            fila_alumno: Fila del DataFrame con los datos del alumno.

        Returns:
            Número de bimestre detectado (1-``self.max_bimestre``). Por defecto 1.
        """
        # Se busca de atrás hacia adelante (del último bimestre al 1) cuál es
        # el primero con datos válidos.
        for b in range(self.max_bimestre, 0, -1):
            valor_nota = fila_alumno.get(f"nota_b{b}", np.nan)
            valor_asist = fila_alumno.get(f"asist_b{b}", np.nan)
            tiene_nota = pd.notna(valor_nota) and valor_nota > 0
            tiene_asist = pd.notna(valor_asist) and valor_asist > 0
            if tiene_nota or tiene_asist:
                return b
        return 1  # Por defecto, si no hay notas registradas aún, se evalúa con el modelo del Bimestre 1

    def _sin_datos_reales_bimestre(self, fila_alumno: pd.Series, bimestre: int) -> bool:
        """Determina si el alumno no tiene ningún registro real hasta el
        bimestre detectado (``nota_bN`` y ``asist_bN`` valen 0.0 a la vez).

        :meth:`src.preparador_datos.PreparadorDatos.ejecutar_preparacion`
        rellena con ``0.0`` las columnas de bimestres sin registro real, así
        que ``nota_bN == 0.0`` puede significar tanto "sacó un cero" (un
        predictor de abandono genuino) como "todavía no hay dato" (el fichero
        de notas de origen no cubre todavía el bimestre en curso para cerca
        del 90 % del alumnado activo, en el caso de B1). Se verificó
        empíricamente que, de los alumnos que sí tienen algún dato parcial del
        bimestre, siempre hay al menos uno de los dos valores (nota o
        asistencia) mayor que 0 — el doble-cero conjunto es la señal fiable de
        ausencia de registro, sin falsos positivos detectados sobre los datos
        reales del proyecto.

        Args:
            fila_alumno: Fila del DataFrame con los datos del alumno.
            bimestre: Bimestre detectado para ese alumno.

        Returns:
            ``True`` si ``nota_bN`` y ``asist_bN`` existen y valen ambos
            0.0 para el bimestre indicado.
        """
        col_nota = f"nota_b{bimestre}"
        col_asist = f"asist_b{bimestre}"
        valor_nota = fila_alumno.get(col_nota, np.nan)
        valor_asist = fila_alumno.get(col_asist, np.nan)
        return (
            pd.notna(valor_nota)
            and pd.notna(valor_asist)
            and float(valor_nota) == 0.0
            and float(valor_asist) == 0.0
        )

    def _traducir_regla(
        self, nombre_caracteristica: str, umbral: float, valor: float
    ) -> str:
        """Traduce una única regla técnica del árbol a una cláusula en
        lenguaje natural.

        Distingue variables numéricas continuas identificables por nombre
        (``nota_bN``, ``asist_bN``, ``edad``, ``postgrado``) de variables
        categóricas codificadas con one-hot (``<variable_base>_<categoria>``),
        ya que unas y otras requieren una redacción distinta para resultar
        legibles a un usuario no técnico (a una categórica no le aporta nada
        hablar de "superior a un umbral", que siempre ronda 0.5).

        Args:
            nombre_caracteristica: Nombre de columna tal como lo conoce el
                modelo entrenado.
            umbral: Umbral de la regla en ese nodo del árbol.
            valor: Valor real del alumno para esa columna.

        Returns:
            Cláusula en lenguaje natural que describe la condición cumplida.
            Si la columna no encaja en ninguna traducción conocida, devuelve
            una frase genérica de respaldo basada en el nombre técnico y
            registra un aviso vía :class:`GestorLogs` para detectar huecos
            en el mapeo.
        """
        es_menor_igual = valor <= umbral

        match_bimestre = re.match(r"^(nota|asist)_b(\d+)$", nombre_caracteristica)
        if match_bimestre:
            tipo, bimestre = match_bimestre.groups()
            if tipo == "nota":
                comparador = "igual o inferior a" if es_menor_igual else "superior a"
                return (
                    f"su nota media en el Bimestre {bimestre} es {comparador} "
                    f"{umbral:.1f}"
                )
            comparador = "igual o inferior al" if es_menor_igual else "superior al"
            return (
                f"su asistencia en el Bimestre {bimestre} es {comparador} "
                f"{umbral * 100:.0f}%"
            )

        if nombre_caracteristica == "edad":
            comparador = "igual o inferior a" if es_menor_igual else "superior a"
            return f"su edad es {comparador} {umbral:.0f} años"

        if nombre_caracteristica == "postgrado":
            return (
                "no posee estudios de posgrado"
                if es_menor_igual
                else "posee estudios de posgrado"
            )

        for variable_base, etiqueta in ETIQUETAS_VARIABLES_CATEGORICAS.items():
            prefijo = f"{variable_base}_"
            if nombre_caracteristica.startswith(prefijo):
                categoria = nombre_caracteristica[len(prefijo) :]
                verbo = "no es" if es_menor_igual else "es"
                return f"{etiqueta} {verbo} '{categoria}'"

        self.logger.registrar(
            "EVALUADOR",
            "No hay traducción legible definida para la variable "
            f"'{nombre_caracteristica}'; se usa el nombre técnico como último "
            "recurso en la justificación.",
            "ERROR",
        )
        comparador = "igual o inferior a" if es_menor_igual else "superior a"
        return f"{nombre_caracteristica} es {comparador} {umbral:.2f}"

    def _extraer_justificacion(
        self, df_muestra: pd.DataFrame, modelo: Any, columnas_entrenamiento: list
    ) -> str:
        """Recorre el camino de decisión del árbol y traduce cada regla
        técnica a lenguaje natural.

        Args:
            df_muestra: Fila de datos del alumno (1 registro).
            modelo: Árbol de decisión entrenado.
            columnas_entrenamiento: Lista de columnas del modelo.

        Returns:
            Cláusulas del camino de decisión unidas con "y" (sin el nivel de
            riesgo ni el prefijo de hito, que añade el llamador).
        """
        nodo_indicador = modelo.decision_path(df_muestra)
        nodos_ids = nodo_indicador.indices

        condiciones = []
        valores = df_muestra.values

        for nodo_id in nodos_ids:
            if (
                modelo.tree_.children_left[nodo_id]
                == modelo.tree_.children_right[nodo_id]
            ):
                continue

            caracteristica_id = modelo.tree_.feature[nodo_id]
            umbral = modelo.tree_.threshold[nodo_id]
            nombre_caracteristica = columnas_entrenamiento[caracteristica_id]
            valor = valores[0, caracteristica_id]

            condiciones.append(
                self._traducir_regla(nombre_caracteristica, umbral, valor)
            )

        return " y ".join(condiciones)

    def ejecutar_evaluacion(
        self, df: pd.DataFrame, bimestre_corte: Optional[int] = None
    ) -> pd.DataFrame:
        """Ejecuta la evaluación de riesgo completa sobre la tabla maestra.

        Clasifica a cada alumno como HISTÓRICO, PRONÓSTICO o SIN REGISTRO.
        Para los PRONÓSTICOS, infiere probabilidad de abandono usando el modelo
        del bimestre correspondiente, asigna nivel de riesgo y genera
        justificación XAI — salvo que no tenga ningún dato real (nota y
        asistencia en 0.0 a la vez) hasta su bimestre detectado, en cuyo caso
        se marca ``SIN_DATOS_SUFICIENTES`` sin ejecutar el modelo (ver
        :meth:`_sin_datos_reales_bimestre`): un cero relleno por ausencia de
        registro no es lo mismo que un cero real, y tratarlo como tal
        generaría un falso ALTO.

        Args:
            df: Tabla maestra de estudiantes.
            bimestre_corte: Techo temporal opcional. Si es ``None`` (modo
                "Automático", por defecto), cada alumno se autodetecta en su
                propio bimestre real — la "foto de hoy", cada uno en su punto.
                Si se indica un valor N, ningún alumno se autodetecta más allá
                del Bimestre N: se recortan las columnas ``nota_bM``/
                ``asist_bM`` con M > N antes de la autodetección (ver
                :meth:`src.preparador_datos.PreparadorDatos.filtrar_columnas_por_bimestre`),
                simulando "qué se sabría si hoy fuera el Bimestre N". Actúa
                como techo, no como valor forzado: si el bimestre real de un
                alumno es menor que N (todavía no tiene datos hasta ese hito),
                se evalúa con su bimestre real en su lugar — forzarlo a N
                simularía ceros (suspenso falso) en los bimestres que aún no
                tiene, ya que :mod:`src.preparador_datos` rellena con 0.0 los
                bimestres sin datos y no hay forma de distinguir "sin datos
                todavía" de "sacó un cero".

        Returns:
            DataFrame con columnas añadidas: ``tipo_prediccion``,
            ``probabilidad_abandono``, ``nivel_riesgo`` (``ALTO``, ``MEDIO``,
            ``BAJO``, ``SIN_DATOS_SUFICIENTES`` o ``NO_CALCULABLE``),
            ``justificacion_riesgo`` (generada en lenguaje natural para los
            tres niveles de riesgo, para ``SIN_DATOS_SUFICIENTES`` y también
            para los alumnos PRONÓSTICO que quedan ``NO_CALCULABLE`` por
            falta de modelo del hito o por un fallo de preprocesado, donde
            explica cuál de los dos motivos ha sido; queda en ``"N/A"`` solo
            para quien nunca entra a evaluarse, es decir HISTÓRICO y SIN
            REGISTRO),
            ``bimestre_evaluado`` (bimestre realmente usado para evaluar a
            cada alumno PRONÓSTICO, incluidos los marcados
            ``SIN_DATOS_SUFICIENTES`` y los ``NO_CALCULABLE`` que sí llegaron
            a detectar hito; entero nulo para los que no se evalúan).
        """
        df_riesgo = df.copy()

        # 1. Clasificar alumnos: HISTÓRICO (tienen target conocido) vs PRONÓSTICO (activos sin etiqueta)
        def es_etiquetado(estado: str) -> bool:
            est = str(estado).upper()
            return any(
                p in est
                for p in self.reglas.get("palabras_bajo_riesgo", [])
                + self.reglas.get("palabras_alto_riesgo", [])
            )

        def es_evaluable(estado: str) -> bool:
            est = str(estado).upper()
            return any(p in est for p in self.reglas.get("palabras_medio_riesgo", []))

        mask_etiquetados = df_riesgo["estado_actual"].apply(es_etiquetado)
        mask_evaluables = df_riesgo["estado_actual"].apply(es_evaluable)

        df_riesgo["tipo_prediccion"] = "SIN REGISTRO"
        df_riesgo.loc[mask_etiquetados, "tipo_prediccion"] = "HISTÓRICO"
        df_riesgo.loc[mask_evaluables, "tipo_prediccion"] = "PRONÓSTICO"

        # Se inicializan las columnas de resultados.
        df_riesgo["probabilidad_abandono"] = np.nan
        df_riesgo["nivel_riesgo"] = "NO_CALCULABLE"
        df_riesgo["justificacion_riesgo"] = "N/A"
        # Int64 (nullable) porque solo los alumnos evaluados tienen un valor;
        # el resto debe quedar en <NA>, no en un 0 que podría confundirse con
        # un bimestre real.
        df_riesgo["bimestre_evaluado"] = pd.array(
            [pd.NA] * len(df_riesgo), dtype="Int64"
        )

        df_evaluar = df_riesgo[mask_evaluables].copy()

        if df_evaluar.empty:
            return df_riesgo

        # Si hay un techo temporal fijado, se recortan las columnas de bimestres
        # futuros SOLO para la autodetección del hito de cada alumno. La
        # inferencia en sí sigue usando la fila original sin recortar: una vez
        # detectado el bimestre b_alumno (que ya nunca podrá superar el techo),
        # preprocesar_fila_alumno alinea contra columnas_b, que nunca incluye
        # columnas posteriores a b_alumno.
        if bimestre_corte is not None:
            df_deteccion = PreparadorDatos.filtrar_columnas_por_bimestre(
                df_evaluar, bimestre_corte
            )
        else:
            df_deteccion = df_evaluar

        # Contadores de trazabilidad: permiten cerrar el bucle con un resumen
        # auditable de cuántos alumnos llegaron realmente al modelo y por qué
        # motivo concreto se quedó fuera cada uno de los demás.
        total_con_modelo = 0
        total_sin_datos = 0
        motivos_no_calculable: Dict[str, int] = {
            "sin_modelo_bimestre": 0,
            "preprocesado_fallido": 0,
        }

        # 2. Inferencia individualizada por hito temporal
        for idx in df_evaluar.index:
            fila_original = df_evaluar.loc[idx]
            id_alumno = fila_original.get("n_siu", idx)

            # Detectar en qué hito temporal (bimestre) se encuentra este alumno concreto
            # (recortado al techo bimestre_corte si se indicó uno)
            b_alumno = self._detectar_bimestre_alumno(df_deteccion.loc[idx])

            # Si no hay ningún dato real (nota y asistencia en 0.0 a la vez)
            # para el bimestre detectado, no hay señal sobre la que inferir:
            # ejecutar el modelo sería predecir sobre ruido. Se marca un
            # estado distinto de ALTO/MEDIO/BAJO/NO_CALCULABLE en lugar de
            # arriesgar un falso ALTO (el fichero de notas de origen no cubre
            # todavía el bimestre en curso para cerca del 90 % del alumnado
            # activo).
            if self._sin_datos_reales_bimestre(fila_original, b_alumno):
                df_riesgo.loc[idx, "nivel_riesgo"] = "SIN_DATOS_SUFICIENTES"
                df_riesgo.loc[idx, "bimestre_evaluado"] = b_alumno
                df_riesgo.loc[idx, "justificacion_riesgo"] = (
                    f"[Hito B{b_alumno}] Sin datos de asistencia ni de notas "
                    f"registrados hasta el Bimestre {b_alumno}; no se puede "
                    "evaluar el riesgo de forma fiable todavía."
                )
                total_sin_datos += 1
                continue

            # Recuperar el modelo adaptado a su realidad temporal
            modelo_b, columnas_b = self._obtener_modelo_bimestre(b_alumno)

            if modelo_b is None or columnas_b is None:
                # Si el modelo de ese hito no existe, el alumno queda como no
                # calculable por seguridad, pero dejando constancia del motivo:
                # un NO_CALCULABLE silencioso es indistinguible de un fallo del
                # pipeline para quien audita los resultados después.
                self.logger.registrar(
                    "EVALUADOR",
                    f"No existe el modelo entrenado o el fichero de columnas "
                    f"del Bimestre {b_alumno} (alumno {id_alumno}); no se "
                    "puede calcular su riesgo.",
                    "ERROR",
                )
                df_riesgo.loc[idx, "bimestre_evaluado"] = b_alumno
                df_riesgo.loc[idx, "justificacion_riesgo"] = (
                    f"[Hito B{b_alumno}] No hay ningún modelo entrenado para "
                    f"el Bimestre {b_alumno}, así que no se ha podido calcular "
                    "el riesgo de este alumno."
                )
                motivos_no_calculable["sin_modelo_bimestre"] += 1
                continue

            # Preprocesar la fila del alumno alineándola al esquema del modelo del bimestre actual
            X_inferencia = preprocesar_fila_alumno(
                fila_original, columnas_b, self.logger
            )
            if X_inferencia is None:
                self.logger.registrar(
                    "EVALUADOR",
                    f"No se ha podido preparar el vector de características "
                    f"del alumno {id_alumno} para el Bimestre {b_alumno}; "
                    "queda como NO_CALCULABLE.",
                    "ERROR",
                )
                df_riesgo.loc[idx, "bimestre_evaluado"] = b_alumno
                df_riesgo.loc[idx, "justificacion_riesgo"] = (
                    f"[Hito B{b_alumno}] No se han podido preparar los datos "
                    f"de este alumno para el modelo del Bimestre {b_alumno}, "
                    "así que no se ha podido calcular su riesgo."
                )
                motivos_no_calculable["preprocesado_fallido"] += 1
                continue

            total_con_modelo += 1

            # Calcular probabilidad (Clase 1 = Abandono)
            prob = float(modelo_b.predict_proba(X_inferencia)[0, 1])
            df_riesgo.loc[idx, "probabilidad_abandono"] = prob
            df_riesgo.loc[idx, "bimestre_evaluado"] = b_alumno

            # Aplicar reglas de negocio para los umbrales
            if prob >= self.umbral_alto:
                nivel = "ALTO"
            elif prob >= self.umbral_medio:
                nivel = "MEDIO"
            else:
                nivel = "BAJO"

            df_riesgo.loc[idx, "nivel_riesgo"] = nivel

            # 4. Explicabilidad Local (XAI) con el árbol correcto, traducida
            # a una frase en lenguaje natural para el usuario no técnico.
            # Se genera para los tres niveles (también BAJO): la ruta de
            # decisión ya se resalta en el árbol para BAJO igual que para
            # ALTO/MEDIO, así que dejar "N/A" ahí es una
            # inconsistencia — el alumno también "se clasifica en riesgo BAJO
            # porque..." tiene una explicación igual de genuina y accionable
            # (confirma que el low-risk no es un valor por defecto sin más).
            condiciones = self._extraer_justificacion(
                X_inferencia, modelo_b, columnas_b
            )
            justificacion = (
                f"El alumno se clasifica en riesgo {nivel} porque {condiciones}."
            )
            df_riesgo.loc[idx, "justificacion_riesgo"] = (
                f"[Hito B{b_alumno}] {justificacion}"
            )

        # Resumen final de la evaluación: deja en la auditoría el reparto real
        # de los alumnos PRONÓSTICO entre inferencia efectiva, falta de datos y
        # fallos, con el desglose por motivo de estos últimos.
        total_no_calculable = sum(motivos_no_calculable.values())
        desglose = ", ".join(
            f"{motivo}: {cuenta}" for motivo, cuenta in motivos_no_calculable.items()
        )
        self.logger.registrar(
            "EVALUADOR",
            f"Evaluación finalizada sobre {len(df_evaluar)} alumnos PRONÓSTICO: "
            f"{total_con_modelo} evaluados con modelo, "
            f"{total_sin_datos} SIN_DATOS_SUFICIENTES, "
            f"{total_no_calculable} NO_CALCULABLE ({desglose}).",
            "ERROR" if total_no_calculable else "EXITO",
        )

        return df_riesgo
