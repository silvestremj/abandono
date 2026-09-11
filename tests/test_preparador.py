import re

import pandas as pd

from src.preparador_datos import PreparadorDatos


class TestPreparadorDatos:
    """Batería de tests unitarios para la limpieza y segmentación de datos."""

    def setup_method(self):
        """Prepara el entorno antes de cada test."""
        # Se instancia el preparador con un diccionario vacío.
        # No hace falta cargar los Excel reales porque se prueban funciones aisladas.
        self.preparador = PreparadorDatos(datasets_dict={})

    def test_normalizar_espacios_y_mayusculas(self):
        # Columna con espacios a los lados y mezcla de mayúsculas/minúsculas.
        df_prueba = pd.DataFrame(
            {"estudio": ["  cese  ", "CeIoT", " Especializacion "]}
        )

        resultado = self.preparador._normalizar(df_prueba, "estudio")

        # Se comprueba que el resultado es exacto.
        assert resultado.iloc[0] == "CESE"
        assert resultado.iloc[1] == "CEIOT"
        assert resultado.iloc[2] == "ESPECIALIZACION"

    def test_normalizar_tildes_y_caracteres_especiales(self):
        # Textos con tildes y la letra eñe.
        df_prueba = pd.DataFrame(
            {"asignatura": ["Estadística", "Áéíóú", "Diseño", "Gestión"]}
        )

        resultado = self.preparador._normalizar(df_prueba, "asignatura")

        # Se comprueba que las tildes desaparecen y la Ñ pasa a N (por la normalización ASCII).
        assert resultado.iloc[0] == "ESTADISTICA"
        assert resultado.iloc[1] == "AEIOU"
        assert resultado.iloc[2] == "DISENO"
        assert resultado.iloc[3] == "GESTION"

    def test_normalizar_numeros_como_texto(self):
        # Pandas ha leído el ID del alumno como un número en lugar de texto.
        df_prueba = pd.DataFrame({"n_siu": [12345, 67890]})

        resultado = self.preparador._normalizar(df_prueba, "n_siu")

        # Se comprueba que se ha forzado la conversión a texto (string).
        assert resultado.iloc[0] == "12345"
        assert resultado.iloc[1] == "67890"

    def test_filtrar_columnas_por_bimestre(self):
        """Verifica que el filtro temporal oculte el futuro y mantenga el presente y los datos fijos."""
        # Creación de un DataFrame con estructura maestra completa (B1 a B6 + datos fijos)
        df_maestro_test = pd.DataFrame(
            [
                {
                    "n_siu": "1001",
                    "target_ml": 1.0,
                    "provincia": "BUENOS AIRES",
                    "nota_b1": 8.0,
                    "asist_b1": 0.85,
                    "nota_b2": 7.0,
                    "asist_b2": 0.90,
                    "nota_b3": 6.0,
                    "asist_b3": 0.50,
                    "nota_b4": 5.0,
                    "asist_b4": 0.40,
                    "nota_b5": 4.0,
                    "asist_b5": 0.30,
                    "nota_b6": 0.0,
                    "asist_b6": 0.0,
                }
            ]
        )

        # Se solicita el corte en el Bimestre 2 (el futuro es del B3 al B6).
        df_resultado = self.preparador.filtrar_columnas_por_bimestre(
            df_maestro_test, bimestre_corte=2
        )

        # 1. COMPROBACIÓN DE COLUMNAS CONSERVADAS
        assert "n_siu" in df_resultado.columns  # Clave primaria fija
        assert "target_ml" in df_resultado.columns  # Variable objetivo fija
        assert "provincia" in df_resultado.columns  # Variable demográfica fija
        assert "nota_b1" in df_resultado.columns  # Pasado
        assert "nota_b2" in df_resultado.columns  # Presente / Límite

        # 2. COMPROBACIÓN DE COLUMNAS ELIMINADAS (El futuro no debe existir)
        assert "nota_b3" not in df_resultado.columns
        assert "asist_b3" not in df_resultado.columns
        assert "nota_b6" not in df_resultado.columns
        assert "asist_b6" not in df_resultado.columns

        # 3. VERIFICACIÓN DE INTEGRIDAD DE FILAS
        # El filtro debe reducir columnas, pero nunca alterar la cantidad de registros
        assert len(df_resultado) == 1

    def test_deduplicacion_prioriza_fila_con_datos_reales(self):
        """Si la fila duplicada más reciente de un alumno está en blanco pero
        una fila ANTERIOR tenía su desenlace real, la deduplicación no debe
        descartar el dato real. Antes se aplicaba "prevalece el último
        registro" a secas; se detectó empíricamente en los datos de origen
        que ~13% de los grupos duplicados pierden así un Abandono/Recibido
        real en favor de una fila vacía posterior del mismo alumno."""
        df_actual_test = pd.DataFrame(
            [
                {
                    "n_siu": "1001",
                    "estudio": "CESE",
                    "nota": "7,5",
                    "estado_actual": "Abandono",
                },
                {
                    # Fila en blanco posterior del MISMO alumno/estudio: no
                    # debe "ganar" solo por venir después.
                    "n_siu": "1001",
                    "estudio": "CESE",
                    "nota": None,
                    "estado_actual": None,
                },
            ]
        )
        df_notas_test = pd.DataFrame(
            [
                {
                    # Alumno distinto, solo para que el pivot tenga con qué trabajar.
                    "n_siu": "9999",
                    "Estudio": "CESE",
                    "nota_m": 5.0,
                    "asist_m": 0.5,
                    "Comentario": "Bimestre: 1.0",
                }
            ]
        )

        preparador = PreparadorDatos(
            {"actual": df_actual_test, "notas_bimestre": df_notas_test}
        )
        resultado = preparador.ejecutar_preparacion()

        assert len(resultado) == 1
        assert resultado.loc[0, "estado_actual"] == "Abandono"

    def test_columnas_unnamed_y_sd_no_sobreviven_a_preparar_dataset_ml(self):
        """Bug: COLUMNAS_EXCLUIDAS_ML listaba "Unnamed: N" con mayúscula
        inicial, pero ejecutar_preparacion normaliza las columnas del Excel
        a minúsculas antes — la comparación (case-sensitive) nunca
        coincidía y una columna "Unnamed: 30" con texto suelto de Excel
        colaba como feature real tras el One-Hot Encoding. Además,
        sd_nota/sd_asist (desviación típica de la media global) son la
        misma fuga de datos que nota/asist y tampoco estaban excluidas."""
        df_maestro_test = pd.DataFrame(
            [
                {
                    "estado_actual": "Recibido",
                    "nota_b1": 8.0,
                    "asist_b1": 0.85,
                    "Unnamed: 28": "Formulario de baja",
                    "unnamed: 29": "Identificación curso/estudiante",
                    "sd_nota": 1.2,
                    "sd_asist": 0.1,
                }
            ]
        )

        df_ml = self.preparador.preparar_dataset_ml(df_maestro_test)

        # Ninguna columna resultante (ni siquiera tras el One-Hot Encoding,
        # que generaría "Unnamed: 28_Formulario de baja") debe rastrear
        # las columnas "Unnamed" del Excel de origen.
        assert not any("unnamed" in str(c).lower() for c in df_ml.columns)
        assert "sd_nota" not in df_ml.columns
        assert "sd_asist" not in df_ml.columns
        # Y sigue conservando las columnas predictivas legítimas.
        assert "nota_b1" in df_ml.columns
        assert "asist_b1" in df_ml.columns

    def test_columnas_baja_no_se_confunden_con_columnas_de_bimestre(self):
        """Las columnas fecha_baja, estado_baja, causa_baja y comentario_baja
        contienen la subcadena "_b" y antes se detectaban como columnas de
        bimestre (nota_bN/asist_bN), quedando convertidas a 0.0 por el
        relleno de nulos y la coerción numérica de esas columnas."""
        df_actual_test = pd.DataFrame(
            [
                {
                    "n_siu": "1001",
                    "estudio": "CESE",
                    "estado_actual": "Abandono",
                    "estado_baja": "ABANDONÓ",
                    "causa_baja": "Tema personal.",
                    "fecha_baja": 2021,
                    "comentario_baja": None,
                    "Unnamed: 28": None,
                },
                {
                    "n_siu": "1002",
                    "estudio": "CESE",
                    "estado_actual": "Recibido",
                    "estado_baja": None,
                    "causa_baja": None,
                    "fecha_baja": None,
                    "comentario_baja": None,
                    "Unnamed: 28": None,
                },
            ]
        )
        df_notas_test = pd.DataFrame(
            [
                {
                    "n_siu": "1001",
                    "Estudio": "CESE",
                    "nota_m": 8.0,
                    "asist_m": 0.8,
                    "Comentario": "Bimestre: 1.0",
                },
                {
                    "n_siu": "1002",
                    "Estudio": "CESE",
                    "nota_m": 6.0,
                    "asist_m": 0.6,
                    "Comentario": "Bimestre: 1.0",
                },
            ]
        )

        preparador = PreparadorDatos(
            {"actual": df_actual_test, "notas_bimestre": df_notas_test}
        )
        resultado = preparador.ejecutar_preparacion()

        fila_baja = resultado.loc[resultado["n_siu"] == "1001"].iloc[0]
        fila_sin_baja = resultado.loc[resultado["n_siu"] == "1002"].iloc[0]

        # 1. Las columnas de baja del alumno con baja conservan su texto exacto.
        assert fila_baja["estado_baja"] == "ABANDONÓ"
        assert fila_baja["causa_baja"] == "Tema personal."

        # 2. El alumno sin baja conserva nulos, no 0.0.
        assert pd.isna(fila_sin_baja["estado_baja"])
        assert pd.isna(fila_sin_baja["causa_baja"])

        # 3. Las columnas de bimestre sí son numéricas.
        assert resultado["nota_b1"].dtype == float
        assert resultado["asist_b1"].dtype == float

        # 4. Ninguna columna "Unnamed" sobrevive en la tabla maestra.
        assert not any(
            re.match(r"^unnamed", str(c), re.IGNORECASE) for c in resultado.columns
        )

    def test_deduplicacion_sin_datos_reales_conserva_comportamiento_anterior(self):
        """Si NINGUNA fila del grupo duplicado tiene datos reales, se sigue
        aplicando el criterio anterior de quedarse con la última (no hay
        ninguna candidata "informativa" que priorizar)."""
        df_actual_test = pd.DataFrame(
            [
                {"n_siu": "1002", "estudio": "CESE", "nota": None, "estado_actual": None},
                {"n_siu": "1002", "estudio": "CESE", "nota": None, "estado_actual": None},
            ]
        )
        df_notas_test = pd.DataFrame(
            [
                {
                    "n_siu": "9999",
                    "Estudio": "CESE",
                    "nota_m": 5.0,
                    "asist_m": 0.5,
                    "Comentario": "Bimestre: 1.0",
                }
            ]
        )

        preparador = PreparadorDatos(
            {"actual": df_actual_test, "notas_bimestre": df_notas_test}
        )
        resultado = preparador.ejecutar_preparacion()

        assert len(resultado) == 1
        assert resultado.loc[0, "estado_actual"] == "SIN REGISTRO"
