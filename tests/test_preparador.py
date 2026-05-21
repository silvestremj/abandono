import pandas as pd

from src.preparador_datos import PreparadorDatos


class TestPreparadorDatos:
    """Batería de tests unitarios para la limpieza y segmentación de datos (Espiral 3)."""

    def setup_method(self):
        """Prepara el entorno antes de cada test."""
        # Instanciamos el preparador con un diccionario vacío.
        # No necesitamos cargar los Excel reales porque probamos funciones aisladas.
        self.preparador = PreparadorDatos(datasets_dict={})

    def test_normalizar_espacios_y_mayusculas(self):
        # Simulamos una columna con espacios a los lados y mezcla de mayúsculas/minúsculas
        df_prueba = pd.DataFrame(
            {"estudio": ["  cese  ", "CeIoT", " Especializacion "]}
        )

        resultado = self.preparador._normalizar(df_prueba, "estudio")

        # Comprobamos que el resultado es exacto
        assert resultado.iloc[0] == "CESE"
        assert resultado.iloc[1] == "CEIOT"
        assert resultado.iloc[2] == "ESPECIALIZACION"

    def test_normalizar_tildes_y_caracteres_especiales(self):
        # Simulamos textos con tildes y la letra eñe
        df_prueba = pd.DataFrame(
            {"asignatura": ["Estadística", "Áéíóú", "Diseño", "Gestión"]}
        )

        resultado = self.preparador._normalizar(df_prueba, "asignatura")

        # Comprobamos que las tildes desaparecen y la Ñ pasa a N (por la normalización ASCII)
        assert resultado.iloc[0] == "ESTADISTICA"
        assert resultado.iloc[1] == "AEIOU"
        assert resultado.iloc[2] == "DISENO"
        assert resultado.iloc[3] == "GESTION"

    def test_normalizar_numeros_como_texto(self):
        # Simulamos que Pandas ha leído el ID del alumno como un número en lugar de texto
        df_prueba = pd.DataFrame({"n_siu": [12345, 67890]})

        resultado = self.preparador._normalizar(df_prueba, "n_siu")

        # Comprobamos que lo ha forzado a convertirse en texto (string)
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

        # Simulamos que solicitamos el corte en el Bimestre 2 (El futuro es del B3 al B6)
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
