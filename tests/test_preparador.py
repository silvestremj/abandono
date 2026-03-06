import pandas as pd

from src.preparador_datos import PreparadorDatos


class TestPreparadorDatos:
    """Batería de tests unitarios para la limpieza de datos (Espiral 2)."""

    def setup_method(self):
        """Prepara el entorno antes de cada test."""
        # Instanciamos el preparador con un diccionario vacío.
        # No necesitamos cargar los Excel reales porque solo vamos a probar una función aislada.
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
