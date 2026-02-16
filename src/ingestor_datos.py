import os

import pandas as pd


class IngestorDatos:
    """Módulo 1: Carga y validación de datos."""

    def __init__(self, ruta_data=None):
        if ruta_data is None:
            # Sube un nivel desde 'src' y entra en 'data'
            self.ruta = os.path.join(os.path.dirname(__file__), "..", "data")
        else:
            self.ruta = ruta_data
        self.datasets = {}

    def leer_datos(self):
        # Configuramos cada archivo de forma individual y limpia
        archivos = {
            "inscripciones": {
                "nombre": "LSE_Inscrip_Baja_Recibido.csv",
                "tipo": "csv",
                "sep": ";",
                "enc": "latin1",
            },
            "notas_bimestre": {
                "nombre": "LSE_Notas_Estadistica_Bimestre.csv",
                "tipo": "csv",
                "sep": ";",
                "enc": "latin1",
            },
            "actual": {"nombre": "LSE_Notas_Inscrip_Baja_Actual.xlsx", "tipo": "excel"},
        }

        for clave, conf in archivos.items():
            path = os.path.join(self.ruta, conf["nombre"])

            if os.path.exists(path):
                print(f"Cargando {clave} desde {conf['nombre']}...")
                if conf["tipo"] == "csv":
                    self.datasets[clave] = pd.read_csv(
                        path, sep=conf["sep"], encoding=conf["enc"]
                    )
                elif conf["tipo"] == "excel":
                    # Importante: Asegúrate de tener instalado 'openpyxl'
                    self.datasets[clave] = pd.read_excel(path)
            else:
                print(f"⚠️ Error: No se encuentra el archivo en {path}")
                print(
                    "   Por favor, verifica que el archivo esté en la carpeta 'data' con el nombre exacto."
                )

        return self.datasets
