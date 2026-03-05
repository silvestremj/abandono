import os
from typing import Dict, Optional

import pandas as pd


class IngestorDatos:
    """Módulo 1: Carga y validación de datos."""

    def __init__(self, ruta_data: Optional[str] = None) -> None:
        if ruta_data is None:
            # Sube un nivel desde 'src' y entra en 'data'
            self.ruta = os.path.join(os.path.dirname(__file__), "..", "data")
        else:
            self.ruta = ruta_data
        self.datasets: Dict[str, pd.DataFrame] = {}

    def leer_datos(self) -> Dict[str, pd.DataFrame]:
        # Configuramos cada archivo de forma individual y limpia
        archivos: Dict[str, Dict[str, str]] = {
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
            path: str = os.path.join(self.ruta, conf["nombre"])

            if os.path.exists(path):
                print(f"Cargando {clave} desde {conf['nombre']}...")
                if conf["tipo"] == "csv":
                    self.datasets[clave] = pd.read_csv(
                        path, sep=conf["sep"], encoding=conf["enc"]
                    )
                elif conf["tipo"] == "excel":
                    # Importante: Tener instalado 'openpyxl'
                    self.datasets[clave] = pd.read_excel(path)
            else:
                print(f" Error: No se encuentra el archivo en {path}")
                print(
                    "   Por favor, verifica que el archivo esté en la carpeta 'data' con el nombre exacto."
                )

        return self.datasets
