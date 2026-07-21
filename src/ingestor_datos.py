import os
from typing import Dict, Optional

import pandas as pd

# Importamos la configuración
# Nota: Asegúrate de que 'config.py' esté en el mismo nivel que 'ingestor_datos.py' o ajusta la ruta de importación según sea necesario.
from src.config import config


class IngestorDatos:
    """Módulo 1: Carga y validación de datos."""

    def __init__(self, ruta_data: Optional[str] = None) -> None:
        if ruta_data is None:
            # Sube un nivel desde 'src' y concatena con la ruta del YAML
            base_dir: str = os.path.dirname(os.path.dirname(__file__))
            nombre_carpeta: str = config.paths.get("data_dir", "data")
            self.ruta = os.path.join(base_dir, nombre_carpeta)
        else:
            self.ruta = ruta_data
        self.datasets: Dict[str, pd.DataFrame] = {}

    def leer_datos(self) -> Dict[str, pd.DataFrame]:
        # Trae la configuración de los datasets desde el YAML
        archivos: Dict[str, Dict[str, str]] = config.datasets

        for clave, conf in archivos.items():
            path: str = os.path.join(self.ruta, conf["nombre"])

            if os.path.exists(path):
                print(f"Cargando {clave} desde {conf['nombre']}...")
                if conf["tipo"] == "csv":
                    encoding = conf.get("encoding", "utf-8")
                    # utf-8-sig elimina el BOM al inicio de archivos provenientes de Excel
                    if encoding == "utf-8":
                        encoding = "utf-8-sig"
                    self.datasets[clave] = pd.read_csv(
                        path, sep=conf["sep"], encoding=encoding
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
