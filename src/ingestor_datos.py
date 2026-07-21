"""
Módulo de ingesta y carga de datos.

Lee los archivos de entrada (CSV y Excel) definidos en ``config.yaml``,
los carga en DataFrames de pandas y los almacena para su uso posterior
en el pipeline de preparación.

Uso::

    from src.ingestor_datos import IngestorDatos
    ingestor = IngestorDatos()
    datasets = ingestor.leer_datos()
"""

import os
from typing import Dict, Optional

import pandas as pd

from src.config import config


class IngestorDatos:
    """Módulo 1: Carga y validación de datos de entrada.

    Attributes:
        ruta: Ruta absoluta al directorio de datos.
        datasets: Diccionario con los DataFrames cargados, indexados por clave.
    """

    def __init__(self, ruta_data: Optional[str] = None) -> None:
        """Inicializa el ingestor con la ruta al directorio de datos.

        Args:
            ruta_data: Ruta al directorio de datos. Si es ``None``, se usa
                ``config.paths.data_dir`` relativo a la raíz del proyecto.
        """
        if ruta_data is None:
            # Sube un nivel desde 'src' y concatena con la ruta del YAML
            base_dir: str = os.path.dirname(os.path.dirname(__file__))
            nombre_carpeta: str = config.paths.get("data_dir", "data")
            self.ruta = os.path.join(base_dir, nombre_carpeta)
        else:
            self.ruta = ruta_data
        self.datasets: Dict[str, pd.DataFrame] = {}

    def leer_datos(self) -> Dict[str, pd.DataFrame]:
        """Lee todos los archivos definidos en ``config.datasets``.

        Returns:
            Diccionario ``{clave: DataFrame}`` con los datos cargados.
        """
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
