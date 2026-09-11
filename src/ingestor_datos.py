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

    @staticmethod
    def _tiene_bom_utf8(path: str) -> bool:
        """Comprueba si un fichero empieza por el BOM de UTF-8 (``EF BB BF``).

        Args:
            path: Ruta al fichero a inspeccionar.

        Returns:
            ``True`` si los 3 primeros bytes son el BOM UTF-8.
        """
        with open(path, "rb") as fh:
            return fh.read(3) == b"\xef\xbb\xbf"

    @classmethod
    def leer_archivo(cls, path: str, conf: Dict[str, str]) -> pd.DataFrame:
        """Lee un único fichero con los parámetros declarados para su clave.

        Es la fuente única de verdad sobre *cómo* se lee cada fuente (tipo,
        separador, encoding y detección de BOM). Además de :meth:`leer_datos`,
        la reutiliza la validación de ficheros subidos por la interfaz web
        (:func:`src.visualizador.validar_estructura_dataset`), para que lo que
        se valida al subir sea exactamente lo que luego se ingesta.

        Args:
            path: Ruta al fichero a leer.
            conf: Entrada de ``config.datasets`` con ``tipo`` y, para CSV,
                ``sep`` y ``encoding``.

        Returns:
            DataFrame con el contenido del fichero.

        Raises:
            ValueError: Si ``conf["tipo"]`` no es ``csv`` ni ``excel``.
        """
        if conf["tipo"] == "csv":
            encoding = conf.get("encoding", "utf-8")
            # utf-8-sig elimina el BOM al inicio de archivos provenientes de Excel
            if encoding == "utf-8":
                encoding = "utf-8-sig"
            elif cls._tiene_bom_utf8(path):
                # El encoding declarado en config.yaml (p.ej. "latin1")
                # puede no coincidir con el real: un CSV con BOM UTF-8
                # leído como latin1 decodifica esos 3 bytes como
                # caracteres sueltos que se prepend al primer nombre
                # de columna (p.ej. "n_siu" -> "ï»¿n_siu"), rompiendo
                # cualquier normalización/cruce posterior sobre esa
                # columna. El BOM es una prueba inequívoca de que el
                # fichero es UTF-8 pese a lo declarado.
                encoding = "utf-8-sig"
            return pd.read_csv(path, sep=conf["sep"], encoding=encoding)

        if conf["tipo"] == "excel":
            # Importante: Tener instalado 'openpyxl'
            return pd.read_excel(path)

        raise ValueError(f"Tipo de fichero no soportado: {conf['tipo']!r}")

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
                if conf["tipo"] in ("csv", "excel"):
                    self.datasets[clave] = self.leer_archivo(path, conf)
            else:
                print(f" Error: No se encuentra el archivo en {path}")
                print(
                    "   Por favor, verifica que el archivo esté en la carpeta 'data' con el nombre exacto."
                )

        return self.datasets
