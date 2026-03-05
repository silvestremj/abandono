import logging
import os
from datetime import datetime
from typing import Optional

from src.gestor_base_datos import GestorBaseDatos


class GestorLogs:
    """Módulo 6: Trazabilidad y Logs (Triple salida: Consola, DB, Fichero)."""

    def __init__(self, gestor_db: Optional[GestorBaseDatos] = None) -> None:
        """Inicializa la sesión y los manejadores de logs.

        Args:
            gestor_db (GestorBaseDatos, optional): Instancia para persistir logs en SQLite.
        """
        self.db = gestor_db
        self.sesion: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Crear carpeta de logs en la raíz si no existe
        self.log_dir: str = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 2. Configurar el nombre del archivo (un log distinto por día)
        fecha_archivo: str = datetime.now().strftime("%Y%m%d")
        log_path: str = os.path.join(self.log_dir, f"ejecucion_{fecha_archivo}.log")

        # 3. Configurar el logger nativo de Python
        self.logger = logging.getLogger("TFM_Logger")
        self.logger.setLevel(logging.INFO)

        # Evitamos duplicar handlers si la clase se instancia varias veces
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            # Formato estándar de la industria para el interior del fichero de texto
            formatter = logging.Formatter(
                "%(asctime)s - [%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Imprimimos en consola y guardamos en el fichero el inicio de sesión
        print(f"--- Inicio de Sesión: {self.sesion} ---")
        self.logger.info(f"--- Inicio de Sesión: {self.sesion} ---")

    def registrar(self, modulo: str, mensaje: str, estado: str = "EXITO") -> None:
        """Registra un evento en consola, fichero y base de datos."""
        hora: str = datetime.now().strftime("%H:%M:%S")

        # 1. Salida por consola (vista en tiempo real)
        print(f"[{hora}] [{modulo}] {mensaje}")

        # 2. Salida a Fichero (Fallback de seguridad)
        mensaje_fichero: str = f"[{modulo}] {mensaje}"
        if estado == "ERROR":
            self.logger.error(mensaje_fichero)
        else:
            self.logger.info(mensaje_fichero)

        # 3. Persistencia en DB (Auditoría estructurada)
        if self.db:
            self.db.registrar_log(modulo, estado, mensaje)
