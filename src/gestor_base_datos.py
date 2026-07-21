"""
Gestor de persistencia en SQLite.

Proporciona métodos para inicializar tablas, guardar datos RAW y la tabla
maestra, y registrar eventos de trazabilidad en la base de datos.

Uso::

    from src.gestor_base_datos import GestorBaseDatos
    db = GestorBaseDatos()
    db.inicializar_tablas_fijas()
"""

import os
import sqlite3
from typing import Dict, Optional

import pandas as pd

# Importa la configuración global
from src.config import config


class GestorBaseDatos:
    """Gestor de persistencia en SQLite con estrategia de reemplazo dinámico.

    Attributes:
        db_path: Ruta absoluta al archivo de base de datos SQLite.
    """

    def __init__(self, db_name: Optional[str] = None) -> None:
        """Inicializa la ruta de la base de datos.

        Args:
            db_name: Nombre del archivo de SQLite. Si es ``None``, se usa
                el valor de ``config.paths.db_name``.
        """
        if db_name is None:
            db_name = config.paths.get("db_name", "estudiantes.db")
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.db_path: str = os.path.join(base_dir, str(db_name))

    def inicializar_tablas_fijas(self) -> None:
        """Crea la tabla de logs necesaria para el arranque."""
        query_logs = """
        CREATE TABLE IF NOT EXISTS logs_ejecucion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            modulo TEXT,
            estado TEXT,
            mensaje TEXT
        );
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query_logs)

    def guardar_datos_raw(self, datasets_dict: Dict[str, pd.DataFrame]) -> None:
        """Guarda los datos de entrada en tablas ``raw_`` de SQLite.

        Args:
            datasets_dict: Diccionario ``{clave: DataFrame}`` a persistir.
        """
        with sqlite3.connect(self.db_path) as conn:
            for nombre, df in datasets_dict.items():
                df.to_sql(f"raw_{nombre}", conn, if_exists="replace", index=False)

    def guardar_datos_master(self, df: pd.DataFrame) -> None:
        """Guarda la tabla maestra en SQLite con ``if_exists='replace'``.

        Args:
            df: DataFrame con la tabla maestra de estudiantes.
        """
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("master_estudiantes", conn, if_exists="replace", index=False)

    def registrar_log(self, modulo: str, estado: str, mensaje: str) -> None:
        """Inserta un registro en la tabla ``logs_ejecucion``.

        Args:
            modulo: Identificador del módulo que genera el log.
            estado: Estado del evento (``EXITO``, ``ERROR``, etc.).
            mensaje: Descripción del evento.
        """
        query = "INSERT INTO logs_ejecucion (modulo, estado, mensaje) VALUES (?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, (modulo, estado, mensaje))
