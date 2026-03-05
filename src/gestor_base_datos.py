import os
import sqlite3
from typing import Dict

import pandas as pd


class GestorBaseDatos:
    """Gestor de persistencia en SQLite con estrategia de reemplazo dinámico."""

    def __init__(self, db_name: str = "estudiantes.db") -> None:
        self.db_path: str = os.path.join(os.path.dirname(__file__), "..", db_name)

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
        """Guarda los datos de entrada en tablas raw_."""
        with sqlite3.connect(self.db_path) as conn:
            for nombre, df in datasets_dict.items():
                df.to_sql(f"raw_{nombre}", conn, if_exists="replace", index=False)

    def guardar_datos_master(self, df: pd.DataFrame) -> None:
        """Guarda la tabla maestra permitiendo cambios de estructura."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("master_estudiantes", conn, if_exists="replace", index=False)

    def registrar_log(self, modulo: str, estado: str, mensaje: str) -> None:
        """Método interno para insertar logs en la tabla."""
        query = "INSERT INTO logs_ejecucion (modulo, estado, mensaje) VALUES (?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, (modulo, estado, mensaje))
