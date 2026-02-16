from datetime import datetime


class GestorLogs:
    """Módulo 6: Trazabilidad y Logs (Híbrido)."""

    def __init__(self, gestor_db=None):
        """Inicializa la sesión.

        Args:
            gestor_db (GestorBaseDatos, optional): Instancia para persistir logs en SQLite.
        """
        self.db = gestor_db
        self.sesion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"--- Inicio de Sesión: {self.sesion} ---")

    def registrar(self, modulo, mensaje, estado="EXITO"):
        """Registra un evento en consola y, opcionalmente, en la base de datos."""
        hora = datetime.now().strftime("%H:%M:%S")

        # 1. Salida por consola (lo de siempre)
        print(f"[{hora}] [{modulo}] {mensaje}")

        # 2. Persistencia en DB (si el gestor está disponible)
        if self.db:
            self.db.registrar_log(modulo, estado, mensaje)
