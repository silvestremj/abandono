from datetime import datetime


class GestorLogs:
    """Módulo 6: Trazabilidad y Logs."""

    def __init__(self):
        self.sesion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"--- Inicio de Sesión: {self.sesion} ---")

    def registrar(self, modulo, mensaje):
        hora = datetime.now().strftime("%H:%M:%S")
        print(f"[{hora}] [{modulo}] {mensaje}")
