class EvaluadorRiesgo:
    """Módulo 4: Cálculo de riesgo (Reglas de negocio)."""

    def asignar_riesgo(self, estado):
        # Lógica Hardcoded inicial [cite: 4]
        estado = str(estado).upper()
        if "ABANDONO" in estado:
            return "ALTO"
        if "RECIBIDO" in estado:
            return "BAJO"
        # 'En curso' o 'En pausa' -> Riesgo MEDIO [cite: 4]
        # (Posteriormente refinaremos 'En pausa' como riesgo mayor)
        return "MEDIO"

    def ejecutar_evaluacion(self, df):
        if df is None or "estado_actual" not in df.columns:
            raise ValueError("El DataFrame no tiene la columna 'estado_actual'")

        df_riesgo = df.copy()
        df_riesgo["nivel_riesgo"] = df_riesgo["estado_actual"].apply(
            self.asignar_riesgo
        )
        return df_riesgo
