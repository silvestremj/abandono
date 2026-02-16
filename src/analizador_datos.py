class Analizador:
    """Módulo 3: Análisis descriptivo y temporal."""

    def calcular_estadisticas_basicas(self, df):
        # Validación defensiva
        if df is None or df.empty:
            return {"error": "El DataFrame está vacío"}

        # Cálculo simple (Bala Trazadora)
        try:
            # Asumiendo que existen columnas que empiezan por 'nota_'
            cols_notas = [c for c in df.columns if "nota_" in c]
            if not cols_notas:
                return {"warning": "No hay columnas de notas para analizar"}

            # Media de la primera columna de notas encontrada
            col_ref = cols_notas[0]
            media = df[col_ref].mean()
            return {
                "columna_analizada": col_ref,
                "media_aritmetica": round(media, 2),
                "total_alumnos": len(df),
            }
        except Exception as e:
            return {"error": str(e)}
