import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Importa la configuración global
from src.config import config


class Visualizador:
    """Módulo 5: Presentación de resultados."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            nombre_carpeta: str = config.paths.get("output_dir", "output")
            base_dir: str = os.path.dirname(os.path.dirname(__file__))
            self.output_dir: str = os.path.join(base_dir, nombre_carpeta)
        else:
            self.output_dir = output_dir

        # Asegura que exista la carpeta output
        os.makedirs(self.output_dir, exist_ok=True)

    def mostrar_en_consola(
        self, df: Optional[pd.DataFrame], stats: Dict[str, Any]
    ) -> None:
        print("\n" + "=" * 40)
        print("   REPORTE DE SEGUIMIENTO ACADÉMICO")
        print("=" * 40)
        print(f"Alumnos procesados: {stats.get('total_alumnos', 0)}")
        print(
            f"Nota media global ({stats.get('columna_analizada', 'N/A')}): {stats.get('media_aritmetica', 0)}"
        )
        print("-" * 40)
        print("Muestra de Riesgos Asignados:")
        if df is not None:
            cols_ver = ["n_siu", "estado_actual", "nivel_riesgo"]
            # Filtramos solo columnas que existan
            cols_finales = [c for c in cols_ver if c in df.columns]
            print(df[cols_finales].head(10))
        print("=" * 40 + "\n")

    def exportar_csv(self, df: pd.DataFrame) -> None:
        # Guardamos el fichero físico en la carpeta output
        fecha: str = datetime.now().strftime("%Y%m%d_%H%M")
        nombre_fichero: str = f"resultado_riesgos_{fecha}.csv"
        ruta_completa: str = os.path.join(self.output_dir, nombre_fichero)

        df.to_csv(ruta_completa, index=False, sep=";", encoding="latin1")
        print(f"[Output] Archivo generado: {ruta_completa}")
