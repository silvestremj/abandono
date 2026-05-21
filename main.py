import os
import sys

from src.entrenador_modelos import EntrenadorModelos

# Aseguramos que Python encuentre los módulos dentro de 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Imports con los nombres de archivos y clases
from src.analizador_datos import Analizador
from src.config import config
from src.evaluador_riesgo import EvaluadorRiesgo
from src.gestor_base_datos import GestorBaseDatos
from src.gestor_logs import GestorLogs
from src.ingestor_datos import IngestorDatos
from src.preparador_datos import PreparadorDatos
from src.visualizador import Visualizador


def main() -> None:
    # 0. Inicialización de infraestructura
    db = GestorBaseDatos()
    db.inicializar_tablas_fijas()

    # Activamos logs híbridos pasando el gestor de DB
    log = GestorLogs(gestor_db=db)
    log.registrar("MAIN", "Iniciando Pipeline de la Espiral 3 (Machine Learning)")

    try:
        # 1. Ingesta
        ingestor = IngestorDatos()
        datasets = ingestor.leer_datos()
        db.guardar_datos_raw(datasets)
        log.registrar("INF_1", "Carga y persistencia RAW completada")

        # 2. Preparación e Integración
        preparador = PreparadorDatos(datasets)
        df_master = preparador.ejecutar_preparacion()
        db.guardar_datos_master(df_master)
        log.registrar("INF_2", "Tabla maestra generada y persistida")

        # 3. Análisis descriptivo
        analista = Analizador()
        stats = analista.calcular_estadisticas_basicas(df_master)
        log.registrar("INF_3", f"Media calculada: {stats.get('media_aritmetica')}")

        # 4. Preparación de datos
        df_ml = preparador.preparar_dataset_ml(df_master)
        log.registrar("INF_4", "Dataset para ML preparado")

        # 5. Machine learning
        forzar_entrenamiento = config.machine_learning.get(
            "forzar_entrenamiento", False
        )

        # Verificamos si existe al menos el modelo del primer hito para decidir si entrenar
        ruta_modelo_b1 = os.path.join(
            os.path.dirname(__file__), "modelos", "arbol_b1.pkl"
        )

        if forzar_entrenamiento or not os.path.exists(ruta_modelo_b1):
            log.registrar(
                "INF_ML", "Iniciando fase de entrenamiento por puntos de control..."
            )
            entrenador = EntrenadorModelos()

            # Pasamos ambos parámetros: df_ml y el preparador
            df_comparativo = entrenador.entrenar_y_guardar(df_ml, preparador)

            log.registrar("INF_ML", "Modelos bimestrales entrenados y guardados.")

            # Mostrar tabla comparativa
            if not df_comparativo.empty:
                print("\n" + "=" * 65)
                print("      REPORTE COMPARATIVO DE PUNTOS DE CONTROL (HITOS)")
                print("=" * 65)
                print(df_comparativo.to_string(index=False))
                print("=" * 65 + "\n")
        else:
            log.registrar(
                "INF_ML",
                "Saltando entrenamiento. Se usarán los modelos por hitos existentes en disco.",
            )
        # 6. Cálculo de Riesgo (Inferencia mediante modelo guardado)
        evaluador = EvaluadorRiesgo()
        df_final = evaluador.ejecutar_evaluacion(df_master)
        log.registrar("INF_4", "Cálculo de riesgo mediante ML finalizado")

        # 7. Presentación
        vista = Visualizador()
        vista.mostrar_en_consola(df_final, stats)
        vista.exportar_csv(df_final)
        log.registrar("INF_5", "Resultados presentados y exportados")

    except Exception as e:
        log.registrar(
            "MAIN", f"Error crítico en la ejecución: {str(e)}", estado="ERROR"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
