import os
import sys

# Aseguramos que Python encuentre los módulos dentro de 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Imports con los nombres de archivos y clases
from src.analizador_datos import Analizador
from src.evaluador_riesgo import EvaluadorRiesgo
from src.gestor_base_datos import GestorBaseDatos
from src.gestor_logs import GestorLogs
from src.ingestor_datos import IngestorDatos
from src.preparador_datos import PreparadorDatos
from src.visualizador import Visualizador


def main():
    # 0. Inicialización de infraestructura
    db = GestorBaseDatos()
    db.inicializar_tablas_fijas()

    # Activamos logs híbridos pasando el gestor de DB
    log = GestorLogs(gestor_db=db)
    log.registrar("MAIN", "Iniciando Pipeline de la Espiral 1")

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

        # 4. Cálculo de Riesgo
        evaluador = EvaluadorRiesgo()
        df_final = evaluador.ejecutar_evaluacion(df_master)
        log.registrar("INF_4", "Cálculo de riesgo finalizado")

        # 5. Presentación
        vista = Visualizador(output_dir="output")
        vista.mostrar_en_consola(df_final, stats)
        vista.exportar_csv(df_final)
        log.registrar("INF_5", "Resultados exportados a /output")

    except Exception as e:
        log.registrar("ERROR", f"Fallo en la ejecución: {str(e)}", estado="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
