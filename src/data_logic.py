import pandas as pd
import numpy as np
import os

class IngestorDatos:
    """Módulo 1: Carga y validación de datos."""
    def __init__(self, ruta_data=None):
        if ruta_data is None:
            # Sube un nivel desde 'src' y entra en 'data'
            self.ruta = os.path.join(os.path.dirname(__file__), '..', 'data')
        else:
            self.ruta = ruta_data
        self.datasets = {}

    def leer_datos(self):
        # Configuramos cada archivo de forma individual y limpia
        archivos = {
            'inscripciones': {
                'nombre': 'LSE_Inscrip_Baja_Recibido.csv', 
                'tipo': 'csv', 'sep': ';', 'enc': 'latin1'
            },
            'notas_bimestre': {
                'nombre': 'LSE_Notas_Estadistica_Bimestre.csv', 
                'tipo': 'csv', 'sep': ';', 'enc': 'latin1'
            },
            'actual': {
                'nombre': 'LSE_Notas_Inscrip_Baja_Actual.xlsx', 
                'tipo': 'excel'
            }
        }
        
        for clave, conf in archivos.items():
            path = os.path.join(self.ruta, conf['nombre'])
            
            if os.path.exists(path):
                print(f"Cargando {clave} desde {conf['nombre']}...")
                if conf['tipo'] == 'csv':
                    self.datasets[clave] = pd.read_csv(path, sep=conf['sep'], encoding=conf['enc'])
                elif conf['tipo'] == 'excel':
                    # Importante: Asegúrate de tener instalado 'openpyxl'
                    self.datasets[clave] = pd.read_excel(path)
            else:
                print(f"⚠️ Error: No se encuentra el archivo en {path}")
                print(f"   Por favor, verifica que el archivo esté en la carpeta 'data' con el nombre exacto.")
                
        return self.datasets

class PreparadorDatos:
    """Módulo 2: Integración y Limpieza."""
    def __init__(self, datasets_dict):
        self.datasets = datasets_dict

    def _normalizar(self, df, col):
        return df[col].astype(str).str.strip().str.upper()

    def ejecutar_preparacion(self):
        # 1. Base y Target (del archivo Excel)
        df_base = self.datasets['actual'].copy()
        
        # Normalizamos nombres de columnas por si acaso el Excel varía
        df_base.columns = [c.lower().strip() for c in df_base.columns]
        
        df_base['n_siu'] = self._normalizar(df_base, 'n_siu')
        df_base['estudio'] = self._normalizar(df_base, 'estudio')
        
        def mapear_riesgo(estado):
            estado = str(estado).upper()
            if 'RECIBIDO' in estado: return 0
            if 'CURSO' in estado or 'PAUSA' in estado: return 1
            if 'ABANDONO' in estado or 'LIBRE' in estado or 'BAJA' in estado: return 2
            return np.nan

        df_base['target'] = df_base['estado_actual'].apply(mapear_riesgo)

        # 2. Notas por Bimestre (del archivo CSV)
        df_notas = self.datasets['notas_bimestre'].copy()
        df_notas.rename(columns={'Estudio': 'estudio'}, inplace=True)
        df_notas['n_siu'] = self._normalizar(df_notas, 'n_siu')
        df_notas['estudio'] = self._normalizar(df_notas, 'estudio')
        
        df_notas['bimestre_n'] = df_notas['Comentario'].str.extract(r'(\d+)').fillna(0).astype(int)
        df_notas = df_notas[df_notas['bimestre_n'] > 0]

        df_pivot = df_notas.pivot_table(
            index=['n_siu', 'estudio'],
            columns='bimestre_n',
            values=['nota_m', 'asist_m'],
            aggfunc='first'
        )
        df_pivot.columns = [f"{c[0].split('_')[0]}_b{int(c[1])}" for c in df_pivot.columns]
        df_pivot.reset_index(inplace=True)

        # 3. Join Final
        tabla_maestra = pd.merge(df_base, df_pivot, on=['n_siu', 'estudio'], how='left')
        print(f"Tabla Maestra generada con {len(tabla_maestra)} registros.")
        return tabla_maestra