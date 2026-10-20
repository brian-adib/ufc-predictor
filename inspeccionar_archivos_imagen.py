import pandas as pd
import os

RUTA_DATA = 'data/'

archivos_a_inspeccionar = [
    'merged_stats_n_scorecards.csv',
    'stats_processed_all_bouts.csv'
]

for arch in archivos_a_inspeccionar:
    ruta_completa = os.path.join(RUTA_DATA, arch)
    
    if os.path.exists(ruta_completa):
        try:
            df = pd.read_csv(ruta_completa)
            print(f"\n{'='*50}")
            print(f"✅ ARCHIVO: {arch}")
            print(f"📊 Filas: {len(df)} | Columnas: {len(df.columns)}")
            print(f"📋 Columnas: {df.columns.tolist()}")
            print(f"👀 Primeras 2 filas:")
            print(df.head(2).to_string())
        except Exception as e:
            print(f"Error al leer {arch}: {e}")
    else:
        print(f"\n⚠️ No se encontró {arch}.")