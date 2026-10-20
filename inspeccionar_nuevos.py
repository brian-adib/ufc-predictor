import pandas as pd
import json
import os

# Rutas actualizadas apuntando a la carpeta 'data'
RUTA_ROUND = 'data/round.csv'
RUTA_JSON = 'data/ufc_fighters_stats_and_records.json'

# 1. INSPECCIONAR ROUND.CSV
print("="*50)
print("INSPECCIONANDO round.csv")
print("="*50)
try:
    df_round = pd.read_csv(RUTA_ROUND) 
    print(f"✅ Tamaño: {df_round.shape[0]} filas, {df_round.shape[1]} columnas")
    print("\nColumnas:")
    print(df_round.columns.tolist())
    print("\nPrimeras 3 filas:")
    print(df_round.head(3).to_string())
except Exception as e:
    print(f"❌ Error al leer round.csv: {e}")
    print(f"Asegúrate de que el archivo esté en: {RUTA_ROUND}")

# 2. INSPECCIONAR EL JSON
print("\n" + "="*50)
print("INSPECCIONANDO ufc_fighters_stats_and_records.json")
print("="*50)
try:
    with open(RUTA_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"✅ Archivo cargado correctamente. Tipo de dato raíz: {type(data)}")
    
    if isinstance(data, dict):
        primeras_llaves = list(data.keys())[:3]
        print(f"Primeras llaves encontradas: {primeras_llaves}")
        print(f"\nEjemplo de datos para la llave '{primeras_llaves[0]}':")
        print(json.dumps(data[primeras_llaves[0]], indent=2)[:600] + "...") 
        
    elif isinstance(data, list):
        print(f"El JSON contiene una lista de {len(data)} elementos.")
        print(f"\nEjemplo del primer elemento:")
        print(json.dumps(data[0], indent=2)[:600] + "...")
        
except Exception as e:
    print(f"❌ Error al leer el JSON: {e}")
    print(f"Asegúrate de que el archivo esté en: {RUTA_JSON}")