import pandas as pd
import os

# Rutas a los archivos que están dentro de la carpeta 'data'
archivos_a_revisar = [
    'data/master.csv', 
    'data/fight.csv', 
    'data/fighter.csv'
]

print("Iniciando la inspección de datos...\n")

for ruta in archivos_a_revisar:
    print(f"{'='*50}")
    print(f"REVISANDO EL ARCHIVO: {ruta}")
    print(f"{'='*50}")
    
    # Verificamos si el archivo existe
    if not os.path.exists(ruta):
        print(f"❌ ERROR: No se encontró el archivo en {ruta}.")
        print("Asegúrate de que el nombre esté bien escrito y esté dentro de la carpeta 'data'.\n")
        continue
        
    try:
        # Cargamos el archivo CSV
        df = pd.read_csv(ruta)
        
        # 1. Mostramos el tamaño (filas, columnas)
        print(f"✅ Archivo cargado correctamente.")
        print(f"📊 Tamaño: {df.shape[0]} filas y {df.shape[1]} columnas.")
        
        # 2. Mostramos los nombres de las columnas
        print(f"\n📋 Columnas:")
        for col in df.columns:
            print(f"  - {col}")
            
        # 3. Mostramos las primeras 2 filas para ver los datos reales
        print(f"\n👀 Primeras 2 filas de datos:")
        print(df.head(2).to_string()) # to_string() para que se vea ordenado
        print("\n")
        
    except Exception as e:
        print(f"❌ Ocurrió un error al leer el archivo: {e}\n")