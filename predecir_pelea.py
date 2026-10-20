import pandas as pd
import joblib

# 1. Cargamos el modelo predictivo que guardamos
print("Cargando modelo predictivo...")
try:
    modelo = joblib.load('modelo_ufc.pkl')
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'modelo_ufc.pkl'. Ejecuta primero 'entrenar_modelo.py'.")
    exit()

# 2. Cargamos las estadísticas de los peleadores
df_fighters = pd.read_csv('data/fighter.csv')

# Limpiamos los nombres para evitar problemas de mayúsculas/minúsculas
df_fighters['fighter_name_lower'] = df_fighters['fighter_name'].str.lower()

# CORRECCIÓN: Eliminamos nombres duplicados (ej. varios "Bruno Silva")
# Nos quedamos con el primero que aparezca en la lista
df_fighters = df_fighters.drop_duplicates(subset=['fighter_name_lower'], keep='first')

peleadores_dict = df_fighters.set_index('fighter_name_lower').to_dict('index')

def predecir_pelea(peleador_rojo, peleador_azul):
    print(f"\n🥊 Analizando: {peleador_rojo} (Rojo) vs {peleador_azul} (Azul)")
    
    p_rojo = peleadores_dict.get(peleador_rojo.lower())
    p_azul = peleadores_dict.get(peleador_azul.lower())
    
    # Validación por si el peleador no está en la base de datos
    if not p_rojo:
        print(f"❌ Error: No se encontró a '{peleador_rojo}' en la base de datos histórica.")
        return
    if not p_azul:
        print(f"❌ Error: No se encontró a '{peleador_azul}' en la base de datos histórica.")
        return

    # Función auxiliar para calcular edad aproximada
    def calcular_edad(dob):
        if pd.isna(dob): return 30
        return 2024 - int(str(dob)[:4])

    # Función auxiliar para convertir altura "5' 10\"" a pulgadas totales
    def convertir_altura(altura_str):
        if pd.isna(altura_str): return 0
        try:
            pies, pulgadas = altura_str.replace('"', '').split("' ")
            return (int(pies) * 12) + int(pulgadas)
        except:
            return 0

    edad_rojo = calcular_edad(p_rojo['dob'])
    edad_azul = calcular_edad(p_azul['dob'])
    
    altura_rojo = convertir_altura(p_rojo['height'])
    altura_azul = convertir_altura(p_azul['height'])

    # Creamos el diccionario con los datos
    datos_pelea = {
        'diff_slpm': (p_rojo['slpm'] or 0) - (p_azul['slpm'] or 0),
        'diff_str_acc': (p_rojo['str_acc'] or 0) - (p_azul['str_acc'] or 0),
        'diff_sapm': (p_rojo['sapm'] or 0) - (p_azul['sapm'] or 0),
        'diff_str_def': (p_rojo['str_def'] or 0) - (p_azul['str_def'] or 0),
        'diff_td_avg': (p_rojo['td_avg'] or 0) - (p_azul['td_avg'] or 0),
        'diff_td_acc': (p_rojo['td_acc'] or 0) - (p_azul['td_acc'] or 0),
        'diff_td_def': (p_rojo['td_def'] or 0) - (p_azul['td_def'] or 0),
        'diff_sub_avg': (p_rojo['sub_avg'] or 0) - (p_azul['sub_avg'] or 0),
        'diff_height': altura_rojo - altura_azul,
        'diff_reach_inches': (p_rojo['reach_inches'] or 0) - (p_azul['reach_inches'] or 0),
        'diff_age': edad_rojo - edad_azul,
        'is_title_fight': 0 # Asumimos que no es título para pruebas
    }

    X_pred = pd.DataFrame([datos_pelea]).fillna(0)

    # Predecimos
    probabilidades = modelo.predict_proba(X_pred)[0]
    prob_rojo = probabilidades[1] * 100
    prob_azul = probabilidades[0] * 100

    print("\n📊 RESULTADO DE LA PREDICCIÓN:")
    print(f"🔴 {peleador_rojo}: {prob_rojo:.2f}% de probabilidad de ganar")
    print(f"🔵 {peleador_azul}: {prob_azul:.2f}% de probabilidad de ganar")
    
    if prob_rojo > prob_azul:
        print(f"\n🏆 El modelo predice que gana {peleador_rojo}.")
    else:
        print(f"\n🏆 El modelo predice que gana {peleador_azul}.")

# --- ZONA DE PRUEBAS ---
# Prueba con peleadores que sepas que están en la base de datos
predecir_pelea("Jon Jones", "Stipe Miocic")
predecir_pelea("Alex Pereira", "Israel Adesanya")
predecir_pelea("Islam Makhachev", "Charles Oliveira")
predecir_pelea("Ilia Topuria", "Max Holloway")