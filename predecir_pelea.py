import pandas as pd
import joblib

print("Cargando modelo predictivo y datos...")
try:
    modelo = joblib.load('modelo_ufc.pkl')
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'modelo_ufc.pkl'. Ejecuta primero 'entrenar_modelo.py'.")
    exit()

# Cargamos estadísticas de peleadores
df_fighters = pd.read_csv('data/fighter.csv')
df_fighters['fighter_name_lower'] = df_fighters['fighter_name'].str.lower()
df_fighters = df_fighters.drop_duplicates(subset=['fighter_name_lower'], keep='first')
peleadores_dict = df_fighters.set_index('fighter_name_lower').to_dict('index')

# Cargamos master.csv solo para calcular las rachas actuales
df_master = pd.read_csv('data/master.csv')
df_master = df_master[df_master['result_status'] == 'win'].copy()
df_master['event_date'] = pd.to_datetime(df_master['event_date'])
df_master = df_master.sort_values('event_date')

def calcular_racha_actual(fighter_id):
    """Calcula cuántas victorias seguidas tiene un peleador en sus últimas peleas."""
    peleas = df_master[(df_master['r_fighter_id'] == fighter_id) | (df_master['b_fighter_id'] == fighter_id)].tail(15)
    racha = 0
    # Iteramos en orden inverso (de la más reciente a la más antigua)
    for _, row in peleas.iloc[::-1].iterrows():
        if row['winner_id'] == fighter_id:
            racha += 1
        else:
            break # Si pierde, la racha se corta
    return racha

def predecir_pelea(peleador_rojo, peleador_azul):
    print(f"\n🥊 Analizando: {peleador_rojo} (Rojo) vs {peleador_azul} (Azul)")
    
    p_rojo = peleadores_dict.get(peleador_rojo.lower())
    p_azul = peleadores_dict.get(peleador_azul.lower())
    
    if not p_rojo:
        print(f"❌ Error: No se encontró a '{peleador_rojo}'.")
        return
    if not p_azul:
        print(f"❌ Error: No se encontró a '{peleador_azul}'.")
        return

    def calcular_edad(dob):
        if pd.isna(dob): return 30
        return 2024 - int(str(dob)[:4])

    def convertir_altura(altura_str):
        if pd.isna(altura_str): return 0
        try:
            pies, pulgadas = altura_str.replace('"', '').split("' ")
            return (int(pies) * 12) + int(pulgadas)
        except:
            return 0

    # Calculamos rachas y posturas
    racha_rojo = calcular_racha_actual(p_rojo['fighter_id'])
    racha_azul = calcular_racha_actual(p_azul['fighter_id'])
    
    r_is_southpaw = 1 if p_rojo.get('stance') == 'Southpaw' else 0
    b_is_southpaw = 1 if p_azul.get('stance') == 'Southpaw' else 0
    stance_advantage = r_is_southpaw - b_is_southpaw

    # Creamos el vector EXACTO que espera el nuevo modelo
    datos_pelea = {
        'diff_slpm': (p_rojo['slpm'] or 0) - (p_azul['slpm'] or 0),
        'diff_str_acc': (p_rojo['str_acc'] or 0) - (p_azul['str_acc'] or 0),
        'diff_sapm': (p_rojo['sapm'] or 0) - (p_azul['sapm'] or 0),
        'diff_str_def': (p_rojo['str_def'] or 0) - (p_azul['str_def'] or 0),
        'diff_td_avg': (p_rojo['td_avg'] or 0) - (p_azul['td_avg'] or 0),
        'diff_td_acc': (p_rojo['td_acc'] or 0) - (p_azul['td_acc'] or 0),
        'diff_td_def': (p_rojo['td_def'] or 0) - (p_azul['td_def'] or 0),
        'diff_sub_avg': (p_rojo['sub_avg'] or 0) - (p_azul['sub_avg'] or 0),
        'diff_height': convertir_altura(p_rojo['height']) - convertir_altura(p_azul['height']),
        'diff_reach_inches': (p_rojo['reach_inches'] or 0) - (p_azul['reach_inches'] or 0),
        'diff_age': calcular_edad(p_rojo['dob']) - calcular_edad(p_azul['dob']),
        'is_title_fight': 0, # Ponemos 0 por defecto para pruebas
        'diff_win_streak': racha_rojo - racha_azul,
        'stance_advantage': stance_advantage
    }

    X_pred = pd.DataFrame([datos_pelea]).fillna(0)

    probabilidades = modelo.predict_proba(X_pred)[0]
    prob_rojo = probabilidades[1] * 100
    prob_azul = probabilidades[0] * 100

    print("\n📊 RESULTADO:")
    print(f"🔴 {peleador_rojo}: {prob_rojo:.2f}% (Racha actual: {racha_rojo} victorias)")
    print(f"🔵 {peleador_azul}: {prob_azul:.2f}% (Racha actual: {racha_azul} victorias)")
    
    if prob_rojo > prob_azul:
        print(f"🏆 El modelo predice que gana {peleador_rojo}.")
    else:
        print(f"🏆 El modelo predice que gana {peleador_azul}.")

# --- PRUEBAS ---
predecir_pelea("Jon Jones", "Stipe Miocic")
predecir_pelea("Alex Pereira", "Israel Adesanya")
predecir_pelea("Ilia Topuria", "Max Holloway")
predecir_pelea("Sean O'Malley", "Merab Dvalishvili")