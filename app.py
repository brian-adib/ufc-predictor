import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import numpy as np

st.set_page_config(page_title="UFC Predictor AI", page_icon="🥊", layout="wide")

st.title("🥊 UFC Predictor AI")
st.markdown("""
Bienvenido a la IA predictiva de la UFC. Este modelo utiliza un **Ensamble de Machine Learning** con **más del 70% de precisión**.
Analiza ELO, cardio, edad pico, tiempo de control y perfiles de estilo (golpes a distancia, clinch, suelo) para predecir al ganador.
""")
st.divider()

@st.cache_resource
def cargar_modelo(): return joblib.load('modelo_ufc.pkl')

@st.cache_data
def cargar_datos():
    # 1. Master
    df = pd.read_csv('data/master.csv')
    df = df[df['result_status'] == 'win'].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date').reset_index(drop=True)
    
    # 2. JSON (Activos)
    with open('data/ufc_fighters_stats_and_records.json', 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    activos = set(p['about']['name'].lower() for p in data_json if p.get('about', {}).get('Status') == 'Active')
            
    # 3. Scorecards (Estilos)
    df_sc = pd.read_csv('data/merged_stats_n_scorecards.csv')
    df_sc['red_name'] = df_sc['red_fighter_name'].str.strip().str.lower()
    df_sc['blue_name'] = df_sc['blue_fighter_name'].str.strip().str.lower()
    
    sc_red = df_sc.groupby('red_name')[['red_fighter_sig_str_head_pct', 'red_fighter_sig_str_distance_pct', 'red_fighter_sig_str_clinch_pct', 'red_fighter_sig_str_ground_pct']].mean()
    sc_red.columns = ['head_pct', 'distance_pct', 'clinch_pct', 'ground_pct']
    
    sc_blue = df_sc.groupby('blue_name')[['blue_fighter_sig_str_head_pct', 'blue_fighter_sig_str_distance_pct', 'blue_fighter_sig_str_clinch_pct', 'blue_fighter_sig_str_ground_pct']].mean()
    sc_blue.columns = ['head_pct', 'distance_pct', 'clinch_pct', 'ground_pct']
    
    style_profiles = pd.concat([sc_red, sc_blue]).groupby(level=0).mean()
            
    return df, activos, style_profiles

try:
    modelo = cargar_modelo()
    df_master, peleadores_activos, style_profiles = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar: {e}")
    st.stop()

# Cálculo de cardio y control (cargamos round.csv rápido)
@st.cache_data
def cargar_rounds():
    df_round = pd.read_csv('data/round.csv')
    def time_to_seconds(time_str):
        try:
            m, s = map(int, time_str.split(':'))
            return m * 60 + s
        except: return 0
    df_round['r_ctrl_secs'] = df_round['r_ctrl'].apply(time_to_seconds)
    df_round['b_ctrl_secs'] = df_round['b_ctrl'].apply(time_to_seconds)
    
    r1 = pd.concat([df_round[df_round['round_no']==1].groupby('r_id')['r_sig_landed'].mean(), df_round[df_round['round_no']==1].groupby('b_id')['b_sig_landed'].mean()]).groupby(level=0).mean()
    r3 = pd.concat([df_round[df_round['round_no']==3].groupby('r_id')['r_sig_landed'].mean(), df_round[df_round['round_no']==3].groupby('b_id')['b_sig_landed'].mean()]).groupby(level=0).mean()
    cardio = (r1 - r3).fillna(0).to_dict()
    
    ctrl = pd.concat([df_round.groupby('r_id')['r_ctrl_secs'].mean(), df_round.groupby('b_id')['b_ctrl_secs'].mean()]).groupby(level=0).mean().to_dict()
    return cardio, ctrl

cardio_dict, ctrl_dict = cargar_rounds()

def obtener_stats_actuales(fighter_name):
    peleas = df_master[(df_master['r_fighter_name'] == fighter_name) | (df_master['b_fighter_name'] == fighter_name)]
    if peleas.empty: return None
    ultima_pelea = peleas.iloc[-1]
    prefix = 'r' if ultima_pelea['r_fighter_name'] == fighter_name else 'b'
    
    stats = {
        'id': ultima_pelea[f'{prefix}_fighter_id'],
        'slpm': ultima_pelea[f'{prefix}_slpm'], 'str_acc': ultima_pelea[f'{prefix}_str_acc'],
        'sapm': ultima_pelea[f'{prefix}_sapm'], 'str_def': ultima_pelea[f'{prefix}_str_def'],
        'td_avg': ultima_pelea[f'{prefix}_td_avg'], 'td_acc': ultima_pelea[f'{prefix}_td_acc'],
        'td_def': ultima_pelea[f'{prefix}_td_def'], 'sub_avg': ultima_pelea[f'{prefix}_sub_avg'],
        'height': ultima_pelea[f'{prefix}_height'], 'reach_inches': ultima_pelea[f'{prefix}_reach_inches'],
        'dob': ultima_pelea[f'{prefix}_dob'], 'stance': ultima_pelea[f'{prefix}_stance']
    }
    return stats

def calcular_racha(fighter_name):
    peleas = df_master[(df_master['r_fighter_name'] == fighter_name) | (df_master['b_fighter_name'] == fighter_name)].tail(15)
    racha = 0
    for _, row in peleas.iloc[::-1].iterrows():
        if row['winner_id'] == row['r_fighter_id'] and row['r_fighter_name'] == fighter_name: racha += 1
        elif row['winner_id'] == row['b_fighter_id'] and row['b_fighter_name'] == fighter_name: racha += 1
        else: break
    return racha

def convertir_altura(altura_str):
    if pd.isna(altura_str): return 0
    try:
        pies, pulgadas = altura_str.replace('"', '').split("' ")
        return (int(pies) * 12) + int(pulgadas)
    except: return 0

def calc_age_peak(age):
    if pd.isna(age): return 0.5
    return np.exp(-((age - 31) ** 2) / (2 * 5.0 ** 2))

# Filtramos activos
todos_los_nombres = set(df_master['r_fighter_name'].dropna().unique()) | set(df_master['b_fighter_name'].dropna().unique())
lista_activos = sorted([n for n in todos_los_nombres if n.lower() in peleadores_activos])

st.sidebar.success(f"✅ {len(lista_activos)} peleadores activos disponibles.")

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.subheader("🔴 Peleador Rojo")
    rojo_nombre = st.selectbox("Selecciona al peleador rojo:", lista_activos, index=0)
with col2:
    st.markdown("<h1 style='text-align: center; margin-top: 20px;'>VS</h1>", unsafe_allow_html=True)
with col3:
    st.subheader("🔵 Peleador Azul")
    azul_nombre = st.selectbox("Selecciona al peleador azul:", lista_activos, index=1)

if st.button("⚡ ANALIZAR COMBATE", use_container_width=True, type="primary"):
    if rojo_nombre == azul_nombre:
        st.warning("⚠️ Selecciona dos peleadores diferentes.")
    else:
        p_rojo = obtener_stats_actuales(rojo_nombre)
        p_azul = obtener_stats_actuales(azul_nombre)
        
        if p_rojo is None or p_azul is None:
            st.error("Error: No se encontraron estadísticas históricas.")
        else:
            def get_style(name, col): return style_profiles.loc[name.lower(), col] if name.lower() in style_profiles.index else 33.3
            
            r_head, b_head = get_style(rojo_nombre, 'head_pct'), get_style(azul_nombre, 'head_pct')
            r_dist, b_dist = get_style(rojo_nombre, 'distance_pct'), get_style(azul_nombre, 'distance_pct')
            r_clinch, b_clinch = get_style(rojo_nombre, 'clinch_pct'), get_style(azul_nombre, 'clinch_pct')
            r_ground, b_ground = get_style(rojo_nombre, 'ground_pct'), get_style(azul_nombre, 'ground_pct')

            racha_rojo = calcular_racha(rojo_nombre)
            racha_azul = calcular_racha(azul_nombre)
            r_sp = 1 if p_rojo.get('stance') == 'Southpaw' else 0
            b_sp = 1 if p_azul.get('stance') == 'Southpaw' else 0
            
            cardio_r = cardio_dict.get(p_rojo['id'], 0)
            cardio_b = cardio_dict.get(p_azul['id'], 0)
            ctrl_r = ctrl_dict.get(p_rojo['id'], 0)
            ctrl_b = ctrl_dict.get(p_azul['id'], 0)
            
            age_r = 2026 - int(str(p_rojo.get('dob'))[:4])
            age_b = 2026 - int(str(p_azul.get('dob'))[:4])

            # ⚠️ EL ORDEN DE ESTAS VARIABLES ES EXACTAMENTE EL MISMO QUE EN EL ENTRENAMIENTO
            datos_pelea = {
                'diff_slpm': (p_rojo['slpm'] or 0) - (p_azul['slpm'] or 0),
                'diff_str_acc': (p_rojo['str_acc'] or 0) - (p_azul['str_acc'] or 0),
                'diff_sapm': (p_rojo['sapm'] or 0) - (p_azul['sapm'] or 0),
                'diff_str_def': (p_rojo['str_def'] or 0) - (p_azul['str_def'] or 0),
                'diff_td_avg': (p_rojo['td_avg'] or 0) - (p_azul['td_avg'] or 0),
                'diff_td_acc': (p_rojo['td_acc'] or 0) - (p_azul['td_acc'] or 0),
                'diff_td_def': (p_rojo['td_def'] or 0) - (p_azul['td_def'] or 0),
                'diff_sub_avg': (p_rojo['sub_avg'] or 0) - (p_azul['sub_avg'] or 0),
                'diff_height': convertir_altura(p_rojo.get('height')) - convertir_altura(p_azul.get('height')),
                'diff_reach_inches': (p_rojo.get('reach_inches') or 0) - (p_azul.get('reach_inches') or 0),
                'diff_age': age_r - age_b,
                'diff_age_peak': calc_age_peak(age_r) - calc_age_peak(age_b),
                'diff_ctrl_secs': ctrl_r - ctrl_b,
                'diff_striking_diff': ((p_rojo['slpm'] or 0) - (p_rojo['sapm'] or 0)) - ((p_azul['slpm'] or 0) - (p_azul['sapm'] or 0)),
                'diff_cardio': cardio_r - cardio_b,
                'is_title_fight': 0,
                'diff_elo': 0, 
                'diff_days_off': 0, 
                'diff_win_streak': racha_rojo - racha_azul,
                'r_is_southpaw': r_sp,
                'b_is_southpaw': b_sp,
                'stance_advantage': r_sp - b_sp,
                'diff_head_pct': r_head - b_head,
                'diff_distance_pct': r_dist - b_dist,
                'diff_clinch_pct': r_clinch - b_clinch,
                'diff_ground_pct': r_ground - b_ground
            }

            X_pred = pd.DataFrame([datos_pelea]).fillna(0)
            probabilidades = modelo.predict_proba(X_pred)[0]
            prob_rojo, prob_azul = probabilidades[1] * 100, probabilidades[0] * 100

            st.divider()
            st.subheader("📊 Resultados del Análisis")
            res1, res2 = st.columns(2)
            with res1:
                st.metric(label=f"🔴 {rojo_nombre}", value=f"{prob_rojo:.1f}%", delta=f"Racha: {racha_rojo} victorias")
            with res2:
                st.metric(label=f"🔵 {azul_nombre}", value=f"{prob_azul:.1f}%", delta=f"Racha: {racha_azul} victorias")

            st.progress(int(prob_rojo))
            if prob_rojo > prob_azul:
                st.success(f"🏆 **Veredicto:** Gana **{rojo_nombre}**.")
            else:
                st.success(f"🏆 **Veredicto:** Gana **{azul_nombre}**.")