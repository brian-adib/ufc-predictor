import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. Cargando y preparando datos base...")
df = pd.read_csv('data/master.csv')
df = df[df['result_status'] == 'win'].copy()
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)
df = df[df['event_date'].dt.year >= 2015].copy()

print("2. Extrayendo Perfiles de Estilo del nuevo archivo...")
# Cargamos el archivo de scorecards
df_sc = pd.read_csv('data/merged_stats_n_scorecards.csv')

# Limpiamos los nombres para poder cruzarlos
df_sc['red_name'] = df_sc['red_fighter_name'].str.strip().str.lower()
df_sc['blue_name'] = df_sc['blue_fighter_name'].str.strip().str.lower()

# Calculamos el promedio de estilo para los peleadores cuando están en rojo
sc_red = df_sc.groupby('red_name')[['red_fighter_sig_str_head_pct', 'red_fighter_sig_str_distance_pct', 'red_fighter_sig_str_clinch_pct', 'red_fighter_sig_str_ground_pct']].mean()
sc_red.columns = ['head_pct', 'distance_pct', 'clinch_pct', 'ground_pct']

# Calculamos lo mismo cuando están en azul
sc_blue = df_sc.groupby('blue_name')[['blue_fighter_sig_str_head_pct', 'blue_fighter_sig_str_distance_pct', 'blue_fighter_sig_str_clinch_pct', 'blue_fighter_sig_str_ground_pct']].mean()
sc_blue.columns = ['head_pct', 'distance_pct', 'clinch_pct', 'ground_pct']

# Unimos ambos (el promedio total del peleador)
style_profiles = pd.concat([sc_red, sc_blue]).groupby(level=0).mean()

# Cruzamos con nuestro master.csv
df['r_name_clean'] = df['r_fighter_name'].str.strip().str.lower()
df['b_name_clean'] = df['b_fighter_name'].str.strip().str.lower()

for col in ['head_pct', 'distance_pct', 'clinch_pct', 'ground_pct']:
    df[f'r_{col}'] = df['r_name_clean'].map(style_profiles[col]).fillna(33.3) # Si no hay datos, asumimos 33%
    df[f'b_{col}'] = df['b_name_clean'].map(style_profiles[col]).fillna(33.3)

print("3. Procesando Round.csv (Cardio y Control)...")
df_round = pd.read_csv('data/round.csv')
r1_stats = pd.concat([df_round[df_round['round_no']==1].groupby('r_id')['r_sig_landed'].mean(), 
                      df_round[df_round['round_no']==1].groupby('b_id')['b_sig_landed'].mean()]).groupby(level=0).mean()
r3_stats = pd.concat([df_round[df_round['round_no']==3].groupby('r_id')['r_sig_landed'].mean(), 
                      df_round[df_round['round_no']==3].groupby('b_id')['b_sig_landed'].mean()]).groupby(level=0).mean()
cardio_dict = (r1_stats - r3_stats).fillna(0).to_dict()

def time_to_seconds(time_str):
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except: return 0
df_round['r_ctrl_secs'] = df_round['r_ctrl'].apply(time_to_seconds)
df_round['b_ctrl_secs'] = df_round['b_ctrl'].apply(time_to_seconds)
ctrl_dict = pd.concat([df_round.groupby('r_id')['r_ctrl_secs'].mean(), df_round.groupby('b_id')['b_ctrl_secs'].mean()]).groupby(level=0).mean().to_dict()

df['r_cardio_drop'] = df['r_fighter_id'].map(cardio_dict).fillna(0)
df['b_cardio_drop'] = df['b_fighter_id'].map(cardio_dict).fillna(0)
df['r_ctrl_secs'] = df['r_fighter_id'].map(ctrl_dict).fillna(0)
df['b_ctrl_secs'] = df['b_fighter_id'].map(ctrl_dict).fillna(0)

print("4. Calculando ELO, Inactividad, Rachas y Edad Pico...")
elo_ratings, ultima_pelea, racha_actual = {}, {}, {}
def get_elo(fid): return elo_ratings.get(fid, 1500)
def update_elo(w_id, l_id, k=32):
    r_w, r_l = get_elo(w_id), get_elo(l_id)
    exp_w = 1 / (1 + 10**((r_l - r_w) / 400))
    elo_ratings[w_id] = r_w + k * (1 - exp_w)
    elo_ratings[l_id] = r_l + k * (0 - exp_w)

def calc_age_peak(age):
    if pd.isna(age): return 0.5
    return np.exp(-((age - 31) ** 2) / (2 * 5.0 ** 2))

diff_elos, diff_days, diff_streaks = [], [], []
for _, row in df.iterrows():
    r_id, b_id, fecha = row['r_fighter_id'], row['b_fighter_id'], row['event_date']
    diff_elos.append(get_elo(r_id) - get_elo(b_id))
    d_r = (fecha - ultima_pelea[r_id]).days if r_id in ultima_pelea else 365
    d_b = (fecha - ultima_pelea[b_id]).days if b_id in ultima_pelea else 365
    diff_days.append(d_r - d_b)
    s_r, s_b = racha_actual.get(r_id, 0), racha_actual.get(b_id, 0)
    diff_streaks.append(s_r - s_b)
    
    gano_rojo = row['winner_id'] == r_id
    if gano_rojo:
        update_elo(r_id, b_id); racha_actual[r_id], racha_actual[b_id] = s_r+1, 0
    else:
        update_elo(b_id, r_id); racha_actual[b_id], racha_actual[r_id] = s_b+1, 0
    ultima_pelea[r_id], ultima_pelea[b_id] = fecha, fecha

df['diff_elo'] = diff_elos
df['diff_days_off'] = diff_days
df['diff_win_streak'] = diff_streaks

print("5. Creando TODAS las características...")
df = df.drop(columns=[col for col in df.columns if 'total_' in col])
df['target'] = (df['r_fighter_id'] == df['winner_id']).astype(int)

df['r_dob'] = pd.to_datetime(df['r_dob'])
df['b_dob'] = pd.to_datetime(df['b_dob'])
df['r_age'] = df['event_date'].dt.year - df['r_dob'].dt.year
df['b_age'] = df['event_date'].dt.year - df['b_dob'].dt.year
df['r_age_peak'] = df['r_age'].apply(calc_age_peak)
df['b_age_peak'] = df['b_age'].apply(calc_age_peak)

X = pd.DataFrame()
for f in ['slpm','str_acc','sapm','str_def','td_avg','td_acc','td_def','sub_avg','height','reach_inches']:
    if f'r_{f}' in df.columns and f'b_{f}' in df.columns:
        X[f'diff_{f}'] = pd.to_numeric(df[f'r_{f}'], errors='coerce') - pd.to_numeric(df[f'b_{f}'], errors='coerce')

X['diff_age'] = df['r_age'] - df['b_age']
X['diff_age_peak'] = df['r_age_peak'] - df['b_age_peak']
X['diff_ctrl_secs'] = df['r_ctrl_secs'] - df['b_ctrl_secs']
X['diff_striking_diff'] = (df['r_slpm'] - df['r_sapm']) - (df['b_slpm'] - df['b_sapm'])
X['diff_cardio'] = df['r_cardio_drop'] - df['b_cardio_drop']
X['is_title_fight'] = pd.to_numeric(df['title_fight'], errors='coerce').fillna(0)
X['diff_elo'] = df['diff_elo']
X['diff_days_off'] = df['diff_days_off']
X['diff_win_streak'] = df['diff_win_streak']
X['r_is_southpaw'] = (df['r_stance'] == 'Southpaw').astype(int)
X['b_is_southpaw'] = (df['b_stance'] == 'Southpaw').astype(int)
X['stance_advantage'] = X['r_is_southpaw'] - X['b_is_southpaw']

# NUEVAS VARIABLES DE ESTILO (Del archivo de Scorecards)
X['diff_head_pct'] = df['r_head_pct'] - df['b_head_pct']
X['diff_distance_pct'] = df['r_distance_pct'] - df['b_distance_pct']
X['diff_clinch_pct'] = df['r_clinch_pct'] - df['b_clinch_pct']
X['diff_ground_pct'] = df['r_ground_pct'] - df['b_ground_pct']

y = df['target']
X = X.fillna(0)

print("6. Entrenando VotingClassifier DEFINITIVO...")
corte = pd.to_datetime('2021-01-01')
train_idx = df['event_date'] < corte
test_idx = df['event_date'] >= corte
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

modelo_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.8, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=42)
modelo_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))

voting = VotingClassifier(estimators=[('xgb', modelo_xgb), ('rf', modelo_rf), ('lr', modelo_lr)], voting='soft')
voting.fit(X_train, y_train)

probs = voting.predict_proba(X_test)[:, 1]
predicciones = (probs >= 0.50).astype(int)
precision = accuracy_score(y_test, predicciones)
print(f"\n🎯 PRECISIÓN FINAL CON ESTILOS: {precision * 100:.2f}%")

importancias = pd.DataFrame({'Feature': X.columns, 'Importance': voting.named_estimators_['xgb'].feature_importances_})
print("\n💡 Top 12 variables más importantes:")
print(importancias.sort_values(by='Importance', ascending=False).head(12).to_string(index=False))

joblib.dump(voting, 'modelo_ufc.pkl')
print("\n✅ Modelo con Estilos guardado como 'modelo_ufc.pkl'")