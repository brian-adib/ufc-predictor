import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("1. Cargando y preparando datos...")
df = pd.read_csv('data/master.csv')
df = df[df['result_status'] == 'win'].copy()
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

print("2. Calculando ELO, Días de Inactividad y Rachas...")
elo_ratings = {}
ultima_pelea = {}
racha_actual = {}

def get_elo(fighter_id):
    return elo_ratings.get(fighter_id, 1500)

def update_elo(winner_id, loser_id, k=32):
    r_w = get_elo(winner_id)
    r_l = get_elo(loser_id)
    expected_w = 1 / (1 + 10**((r_l - r_w) / 400))
    elo_ratings[winner_id] = r_w + k * (1 - expected_w)
    elo_ratings[loser_id] = r_l + k * (0 - expected_w)

diff_elos, diff_days, diff_streaks = [], [], []

for index, row in df.iterrows():
    r_id, b_id = row['r_fighter_id'], row['b_fighter_id']
    fecha = row['event_date']
    
    # ELO
    r_e, b_e = get_elo(r_id), get_elo(b_id)
    diff_elos.append(r_e - b_e)
    
    # Días inactivos
    d_r = (fecha - ultima_pelea[r_id]).days if r_id in ultima_pelea else 365
    d_b = (fecha - ultima_pelea[b_id]).days if b_id in ultima_pelea else 365
    diff_days.append(d_r - d_b)
    
    # Rachas
    s_r, s_b = racha_actual.get(r_id, 0), racha_actual.get(b_id, 0)
    diff_streaks.append(s_r - s_b)
    
    # Actualizamos para la próxima pelea
    gano_rojo = row['winner_id'] == r_id
    if gano_rojo:
        update_elo(r_id, b_id)
        racha_actual[r_id], racha_actual[b_id] = s_r + 1, 0
    else:
        update_elo(b_id, r_id)
        racha_actual[b_id], racha_actual[r_id] = s_b + 1, 0
    ultima_pelea[r_id], ultima_pelea[b_id] = fecha, fecha

df['diff_elo'] = diff_elos
df['diff_days_off'] = diff_days
df['diff_win_streak'] = diff_streaks

print("3. Creando el resto de características...")
df = df.drop(columns=[col for col in df.columns if 'total_' in col])
df['target'] = (df['r_fighter_id'] == df['winner_id']).astype(int)

features = ['slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg', 'height', 'reach_inches']

df['r_dob'] = pd.to_datetime(df['r_dob'])
df['b_dob'] = pd.to_datetime(df['b_dob'])
df['r_age'] = df['event_date'].dt.year - df['r_dob'].dt.year
df['b_age'] = df['event_date'].dt.year - df['b_dob'].dt.year
features.append('age')

X = pd.DataFrame()
for f in features:
    if f == 'age':
        X['diff_age'] = df['r_age'] - df['b_age']
    else:
        if f'r_{f}' in df.columns and f'b_{f}' in df.columns:
            X[f'diff_{f}'] = pd.to_numeric(df[f'r_{f}'], errors='coerce') - pd.to_numeric(df[f'b_{f}'], errors='coerce')

X['diff_striking_diff'] = (df['r_slpm'] - df['r_sapm']) - (df['b_slpm'] - df['b_sapm'])
X['is_title_fight'] = pd.to_numeric(df['title_fight'], errors='coerce').fillna(0)
X['diff_elo'] = df['diff_elo']
X['diff_days_off'] = df['diff_days_off']
X['diff_win_streak'] = df['diff_win_streak']
X['r_is_southpaw'] = (df['r_stance'] == 'Southpaw').astype(int)
X['b_is_southpaw'] = (df['b_stance'] == 'Southpaw').astype(int)
X['stance_advantage'] = X['r_is_southpaw'] - X['b_is_southpaw']

y = df['target']
X = X.fillna(0)

print("4. Dividiendo datos y Entrenando el ENSAMBLE (XGBoost + Random Forest + Logistic)...")
corte = pd.to_datetime('2021-01-01')
train_idx = df['event_date'] < corte
test_idx = df['event_date'] >= corte

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 1. Modelo XGBoost
modelo_xgb = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.8, random_state=42
)

# 2. Modelo Random Forest (aprende agrupando árboles)
modelo_rf = RandomForestClassifier(
    n_estimators=300, max_depth=7, class_weight='balanced', random_state=42
)

# 3. Modelo de Regresión Logística (¡Necesita escalado de datos!)
modelo_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)

# CREAMOS EL ENSAMBLE (Votación suave)
ensamble = VotingClassifier(
    estimators=[('xgb', modelo_xgb), ('rf', modelo_rf), ('lr', modelo_lr)],
    voting='soft' # Usa las probabilidades para decidir, no solo los votos
)

# Entrenamos el ensamble
ensamble.fit(X_train, y_train)

predicciones = ensamble.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"\n🎯 PRECISIÓN DEL ENSAMBLE (EL MODELO DEFINITIVO): {precision * 100:.2f}%")

print("\nDetalle del rendimiento:")
print(classification_report(y_test, predicciones))

# Guardamos el ensamble
joblib.dump(ensamble, 'modelo_ufc.pkl')
print("\n✅ Ensamble guardado como 'modelo_ufc.pkl'. ¡Listo para la web!")