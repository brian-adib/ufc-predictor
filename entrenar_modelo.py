import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("1. Cargando datos...")
df = pd.read_csv('data/master.csv')
df = df[df['result_status'] == 'win'].copy()

# Ordenamos por fecha para calcular las rachas correctamente
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

print("2. Calculando Rachas (Momentum) e Historial...")
# Creamos un diccionario para llevar la cuenta de las rachas
racha_actual = {}

def calcular_racha(fighter_id, gano):
    if fighter_id not in racha_actual:
        racha_actual[fighter_id] = 0
    
    # Si ganó, sumamos 1 a su racha. Si perdió, la reiniciamos a 0 (o -1 si prefieres)
    if gano:
        racha_actual[fighter_id] += 1
    else:
        racha_actual[fighter_id] = 0
    
    return racha_actual[fighter_id]

# Listas para guardar las rachas ANTES de la pelea
r_rachas = []
b_rachas = []

for index, row in df.iterrows():
    # Guardamos la racha con la que llegan a la pelea
    r_rachas.append(racha_actual.get(row['r_fighter_id'], 0))
    b_rachas.append(racha_actual.get(row['b_fighter_id'], 0))
    
    # Actualizamos las rachas para sus próximas peleas
    gano_rojo = row['winner_id'] == row['r_fighter_id']
    calcular_racha(row['r_fighter_id'], gano_rojo)
    calcular_racha(row['b_fighter_id'], not gano_rojo)

df['r_win_streak'] = r_rachas
df['b_win_streak'] = b_rachas

print("3. Creando características (Feature Engineering)...")
# ELIMINAMOS COLUMNAS DE FUGA
columnas_fuga = [col for col in df.columns if 'total_' in col]
df = df.drop(columns=columnas_fuga)

df['target'] = (df['r_fighter_id'] == df['winner_id']).astype(int)

features = [
    'slpm', 'str_acc', 'sapm', 'str_def', 
    'td_avg', 'td_acc', 'td_def', 'sub_avg', 
    'height', 'reach_inches'
]

df['r_dob'] = pd.to_datetime(df['r_dob'])
df['b_dob'] = pd.to_datetime(df['b_dob'])
df['r_age'] = df['event_date'].dt.year - df['r_dob'].dt.year
df['b_age'] = df['event_date'].dt.year - df['b_dob'].dt.year
features.append('age')

X = pd.DataFrame()

# 1. Diferencias de estadísticas
for f in features:
    if f == 'age':
        X['diff_age'] = df['r_age'] - df['b_age']
    else:
        if f'r_{f}' in df.columns and f'b_{f}' in df.columns:
            col_r = pd.to_numeric(df[f'r_{f}'], errors='coerce')
            col_b = pd.to_numeric(df[f'b_{f}'], errors='coerce')
            X[f'diff_{f}'] = col_r - col_b

# 2. Pelea de título y Rachas
X['is_title_fight'] = pd.to_numeric(df['title_fight'], errors='coerce').fillna(0)
X['diff_win_streak'] = df['r_win_streak'] - df['b_win_streak']

# 3. Estilos de pelea (Stance)
# Si el rojo es Southpaw (zurdo) y el azul es Orthodox (diestro), le damos un 1.
X['r_is_southpaw'] = (df['r_stance'] == 'Southpaw').astype(int)
X['b_is_southpaw'] = (df['b_stance'] == 'Southpaw').astype(int)
X['stance_advantage'] = X['r_is_southpaw'] - X['b_is_southpaw'] # Ventaja del zurdo

y = df['target']
X = X.fillna(0)

print("4. Dividiendo datos (Time-Series)...")
corte = pd.to_datetime('2021-01-01')
train_idx = df['event_date'] < corte
test_idx = df['event_date'] >= corte

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("5. Entrenando el modelo (XGBoost con nuevas variables)...")
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

modelo = xgb.XGBClassifier(
    n_estimators=300,       # Subimos un poco los árboles
    learning_rate=0.03,     # Hacemos que aprenda más despacio (mejor precisión)
    max_depth=5,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss'
)

modelo.fit(X_train, y_train)

print("6. Evaluando el modelo...")
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"\n🎯 NUEVA PRECISIÓN DEL MODELO: {precision * 100:.2f}%")
print("\nDetalle del rendimiento:")
print(classification_report(y_test, predicciones))

# Mostramos las variables más importantes
print("\n💡 Top 8 variables más importantes para el modelo:")
importancias = pd.DataFrame({'Feature': X.columns, 'Importance': modelo.feature_importances_})
print(importancias.sort_values(by='Importance', ascending=False).head(8).to_string(index=False))

# Guardamos el nuevo modelo
joblib.dump(modelo, 'modelo_ufc.pkl')
print("\n✅ Nuevo modelo mejorado guardado como 'modelo_ufc.pkl'")