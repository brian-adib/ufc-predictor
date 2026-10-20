import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

print("1. Cargando datos...")
df = pd.read_csv('data/master.csv')

# Nos quedamos solo con las peleas que terminaron en victoria
df = df[df['result_status'] == 'win'].copy()

# ELIMINAMOS LAS COLUMNAS DE FUGA DE DATOS
columnas_fuga = [col for col in df.columns if 'total_' in col]
df = df.drop(columns=columnas_fuga)

# Creamos nuestra variable objetivo (Target): 1 si gana el rojo, 0 si gana el azul
df['target'] = (df['r_fighter_id'] == df['winner_id']).astype(int)

print("2. Creando características (Feature Engineering)...")
features = [
    'slpm', 'str_acc', 'sapm', 'str_def', 
    'td_avg', 'td_acc', 'td_def', 'sub_avg', 
    'height', 'reach_inches'
]

# Calculamos la edad
df['event_date'] = pd.to_datetime(df['event_date'])
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

# 2. Añadimos si es pelea de título (1 o 0)
X['is_title_fight'] = pd.to_numeric(df['title_fight'], errors='coerce').fillna(0)

y = df['target']

# Rellenamos nulos
X = X.fillna(0)

print("3. Dividiendo datos (Time-Series)...")
corte = pd.to_datetime('2021-01-01')
train_idx = df['event_date'] < corte
test_idx = df['event_date'] >= corte

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Peleas para entrenar: {len(X_train)} | Peleas para probar: {len(X_test)}")

print("4. Entrenando el modelo (XGBoost)...")
# Calculamos el peso para balancear. Si hay más victorias rojas, esto lo compensa.
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# Configuramos XGBoost
modelo = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,             # Evita que el modelo se vuelva loco
    scale_pos_weight=ratio,  # Soluciona el problema de predecir siempre al rojo
    random_state=42,
    eval_metric='logloss'
)

modelo.fit(X_train, y_train)

print("5. Evaluando el modelo...")
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print(f"\n🎯 PRECISIÓN DEL MODELO: {precision * 100:.2f}%")
print("\nDetalle del rendimiento:")
print(classification_report(y_test, predicciones))

# Mostramos las 5 variables más importantes para el modelo
print("\n💡 Las 5 estadísticas más importantes para predecir al ganador:")
importancias = pd.DataFrame({'Feature': X.columns, 'Importance': modelo.feature_importances_})
print(importancias.sort_values(by='Importance', ascending=False).head(5).to_string(index=False))