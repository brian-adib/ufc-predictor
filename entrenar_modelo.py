import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("1. Cargando datos...")
# Cargamos el archivo maestro
df = pd.read_csv('data/master.csv')

# Nos quedamos solo con las peleas que terminaron en victoria (quitamos empates o 'No Contest')
df = df[df['result_status'] == 'win'].copy()

# ELIMINAMOS LAS COLUMNAS DE FUGA DE DATOS (Lo que pasó durante la pelea)
# Buscamos todas las columnas que contengan 'total_' y las borramos
columnas_fuga = [col for col in df.columns if 'total_' in col]
df = df.drop(columns=columnas_fuga)

# Creamos nuestra variable objetivo (Target): 
# 1 si gana el peleador rojo (r), 0 si gana el peleador azul (b)
df['target'] = (df['r_fighter_id'] == df['winner_id']).astype(int)

print("2. Creando características (Feature Engineering)...")
# Vamos a crear "Diferencias". En lugar de decirle al modelo los stats del rojo y del azul por separado,
# le damos la resta. Ej: (Golpes por minuto del rojo) - (Golpes por minuto del azul).
features = [
    'slpm',        # Golpes significativos conectados por minuto
    'str_acc',     # Precisión de golpes
    'sapm',        # Golpes significativos absorbidos por minuto
    'str_def',     # Defensa de golpes
    'td_avg',      # Derribos promedio
    'td_acc',      # Precisión de derribos
    'td_def',      # Defensa de derribos
    'sub_avg',     # Intentos de sumisión promedio
    'height',      # Altura
    'reach_inches',# Alcance
    'weight_lbs'   # Peso
]

# Calculamos la edad en el momento de la pelea (aproximada por el año)
df['event_date'] = pd.to_datetime(df['event_date'])
df['r_dob'] = pd.to_datetime(df['r_dob'])
df['b_dob'] = pd.to_datetime(df['b_dob'])
df['r_age'] = df['event_date'].dt.year - df['r_dob'].dt.year
df['b_age'] = df['event_date'].dt.year - df['b_dob'].dt.year
features.append('age')

# Creamos las columnas de diferencia
X = pd.DataFrame()
for f in features:
    if f == 'age':
        X['diff_age'] = df['r_age'] - df['b_age']
    else:
        # Si la columna existe en el dataset, hacemos la resta (r - b)
        if f'r_{f}' in df.columns and f'b_{f}' in df.columns:
            # Rellenamos los nulos con la media de esa columna para no perder datos
            col_r = pd.to_numeric(df[f'r_{f}'], errors='coerce').fillna(pd.to_numeric(df[f'r_{f}'], errors='coerce').mean())
            col_b = pd.to_numeric(df[f'b_{f}'], errors='coerce').fillna(pd.to_numeric(df[f'b_{f}'], errors='coerce').mean())
            X[f'diff_{f}'] = col_r - col_b

# Nuestra variable a predecir
y = df['target']

# Rellenamos cualquier valor vacío (NaN) que haya quedado con 0 para que el modelo no falle
X = X.fillna(0)

print("3. Dividiendo datos en Entrenamiento y Prueba (Time-Series)...")
# IMPORTANTE: No dividir al azar. Entrenamos con el pasado (hasta 2020) y probamos con el futuro (2021 en adelante)
corte = pd.to_datetime('2021-01-01')
train_idx = df['event_date'] < corte
test_idx = df['event_date'] >= corte

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Peleas para entrenar: {len(X_train)} | Peleas para probar: {len(X_test)}")

print("4. Entrenando el modelo (Random Forest)...")
# Usamos Random Forest, uno de los mejores para este tipo de datos
modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

print("5. Evaluando el modelo...")
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print(f"\n🎯 PRECISIÓN DEL MODELO: {precision * 100:.2f}%")
print("\nDetalle del rendimiento:")
print(classification_report(y_test, predicciones))