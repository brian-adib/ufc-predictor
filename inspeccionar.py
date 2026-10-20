import pandas as pd
df = pd.read_csv('data/master.csv')
df['event_date'] = pd.to_datetime(df['event_date'])
print(f"La pelea más reciente en tu base de datos es del: {df['event_date'].max()}")