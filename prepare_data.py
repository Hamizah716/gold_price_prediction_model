import pandas as pd
import numpy as np

RAW_PATH = r'Hamizah_sem4_extra files\monthly 2000-2025.csv'
OUTPUT_PATH = 'monthly_2000_2025_features.csv'

df = pd.read_csv(RAW_PATH, encoding='latin-1')
df = df.rename(columns={'Gold Price prices in USD per troy ounce': 'Price'})
df = df.drop(columns=['Events', 'Event Type', 'Event code', 'Year'])

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df = df.sort_values('Date').reset_index(drop=True)

df['Major_Event'] = 'Normal'
df['Event_Label'] = 0

event_ranges = {
    'Dot-com Crash':         ('2000-03', '2002-10', 1),
    '9/11 Attacks':          ('2001-09', '2002-03', 2),
    'Global Financial Crisis':('2008-09', '2009-12', 3),
    'European Debt Crisis':  ('2010-04', '2012-12', 4),
    'Oil Price Crash':       ('2014-06', '2015-12', 5),
    'COVID-19 Pandemic':     ('2019-11', '2021-12', 6),
    'Russia-Ukraine War':    ('2022-02', '2025-07', 7),
}

for name, (start, end, label) in event_ranges.items():
    mask = (df['Date'] >= pd.Timestamp(start)) & (df['Date'] <= pd.Timestamp(end))
    for idx in df[mask].index:
        if label > df.at[idx, 'Event_Label']:
            df.at[idx, 'Event_Label'] = label
            df.at[idx, 'Major_Event'] = name

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Price_Lag1'] = df['Price'].shift(1)
df['Price_Lag2'] = df['Price'].shift(2)
df['Price_Lag3'] = df['Price'].shift(3)

df = df.dropna().reset_index(drop=True)
df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')

cols = ['Date', 'Price', 'Year', 'Month', 'Month_Sin', 'Month_Cos',
        'Major_Event', 'Event_Label', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3']
df = df[cols]

df.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {OUTPUT_PATH} with {len(df)} rows")
print(f"Event distribution:\n{df['Major_Event'].value_counts()}")
print(f"Event_Label distribution:\n{df['Event_Label'].value_counts().sort_index()}")
