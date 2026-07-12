import pandas as pd
import numpy as np

EVENT_RANGES = [
    ('Dot-com Crash', '2000-03', '2002-10', 1),
    ('9/11 Attacks', '2001-09', '2002-03', 2),
    ('Global Financial Crisis', '2008-09', '2009-12', 3),
    ('European Debt Crisis', '2010-04', '2012-12', 4),
    ('Oil Price Crash', '2014-06', '2015-12', 5),
    ('COVID-19 Pandemic', '2019-11', '2021-12', 6),
    ('Russia-Ukraine War', '2022-02', '2025-07', 7),
]

class DataLoader:
    def __init__(self, filepath='monthly_2000_2025_features.csv'):
        self.filepath = filepath
        self.df = None
        self.feature_cols = None
        self.lr_features = None
        self.event_dummies = []
        self.is_custom_data = False

    def load(self, raw_df=None):
        if raw_df is not None:
            df = self.process_raw_dataframe(raw_df)
            self.is_custom_data = True
        else:
            df = pd.read_csv(self.filepath)
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df = df.sort_values('Date').reset_index(drop=True)

        for ec in sorted(df['Event_Label'].unique()):
            if ec != 0:
                df[f'Event_{int(ec)}'] = (df['Event_Label'] == ec).astype(int)

        self.event_dummies = [c for c in df.columns if c.startswith('Event_')]
        self.feature_cols = ['Year', 'Month', 'Month_Sin', 'Month_Cos',
                             'Price_Lag1', 'Price_Lag2', 'Price_Lag3'] + self.event_dummies
        self.lr_features = ['Year', 'Month_Sin', 'Month_Cos', 'Price_Lag1'] + self.event_dummies
        self.df = df
        return df

    @staticmethod
    def process_raw_dataframe(raw_df):
        df = raw_df.copy()
        required = {'Date', 'Price'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}. Need Date and Price.")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df[['Date', 'Price']].copy()

        df['Major_Event'] = 'Normal'
        df['Event_Label'] = 0

        for name, start, end, label in EVENT_RANGES:
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
        return df

    def get_split_data(self, split_date='2021-01-01'):
        if self.is_custom_data:
            split_idx = int(len(self.df) * 0.8)
            split_date = self.df.iloc[split_idx]['Date']

        train_df = self.df[self.df['Date'] < split_date].copy()
        test_df = self.df[self.df['Date'] >= split_date].copy()

        if len(train_df) < 10 or len(test_df) < 5:
            split_idx = int(len(self.df) * 0.8)
            split_date = self.df.iloc[split_idx]['Date']
            train_df = self.df[self.df['Date'] < split_date].copy()
            test_df = self.df[self.df['Date'] >= split_date].copy()

        X_train = train_df[self.feature_cols].values
        y_train = train_df['Price'].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df['Price'].values

        lr_X_train = train_df[self.lr_features].values
        lr_X_test = test_df[self.lr_features].values

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'lr_X_train': lr_X_train, 'lr_X_test': lr_X_test,
            'test_dates': test_df['Date'].values,
            'train_df': train_df, 'test_df': test_df,
            'feature_names': self.feature_cols,
            'lr_feature_names': self.lr_features
        }

    def get_event_map(self):
        return {
            'Dot-com Crash': 'Dot-com Crash',
            '9/11 Attacks': '9/11 Attacks',
            'Global Financial Crisis': 'GFC',
            'European Debt Crisis': 'EU Debt Crisis',
            'Oil Price Crash': 'Oil Crash',
            'COVID-19 Pandemic': 'COVID-19',
            'Russia-Ukraine War': 'Rus-Ukr War'
        }

    def get_event_dates(self):
        return {
            'Dot-com Crash': '2000-03-01',
            '9/11 Attacks': '2001-09-01',
            'GFC': '2008-09-01',
            'EU Debt Crisis': '2010-04-01',
            'Oil Crash': '2014-06-01',
            'COVID-19': '2019-11-01',
            'Rus-Ukr War': '2022-02-01'
        }

    @staticmethod
    def get_event_windows():
        return {
            'Dot-com': ('2000-03-01', '2002-10-01'),
            '9/11': ('2001-09-01', '2002-03-01'),
            'GFC': ('2008-09-01', '2009-12-01'),
            'EU Debt': ('2010-04-01', '2012-12-01'),
            'Oil Crash': ('2014-06-01', '2015-12-01'),
            'COVID-19': ('2019-11-01', '2021-12-01'),
            'War': ('2022-02-01', '2025-07-01'),
        }
