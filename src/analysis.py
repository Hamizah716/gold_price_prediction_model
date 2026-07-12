import pandas as pd
import numpy as np

from src.data_loader import DataLoader

class EventAnalyzer:
    @staticmethod
    def get_event_stats(df):
        event_windows = DataLoader.get_event_windows()
        def classify_event(date):
            for event, (s, e) in event_windows.items():
                if pd.to_datetime(s) <= date <= pd.to_datetime(e):
                    return event
            return 'Normal'
        df_copy = df.copy()
        df_copy['Period'] = df_copy['Date'].apply(classify_event)
        df_copy['Monthly_Return'] = df_copy['Price'].pct_change() * 100
        stats = df_copy.groupby('Period').agg(
            mean=('Price', 'mean'),
            std=('Price', 'std'),
            min=('Price', 'min'),
            max=('Price', 'max'),
            count=('Price', 'count'),
            volatility=('Monthly_Return', lambda x: x.std())
        ).round(2)
        return stats.sort_values('volatility', ascending=False)

    @staticmethod
    def get_top_features(feature_importances, feature_cols, top_n=10):
        feat_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances})
        return feat_df.sort_values('importance', ascending=False).head(top_n)

    @staticmethod
    def compute_forecast(current_features, model_trainer, model_name, feature_cols, lr_features, steps=6):
        forecasts = []
        cf = current_features.copy()
        for i in range(steps):
            pred = model_trainer.predict(model_name, cf, feature_cols, lr_features)
            forecasts.append(pred)
            new_month = (cf['Month'] % 12) + 1
            cf.update({
                'Month': new_month,
                'Month_Sin': np.sin(2 * np.pi * new_month / 12),
                'Month_Cos': np.cos(2 * np.pi * new_month / 12),
                'Price_Lag3': cf.get('Price_Lag2', cf['Price_Lag1']),
                'Price_Lag2': cf.get('Price_Lag1', cf['Price_Lag1']),
                'Price_Lag1': pred,
            })
        return forecasts
