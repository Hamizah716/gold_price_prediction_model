import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
from src.data_loader import DataLoader

class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_comparison(self, results, y_test, test_dates):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        names = list(results.keys())

        for idx, name in enumerate(names):
            ax = axes[idx]
            ax.plot(pd.to_datetime(test_dates), y_test, 'b-', label='Actual', alpha=0.8, linewidth=1.5)
            ax.plot(pd.to_datetime(test_dates), results[name]['predictions'], '--',
                    label='Predicted', alpha=0.8, color=colors[idx], linewidth=1.5)
            ax.set_title(f'{name}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Gold Price (USD/oz)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('Actual vs Predicted Gold Prices', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_gold_price_events(self, df, event_dates, event_colors):
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(df['Date'], df['Price'], 'b-', linewidth=1.5, label='Gold Price (USD/oz)')

        for label, dt_str in event_dates.items():
            dt = pd.to_datetime(dt_str)
            clr = event_colors.get(label, 'red')
            ax.axvline(x=dt, color=clr, linestyle='--', alpha=0.6, linewidth=1.2)
            match = df[df['Date'] == dt]
            price_val = match['Price'].values[0] if len(match) > 0 else df['Price'].iloc[-1]
            ax.annotate(label, xy=(dt, price_val), xytext=(15, 15),
                        textcoords='offset points', fontsize=8, rotation=45,
                        alpha=0.8, color=clr, fontweight='bold')

        ax.set_title('Gold Price with Major Global Events (2000-2025)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Gold Price (USD/oz)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gold_price_events.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_metrics_table(self, results):
        _, ax = plt.subplots(figsize=(9, 4))
        ax.axis('tight')
        ax.axis('off')
        table_data = [[n, r['RMSE'], r['MAE'], f'{r["MAPE"]}%'] for n, r in results.items()]
        col_labels = ['Model', 'RMSE', 'MAE', 'MAPE (%)']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='center', loc='center', colWidths=[0.3, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor('#2B579A')
                cell.set_text_props(color='white', weight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/metrics_table.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, feature_cols, importances):
        feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feat_imp)), feat_imp['importance'].values, color='steelblue')
        plt.yticks(range(len(feat_imp)), feat_imp['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_event_volatility(self, df):
        event_windows = DataLoader.get_event_windows()
        def classify_event(date):
            for event, (s, e) in event_windows.items():
                if pd.to_datetime(s) <= date <= pd.to_datetime(e):
                    return event
            return 'Normal'
        df['Period'] = df['Date'].apply(classify_event)
        df['Monthly_Return'] = df['Price'].pct_change() * 100
        volatilities = df.groupby('Period')['Monthly_Return'].std().dropna().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#E81123', '#FF8C00', '#2B579A', '#7B2D8E', '#107C10', '#00B7C3', '#FFB900']
        ax.bar(volatilities.index, volatilities.values, color=colors[:len(volatilities)])
        ax.set_ylabel('Monthly Return Std Dev (%)')
        ax.set_title('Gold Price Volatility by Event Period')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/event_volatility.png', dpi=150, bbox_inches='tight')
        plt.close()
