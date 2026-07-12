import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import DataLoader
from src.models import ModelTrainer
from src.visualization import Visualizer

loader = DataLoader()
df = loader.load()
data_dict = loader.get_split_data()

trainer = ModelTrainer()
results = trainer.train(data_dict)

viz = Visualizer()
viz.plot_model_comparison(results, data_dict['y_test'], data_dict['test_dates'])

event_colors = {
    '9/11 Attacks': 'red', 'GFC': 'darkred', 'EU Debt Crisis': 'purple',
    'COVID-19': 'orange', 'Oil Crash': 'brown', 'Rus-Ukr War': 'magenta',
    'Dot-com Crash': 'darkblue'
}
viz.plot_gold_price_events(df, loader.get_event_dates(), event_colors)
viz.plot_metrics_table(results)

rf_model = trainer.trained_models['Random Forest'][0]
viz.plot_feature_importance(loader.feature_cols, rf_model.feature_importances_)

import pandas as pd
results_df = pd.DataFrame([
    {'Model': n, 'RMSE': r['RMSE'], 'MAE': r['MAE'], 'MAPE': f'{r["MAPE"]}%'}
    for n, r in results.items()
])
results_df.to_csv('output/metrics.csv', index=False)

with open('output/model_summary.json', 'w') as f:
    json.dump({n: {'RMSE': r['RMSE'], 'MAE': r['MAE'], 'MAPE': r['MAPE']}
               for n, r in results.items()}, f, indent=2)

print("=" * 60)
print("TRAINING COMPLETE - Results Summary")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nOutput files saved to 'output/' folder.")
