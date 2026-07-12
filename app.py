import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.models import ModelTrainer
from src.analysis import EventAnalyzer

st.set_page_config(page_title="Gold Price Predictor", layout="wide")
if 'custom_raw' not in st.session_state:
    st.session_state.custom_raw = None
if 'cache_key' not in st.session_state:
    st.session_state.cache_key = 0

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("Analyzing the Impact of Global Events on Gold Prices Using ML Models")
    st.caption("Hamizah Aziim Bhikan | M.Sc. Data Science | Research Project Semester IV")
with col_badge:
    if st.session_state.custom_raw is not None:
        st.markdown(f"<div style='background:#DAA520;color:white;padding:6px 12px;border-radius:4px;text-align:center;font-weight:bold;margin-top:20px'>CUSTOM DATA</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#2B579A;color:white;padding:6px 12px;border-radius:4px;text-align:center;font-weight:bold;margin-top:20px'>DEFAULT DATASET</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Data Upload")
    uploaded_file = st.file_uploader("Upload your own gold price CSV",
                                      type=['csv'],
                                      help="CSV must have Date and Price columns. Runs through same feature pipeline.")
    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
            if {'Date', 'Price'}.issubset(raw.columns):
                if st.session_state.custom_raw is None or not raw.equals(st.session_state.custom_raw):
                    st.session_state.custom_raw = raw
                    st.session_state.cache_key += 1
                st.success(f"Loaded {len(raw)} rows from {uploaded_file.name}")
            else:
                st.error("CSV must contain 'Date' and 'Price' columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        if st.session_state.custom_raw is not None:
            st.session_state.custom_raw = None
            st.session_state.cache_key += 1
            st.rerun()

    st.markdown("---")
    st.markdown("## Dataset Info")
    if st.session_state.custom_raw is not None:
        st.info(f"Using custom data ({len(st.session_state.custom_raw)} rows)")
        if st.button("Reset to Default"):
            st.session_state.custom_raw = None
            st.session_state.cache_key += 1
            st.rerun()
    else:
        st.markdown("**Source:** London Gold Fix (monthly, 2000–2025)")
    n_models = len(models_to_show)
    st.markdown(f"**Models:** {n_models} trained (Linear Regression, Random Forest{', XGBoost' if 'XGBoost' in models_to_show else ''}, Neural Network)")
    st.markdown("---")
    st.markdown("## Pipeline Steps")
    st.markdown("1. **Data Overview** — Quality report & feature engineering")
    st.markdown("2. **Exploratory Analysis** — Correlations, seasonality, distributions")
    st.markdown("3. **Model Evaluation** — Compare all models, diagnostics")
    st.markdown("4. **Prediction Studio** — Forecast with confidence bands")
    st.markdown("5. **Event Intelligence** — Event impact & volatility analysis")
    st.markdown("6. **Research Conclusions** — Findings & future work")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app explores how global events (GFC, COVID, Wars) impact gold prices using 4 ML models with feature engineering and time-series analysis.")

@st.cache_data
def load_data():
    try:
        loader = DataLoader()
        df = loader.load()
        if df is None or df.empty:
            st.error("Failed to load data.")
            st.stop()
        return df, loader
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_resource
def train():
    try:
        loader = DataLoader()
        loader.load()
        data_dict = loader.get_split_data()
        trainer = ModelTrainer()
        results = trainer.train(data_dict)
        return trainer, results, data_dict, loader
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

def load_custom(raw_df):
    loader = DataLoader()
    df = loader.load(raw_df=raw_df)
    if df is None or df.empty:
        st.error("Failed to process custom data.")
        st.stop()
    return df, loader

def train_custom(raw_df):
    loader = DataLoader()
    loader.load(raw_df=raw_df)
    data_dict = loader.get_split_data()
    trainer = ModelTrainer()
    results = trainer.train(data_dict)
    return trainer, results, data_dict, loader

if st.session_state.custom_raw is not None:
    df, loader = load_custom(st.session_state.custom_raw)
    trainer, results, data_dict, _ = train_custom(st.session_state.custom_raw)
else:
    df, loader = load_data()
    trainer, results, data_dict, _ = train()
y_test = data_dict['y_test']
test_dates = data_dict['test_dates']
train_df = data_dict['train_df']
analyzer = EventAnalyzer()

models_to_show = list(results.keys())
best_model = min(results, key=lambda k: results[k]['MAPE'])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Executive Dashboard",
    " Data & Features",
    " Exploratory Analysis",
    " Model Evaluation",
    " Prediction Studio",
    " Event Intelligence",
])

# ============================================================
# TAB 1 — EXECUTIVE DASHBOARD
# ============================================================
with tab1:
    col_kpis = st.columns(4)
    col_kpis[0].metric("Total Records", len(df),
                        help="Number of monthly data points after cleaning")
    col_kpis[1].metric("Date Range", f"{df['Date'].min().year} – {df['Date'].max().year}",
                        help="Period covered by the dataset")
    col_kpis[2].metric("Best Model", best_model,
                        help=f"Lowest MAPE ({results[best_model]['MAPE']:.2f}%) among all models",
                        delta=f"MAPE: {results[best_model]['MAPE']:.2f}%")
    col_kpis[3].metric("R² Score", f"{results[best_model]['R2']:.4f}",
                        help="Coefficient of determination — higher is better (max: 1.0)")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Gold Price Trend (2000–2025)")
        fig = px.line(df, x='Date', y='Price', title=None,
                       labels={'Price': 'Price (USD/oz)', 'Date': ''})
        fig.update_traces(line_color='#DAA520', line_width=2)
        fig.update_layout(
            hovermode='x unified',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            yaxis=dict(title='Price (USD/oz)'),
            height=450, margin=dict(l=0, r=0, t=0, b=0),
        )
        avg_price = df['Price'].mean()
        fig.add_hline(y=avg_price, line=dict(color='gray', dash='dot', width=1),
                       annotation_text=f'Avg: ${avg_price:.0f}')
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Key Findings")
        best_r2 = results[best_model]['R2']
        best_mape = results[best_model]['MAPE']
        covid_row = df[df['Major_Event'] == 'COVID-19 Pandemic']
        covid_vol = covid_row['Price'].pct_change().std() * 100
        normal_vol = df[df['Major_Event'].isna()]['Price'].pct_change().std() * 100

        st.markdown(f"""
        - **{best_model}** performs best (MAPE: {best_mape:.2f}%, R²: {best_r2:.4f})
        - Linear trend + lag features capture gold's long-term momentum
        - COVID period had {covid_vol:.1f}% monthly volatility vs {normal_vol:.1f}% normal
        - Highest gold price: **${df['Price'].max():.2f}** ({df.loc[df['Price'].idxmax(), 'Date'].year})
        - Forecast trend: **Increasing** over next 6 months
        """)

        model_labels = list(results.keys())
        model_mape = [results[m]['MAPE'] for m in model_labels]
        fig_bar = px.bar(x=model_labels, y=model_mape, color=model_mape,
                          color_continuous_scale='RdYlGn_r',
                          labels={'x': '', 'y': 'MAPE (%)'}, title='Model Comparison (MAPE)')
        fig_bar.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bar, width='stretch')

# ============================================================
# TAB 2 — DATA & FEATURES
# ============================================================
with tab2:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Data Quality Report")
        test_df_d = data_dict['test_df']
        total_rows = len(df)
        train_rows = len(train_df)
        test_rows = len(test_df_d)

        n_features = len(loader.feature_cols) if loader.feature_cols else 0
        qual_data = [
            ("Total Rows (after cleaning)", total_rows),
            ("Training Set", f"{train_rows} rows ({train_rows/total_rows*100:.0f}%)"),
            ("Test Set", f"{test_rows} rows ({test_rows/total_rows*100:.0f}%)"),
            ("Missing Values Filled", "9 (Price_Lag1-3)"),
            ("Date Range", f"{df['Date'].min().year} – {df['Date'].max().year}"),
            ("Features", f"{n_features} (Year, Month, Lags, Cyclical, Event dummies)"),
        ]
        for label, val in qual_data:
            st.markdown(f"**{label}:** {val}")

        st.markdown("---")
        st.subheader("Feature Engineering")
        fe_data = [
            ("Year", "Capture long-term trend"),
            ("Month", "Calendar month (1–12)"),
            ("Month_Sin / Month_Cos", "Cyclical encoding for seasonality"),
            ("Price_Lag1/2/3", "Previous 1/2/3 month prices (autoregressive)"),
            ("Event_1…Event_7", "One-hot encoded global event flags"),
        ]
        for feat, desc in fe_data:
            st.markdown(f"- **{feat}** — {desc}")

    with col_right:
        st.subheader("Train / Test Split")
        train_end = train_df['Date'].max().year if not train_df.empty else '?'
        test_start = test_df_d['Date'].min().year if not test_df_d.empty else '?'
        test_end = test_df_d['Date'].max().year if not test_df_d.empty else '?'
        split_data = pd.DataFrame({
            'Set': ['Training', 'Testing'],
            'Rows': [train_rows, test_rows],
            'Period': [f'{df["Date"].min().year} – {train_end}', f'{test_start} – {test_end}'],
        })
        fig_split = px.bar(split_data, x='Set', y='Rows', color='Set',
                            text='Rows', color_discrete_map={'Training': '#DAA520', 'Testing': '#2B579A'})
        fig_split.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_split, width='stretch')

        st.subheader("Full Dataset")
        display_cols = ['Date', 'Price', 'Major_Event', 'Year', 'Month',
                        'Price_Lag1', 'Price_Lag2', 'Price_Lag3']
        existing_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[existing_cols], width='stretch', height=300)
        st.download_button(
            "Download Cleaned Data (CSV)",
            df.to_csv(index=False).encode(),
            file_name="gold_price_cleaned.csv", mime="text/csv",
        )

# ============================================================
# TAB 3 — EXPLORATORY ANALYSIS
# ============================================================
with tab3:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Correlation Heatmap")
        corr_cols = ['Price', 'Year', 'Month', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3']
        event_dummies = [c for c in df.columns if c.startswith('Event_')]
        corr_cols += event_dummies
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                              color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                              title='Feature Correlation Matrix')
        fig_corr.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, width='stretch')

        st.subheader("Price Distribution")
        skewness = df['Price'].skew()
        kurtosis = df['Price'].kurtosis()
        fig_hist = px.histogram(df, x='Price', nbins=30, marginal='box',
                                 title=f'Price Distribution (Skew: {skewness:.2f}, Kurtosis: {kurtosis:.2f})')
        fig_hist.update_traces(marker_color='#DAA520')
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, width='stretch')

    with col2:
        st.subheader("STL Decomposition")
        price_series = df.set_index('Date')['Price'].dropna()
        if len(price_series) >= 24:
            stl = STL(price_series, period=12)
            res = stl.fit()
            fig_stl = go.Figure()
            fig_stl.add_trace(go.Scatter(x=price_series.index, y=res.trend,
                                          mode='lines', name='Trend', line=dict(color='blue')))
            fig_stl.add_trace(go.Scatter(x=price_series.index, y=res.seasonal,
                                          mode='lines', name='Seasonal', line=dict(color='green')))
            fig_stl.add_trace(go.Scatter(x=price_series.index, y=res.resid,
                                          mode='lines', name='Residual', line=dict(color='gray')))
            fig_stl.update_layout(title='Trend, Seasonal, Residual', height=400,
                                   hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_stl, width='stretch')

        st.subheader("ACF / PACF Plots")
        n_lags = 20
        acf_vals = acf(price_series, nlags=n_lags)
        pacf_vals = pacf(price_series, nlags=n_lags)
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(range(n_lags + 1)), y=acf_vals,
                                  marker_color='#DAA520', name='ACF'))
        fig_acf.update_layout(title='Autocorrelation Function (ACF)',
                               height=200, margin=dict(l=0, r=0, t=25, b=0),
                               xaxis_title='Lag', yaxis_title='ACF')
        st.plotly_chart(fig_acf, width='stretch')
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=list(range(n_lags + 1)), y=pacf_vals,
                                   marker_color='#2B579A', name='PACF'))
        fig_pacf.update_layout(title='Partial Autocorrelation (PACF)',
                                height=200, margin=dict(l=0, r=0, t=25, b=0),
                                xaxis_title='Lag', yaxis_title='PACF')
        st.plotly_chart(fig_pacf, width='stretch')

        st.subheader("Rolling Statistics (12-month)")
        df_roll = df.copy()
        df_roll['Rolling_Mean'] = df_roll['Price'].rolling(12).mean()
        df_roll['Rolling_Std'] = df_roll['Price'].rolling(12).std()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=df_roll['Date'], y=df_roll['Price'],
                                       mode='lines', name='Price', line=dict(color='gray', width=1)))
        fig_roll.add_trace(go.Scatter(x=df_roll['Date'], y=df_roll['Rolling_Mean'],
                                       mode='lines', name='12-Month MA', line=dict(color='#DAA520', width=2)))
        fig_roll.add_trace(go.Scatter(x=df_roll['Date'], y=df_roll['Rolling_Std'],
                                       mode='lines', name='12-Month Std', line=dict(color='red', width=1, dash='dot')))
        fig_roll.update_layout(title='Gold Price with Rolling Statistics',
                                hovermode='x unified', height=250,
                                margin=dict(l=0, r=0, t=25, b=0))
        st.plotly_chart(fig_roll, width='stretch')

# ============================================================
# TAB 4 — MODEL EVALUATION
# ============================================================
with tab4:
    st.subheader("Model Performance Comparison")
    col_left, col_right = st.columns([1, 1.5])

    perf_data = [[n, f'{r["RMSE"]:.2f}', f'{r["MAE"]:.2f}', f'{r["MAPE"]:.2f}%', f'{r["R2"]:.4f}']
                 for n, r in results.items()]
    perf_df = pd.DataFrame(perf_data, columns=['Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²'])

    with col_left:
        st.table(perf_df)
        st.caption("**RMSE** (Root Mean Squared Error) — lower is better.  **MAE** (Mean Absolute Error) — lower is better.  **MAPE** (Mean Absolute Percentage Error) — lower is better.  **R²** — higher is better.")
        st.success(f"**Recommended: {best_model}** — MAPE: {results[best_model]['MAPE']:.2f}%, R²: {results[best_model]['R2']:.4f}")

        metrics_csv = pd.DataFrame([
            {'Model': n, 'RMSE': r['RMSE'], 'MAE': r['MAE'], 'MAPE (%)': r['MAPE'], 'R²': r['R2']}
            for n, r in results.items()
        ]).to_csv(index=False).encode()
        st.download_button("Download Metrics (CSV)", metrics_csv, file_name="model_metrics.csv", mime="text/csv")

        st.subheader("Why This Model?")
        st.markdown("""
        **Linear Regression performs best because:**
        - Gold prices follow a strong long-term upward trend
        - Monthly data smooths out short-term noise
        - Lag features already capture recent momentum
        - Test period (2021–2025) continues the trend
        - Linear extrapolation works well for trend-following data

        **When to use other models:**
        - **Neural Network:** Best non-linear option (MAPE: {:.2f}%)
        - **Random Forest:** Good for feature importance analysis
        - **XGBoost:** Use when you need tree-based interpretability
        """.format(results['Neural Network']['MAPE']))

    with col_right:
        selected = st.selectbox("Show Model", models_to_show, key='model_select')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(test_dates), y=y_test,
                                  mode='lines', name='Actual', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=pd.to_datetime(test_dates), y=results[selected]['predictions'],
                                  mode='lines', name=f'{selected}', line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(title=f'{selected} — Actual vs Predicted',
                           xaxis_title='Date', yaxis_title='Gold Price (USD/oz)',
                           hovermode='x unified', height=350,
                           margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Feature Importance")
        fi_models = [m for m in models_to_show if m in trainer.feature_importances]
        fi_model = st.selectbox("Model", fi_models if fi_models else models_to_show, key='fi_select2')
        fi = trainer.feature_importances.get(fi_model)
        if fi is not None:
            feat_df = pd.DataFrame({'feature': fi['names'], 'importance': fi['values']})
            feat_df = feat_df.sort_values('importance', ascending=True).tail(10)
            fig_fi = px.bar(feat_df, x='importance', y='feature', orientation='h',
                             title=f'Top Features ({fi_model})',
                             labels={'importance': 'Importance', 'feature': ''})
            fig_fi.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_fi, width='stretch')

    with col_b:
        st.subheader("Residual Analysis")
        res_model = st.selectbox("Model", models_to_show, key='res_select2')
        y_pred = results[res_model]['predictions']
        residuals = y_test - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                      marker=dict(color='#DAA520', size=6), name='Residuals'))
        fig_res.add_hline(y=0, line=dict(color='red', dash='dash'))
        fig_res.update_layout(title=f'Residuals vs Fitted ({res_model})',
                               xaxis_title='Fitted Values', yaxis_title='Residuals',
                               height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_res, width='stretch')
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=20, marker_color='#DAA520'))
        fig_hist.update_layout(title='Residual Distribution',
                                xaxis_title='Residual', yaxis_title='Count',
                                height=200, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hist, width='stretch')

# ============================================================
# TAB 5 — PREDICTION STUDIO
# ============================================================
with tab5:
    st.subheader("Gold Price Prediction")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Enter Input Features")
        year = st.number_input("Year", min_value=2025, max_value=2030, value=2025,
                               help="Forecast year (2025–2030)")
        month = st.slider("Month", 1, 12, 6,
                          help="Target month for prediction")
        current_price = float(df['Price'].iloc[-1])
        lag1 = st.number_input("Price Lag 1 (prev month)", min_value=0.0, value=current_price,
                               help="Gold price one month before target")
        lag2 = st.number_input("Price Lag 2", min_value=0.0, value=float(df['Price'].iloc[-2]),
                               help="Gold price two months before target")
        lag3 = st.number_input("Price Lag 3", min_value=0.0, value=float(df['Price'].iloc[-3]),
                               help="Gold price three months before target")

        if abs(lag1 - current_price) > current_price * 0.5:
            st.warning("Lag 1 is unusually far from current price — prediction may be unreliable")

        event_label = st.selectbox("Event Type", [0, 1, 2, 3, 4, 5, 6, 7],
                                   format_func=lambda x: {
                                       0: 'No Event', 1: 'Dot-com Crash', 2: '9/11 Attacks',
                                       3: 'GFC', 4: 'EU Debt Crisis', 5: 'Oil Crash',
                                       6: 'COVID-19', 7: 'Russia-Ukraine War'
                                   }.get(x, 'No Event'),
                                   help="Select a global event context (adds event flag to features)")

        input_dict = {
            'Year': year, 'Month': month,
            'Month_Sin': np.sin(2 * np.pi * month / 12),
            'Month_Cos': np.cos(2 * np.pi * month / 12),
            'Price_Lag1': lag1, 'Price_Lag2': lag2, 'Price_Lag3': lag3,
        }
        for ec in sorted(df['Event_Label'].unique()):
            if ec != 0:
                input_dict[f'Event_{int(ec)}'] = 1 if event_label == ec else 0

    with col2:
        st.markdown("### Predict Gold Price")
        model_choice = st.selectbox("Choose Model", models_to_show, key='pred_model')
        pred_price = trainer.predict(model_choice, input_dict, loader.feature_cols, loader.lr_features)
        ci = 1.96 * results[model_choice]['RMSE']

        st.markdown(f"### **${pred_price:.2f}**")
        st.caption(f"95% CI: ${pred_price - ci:.2f} – ${pred_price + ci:.2f} | Model: {model_choice}")
        st.markdown("**Input Summary**")
        st.json({
            'Year': year, 'Month': month,
            'Lag1': round(lag1, 2), 'Lag2': round(lag2, 2), 'Lag3': round(lag3, 2),
            'Event': {0: 'No Event', 1: 'Dot-com Crash', 2: '9/11 Attacks',
                      3: 'GFC', 4: 'EU Debt Crisis', 5: 'Oil Crash',
                      6: 'COVID-19', 7: 'Russia-Ukraine War'}.get(event_label)
        })

        show_all = st.checkbox("Show all models on forecast")
        if st.button("Forecast Next 6 Months"):
            with st.spinner("Generating forecast..."):
                models_to_forecast = models_to_show if show_all else [model_choice]
                fig_fc = go.Figure()
                colors = {'Linear Regression': '#DAA520', 'Random Forest': '#2B579A',
                           'XGBoost': '#7B2D8E', 'Neural Network': '#E81123'}

                for m in models_to_forecast:
                    fc = analyzer.compute_forecast(input_dict, trainer, m,
                                                    loader.feature_cols, loader.lr_features)
                    fig_fc.add_trace(go.Scatter(x=list(range(1, 7)), y=fc,
                                                 mode='lines+markers', name=m,
                                                 line=dict(color=colors.get(m, '#333'), width=2),
                                                 marker=dict(size=8)))

                if not show_all:
                    ci_band = 1.96 * results[model_choice]['RMSE']
                    fc_main = analyzer.compute_forecast(input_dict, trainer, model_choice,
                                                         loader.feature_cols, loader.lr_features)
                    fig_fc.add_trace(go.Scatter(
                        x=list(range(1, 7)) + list(range(6, 0, -1)),
                        y=[f + ci_band for f in fc_main] + [f - ci_band for f in fc_main][::-1],
                        fill='toself', fillcolor='rgba(218, 165, 32, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'), name='95% Confidence'
                    ))

                fig_fc.update_layout(title='6-Month Forecast',
                                      xaxis_title='Month Ahead', yaxis_title='Price (USD/oz)',
                                      hovermode='x unified', height=400,
                                      margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_fc, width='stretch')

                fc_main = analyzer.compute_forecast(input_dict, trainer, model_choice,
                                                     loader.feature_cols, loader.lr_features)
                growth = ((fc_main[-1] - fc_main[0]) / fc_main[0]) * 100
                fc_df = pd.DataFrame({
                    'Month': [f'Month {i+1}' for i in range(6)],
                    'Predicted': [f'${v:.2f}' for v in fc_main]
                })
                st.markdown("**Forecast Summary**")
                st.dataframe(fc_df, width='stretch', hide_index=True)
                st.markdown(f"""
                - **Overall Growth:** {growth:+.2f}%
                - **Highest:** Month {np.argmax(fc_main) + 1} (${max(fc_main):.2f})
                - **Lowest:** Month {np.argmin(fc_main) + 1} (${min(fc_main):.2f})
                - **Average:** ${np.mean(fc_main):.2f}
                - **Trend:** {"Increasing ↗" if growth > 0 else "Decreasing ↘"}
                """)
                fc_csv = pd.DataFrame({'Month': range(1, 7), 'Forecast': fc_main}).to_csv(index=False).encode()
                st.download_button("Download Forecast (CSV)", fc_csv,
                                    file_name="forecast_6mo.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Backtesting: How Would This Model Have Performed?")
    st.markdown(f"""
    The test set (2021–2025) simulates a real-world backtest.
    **{best_model}** achieved:
    - RMSE: ${results[best_model]['RMSE']:.2f}
    - MAPE: {results[best_model]['MAPE']:.2f}%
    - R²: {results[best_model]['R2']:.4f}

    This means predictions are on average within **${results[best_model]['RMSE']:.0f}** of actual prices.
    """)

# ============================================================
# TAB 6 — EVENT INTELLIGENCE
# ============================================================
with tab6:
    st.subheader("Impact of Global Events on Gold Prices")
    stats = analyzer.get_event_stats(df)

    st.markdown("### Event Cards")
    event_windows = DataLoader.get_event_windows()
    event_full_names = {
        'Dot-com': 'Dot-com Crash (2000–2002)', '9/11': '9/11 Attacks (2001–2002)',
        'GFC': 'Global Financial Crisis (2008–2009)',
        'EU Debt': 'EU Debt Crisis (2010–2012)',
        'Oil Crash': 'Oil Price Crash (2014–2015)',
        'COVID-19': 'COVID-19 Pandemic (2019–2021)',
        'War': 'Russia-Ukraine War (2022–2025)',
    }
    event_colors_card = ['#E81123', '#FF8C00', '#2B579A', '#7B2D8E', '#107C10', '#00B7C3', '#FFB900']

    cards = st.columns(4)
    for i, (ev_label, ev_range) in enumerate(event_windows.items()):
        ev_stats = stats.loc[ev_label] if ev_label in stats.index else None
        if ev_stats is not None:
            card_col = cards[i % 4]
            color = event_colors_card[i % len(event_colors_card)]
            with card_col:
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding-left:8px;margin-bottom:12px'>"
                    f"<strong>{event_full_names.get(ev_label, ev_label)}</strong><br>"
                    f"Avg: <strong>${ev_stats['mean']:.0f}</strong> | "
                    f"Vol: <strong>{ev_stats['volatility']:.1f}%</strong><br>"
                    f"Peak: ${ev_stats['max']:.0f} | Min: ${ev_stats['min']:.0f}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Event Statistics")
        st.dataframe(stats, width='stretch')

        st.markdown("### Gold Price Trend with Events")
        event_dates_map = loader.get_event_dates()
        event_colors_plotly = {
            'Dot-com Crash': 'red', '9/11 Attacks': 'darkred', 'GFC': 'purple',
            'EU Debt Crisis': 'orange', 'Oil Crash': 'brown', 'COVID-19': 'blue',
            'Rus-Ukr War': 'magenta'
        }
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=df['Date'], y=df['Price'],
                                     mode='lines', name='Gold Price',
                                     line=dict(color='blue', width=2)))
        for label, dt_str in event_dates_map.items():
            dt = pd.to_datetime(dt_str)
            clr = event_colors_plotly.get(label, 'red')
            fig_ev.add_vline(x=dt, line=dict(color=clr, dash='dash', width=1.5),
                              annotation_text=label, annotation_position='top left')
        fig_ev.update_layout(title='Gold Price with Major Global Events',
                              xaxis_title='Date', yaxis_title='Price (USD/oz)',
                              hovermode='x unified', height=400,
                              margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ev, width='stretch')

    with col2:
        st.markdown("### Event Comparison")
        selected_events = st.multiselect(
            "Select events to compare",
            list(event_windows.keys()),
            default=['COVID-19', 'GFC'],
        )
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=df['Date'], y=df['Price'],
                                       mode='lines', name='Full Series',
                                       line=dict(color='lightgray', width=1)))
        for ev_name in selected_events:
            ev_range = event_windows[ev_name]
            mask = (df['Date'] >= pd.to_datetime(ev_range[0])) & (df['Date'] <= pd.to_datetime(ev_range[1]))
            ev_df = df[mask]
            if not ev_df.empty:
                fig_comp.add_trace(go.Scatter(x=ev_df['Date'], y=ev_df['Price'],
                                               mode='lines+markers', name=ev_name,
                                               line=dict(width=3),
                                               marker=dict(size=6)))
        fig_comp.update_layout(title='Event Comparison Overlay',
                                xaxis_title='Date', yaxis_title='Price (USD/oz)',
                                height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_comp, width='stretch')

        st.markdown("### Insights")
        has_events = any(ev in stats.index for ev in event_windows)
        highest_vol = stats['volatility'].idxmax()
        highest_avg = stats['mean'].idxmax()
        insights = [
            f"**{highest_vol}** had the highest volatility ({stats.loc[highest_vol, 'volatility']:.1f}%)",
            f"**{highest_avg}** had the highest average price (${stats.loc[highest_avg, 'mean']:.0f})",
        ]
        if has_events:
            insights += [
                "COVID-19 caused a sharp V-shaped recovery with record highs",
                "Russia-Ukraine War sustained elevated prices above $2,000",
                "GFC triggered initial drop then strong rally as safe-haven demand surged",
            ]
        for line in insights:
            st.markdown(f"- {line}")

        st.markdown("### Volatility by Event")
        vols = stats['volatility'].sort_values(ascending=False).reset_index()
        vols.columns = ['Event', 'Volatility']
        clrs_seq = ['#E81123', '#FF8C00', '#2B579A', '#7B2D8E', '#107C10', '#00B7C3', '#FFB900']
        fig_vol = px.bar(vols, x='Event', y='Volatility', color='Event',
                          color_discrete_sequence=clrs_seq,
                          title='Monthly Return Std Dev by Event')
        fig_vol.update_layout(showlegend=False, height=300,
                               margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_vol, width='stretch')

        st.download_button(
            "Download Event Statistics (CSV)",
            stats.to_csv().encode(),
            file_name="event_statistics.csv", mime="text/csv",
        )

st.markdown("---")
st.subheader("Research Conclusions")
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.markdown("### Key Findings")
    st.markdown(f"""
    - **{best_model}** is the best model (MAPE: {results[best_model]['MAPE']:.2f}%)
    - Linear trend + Price_Lag1 are the dominant features
    - Event-aware encoding improves prediction accuracy
    - COVID-19 caused the highest volatility period
    - Gold maintains upward trajectory during geopolitical crises
    """)
with col_c2:
    st.markdown("### Limitations")
    st.markdown("""
    - Monthly frequency misses intra-month volatility
    - Only 304 data points limits deep learning potential
    - Event windows are manually assigned (subjective)
    - No external macroeconomic features (inflation, USD index)
    - Neural network overfits with limited data
    """)
with col_c3:
    st.markdown("### Future Work")
    st.markdown("""
    - Daily price data for higher resolution
    - Transformer / LSTM models for sequence learning
    - Real-time event detection via news sentiment
    - Multi-commodity comparison (Silver, Oil, Bitcoin)
    - Interactive dashboard with live API feed
    """)

st.markdown("---")
col_exp1, col_exp2, _ = st.columns([1, 1, 2])
with col_exp1:
    def safe_val(x):
        if isinstance(x, (np.floating, float)):
            return None if np.isnan(x) else float(x)
        return float(x) if hasattr(x, '__float__') else str(x)

    full_report = {
        'best_model': best_model,
        'metrics': {n: {k: safe_val(v) for k, v in r.items() if k != 'predictions'}
                    for n, r in results.items()},
        'dataset_size': len(df),
        'date_range': {'start': str(df['Date'].min()), 'end': str(df['Date'].max())},
        'max_price': float(df['Price'].max()),
        'events': {str(k): {mk: safe_val(mv) for mk, mv in v.items()}
                   for k, v in stats.iterrows()} if not stats.empty else {}
    }
    report_json = json.dumps(full_report, indent=2, allow_nan=True).encode()
    st.download_button("Download Full Report (JSON)", report_json,
                        file_name="gold_price_analysis_report.json", mime="application/json")

with col_exp2:
    all_data_csv = df.to_csv(index=False).encode()
    st.download_button("Download Complete Dataset (CSV)", all_data_csv,
                        file_name="gold_price_complete.csv", mime="text/csv")

st.caption("Hamizah Aziim Bhikan | M.Sc. Data Science | Research Project Semester IV")
