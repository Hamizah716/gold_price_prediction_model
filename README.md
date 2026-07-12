# Analyzing the Impact of Global Events on Gold Prices Using ML Models

**M.Sc. Data Science (Semester IV) Research Project**  
**Hamizah Aziim Bhikan | Roll No. DS24105**  
**University of Mumbai, Department of Computer Science (2025-26)**

---

## Overview

An interactive Streamlit web application that analyzes monthly gold price data (2000-2025) and trains 4 ML regression models to predict gold prices, with 6-month forecasting and event impact analysis.

### Key Features
- **Data Overview** — Explore gold price trends from 2000-2025 with 7 major global events labeled
- **Model Performance** — Compare 4 ML models: Linear Regression, Random Forest, XGBoost, Neural Network
- **Prediction** — Interactive price prediction with 6-month forecasting
- **Event Analysis** — Analyze volatility and impact of global events on gold prices

### Models Used
| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Linear Regression | 92.97 | 68.85 | 3.03% |
| Neural Network (MLP) | 323.57 | 207.73 | 8.24% |
| Random Forest | 522.57 | 311.30 | 11.81% |
| XGBoost | 546.27 | 329.29 | 12.55% |

### Global Events Analyzed
1. Dot-com Crash (2000-2002)
2. 9/11 Attacks (2001-2002)
3. Global Financial Crisis (2008-2009)
4. European Debt Crisis (2010-2012)
5. Oil Price Crash (2014-2015)
6. COVID-19 Pandemic (2019-2021)
7. Russia-Ukraine War (2022-2025)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo
https://gold-price-prediction-ml.streamlit.app

## Dataset
Monthly gold prices (USD/oz) from 2000-2025 sourced from DataHub Core Gold Prices.
