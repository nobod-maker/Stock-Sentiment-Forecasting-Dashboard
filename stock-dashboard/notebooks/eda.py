# Stock Sentiment & Forecasting Dashboard — EDA Notebook
# =========================================================
# This is a plain-Python script version of the EDA.
# Open as Jupyter notebook by converting with: jupyter nbconvert --to notebook eda.py
# Or just run it as a script: python notebooks/eda.py

"""
Exploratory Data Analysis
=========================
Run this notebook to understand the dataset before model training.
"""

# Cell 1: Imports and Setup
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv('../.env')

from src.data_loader import fetch_price_data, fetch_news_headlines, headlines_to_dataframe
from src.features import build_features, time_split
from src.sentiment import score_headlines, aggregate_daily_sentiment

TICKER = "AAPL"
START = "2021-01-01"
END = "2024-01-01"

print(f"Analysing {TICKER} from {START} to {END}")

# Cell 2: Price Data
price_df = fetch_price_data(TICKER, START, END)
print(f"\nPrice data shape: {price_df.shape}")
print(price_df.head())
print("\nDescriptive stats:")
print(price_df.describe())

# Cell 3: Missing Data Check
print("\nMissing values:")
print(price_df.isnull().sum())

# Cell 4: Returns Distribution
price_df['Daily_Return'] = price_df['Close'].pct_change()
fig = px.histogram(
    price_df.dropna(),
    x='Daily_Return',
    nbins=100,
    title=f'{TICKER} Daily Returns Distribution',
    color_discrete_sequence=['#636EFA']
)
fig.show()

# Cell 5: Correlation Matrix
feature_df = build_features(price_df)
feature_cols = [c for c in feature_df.columns if c not in ['Next_Close', 'Target_Direction']]

corr = feature_df[feature_cols + ['Next_Close']].corr()
fig2 = px.imshow(
    corr,
    title='Feature Correlation Matrix',
    color_continuous_scale='RdBu',
    aspect='auto'
)
fig2.show()

# Cell 6: Train/Test Split Visualisation
train_df, test_df = time_split(feature_df)
print(f"\nTrain: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} rows)")
print(f"Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df)} rows)")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], name='Train', line=dict(color='#636EFA')))
fig3.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Test', line=dict(color='#EF553B')))
fig3.update_layout(title=f'{TICKER} Train/Test Split', template='plotly_dark')
fig3.show()

# Cell 7: News Headlines
headlines = fetch_news_headlines(TICKER, days=30)
print(f"\nFetched {len(headlines)} headlines")
if headlines:
    hdf = headlines_to_dataframe(headlines)
    print(hdf[['date', 'title', 'source']].head(10))

# Cell 8: Sentiment Distribution (if headlines available)
if headlines:
    scored = score_headlines(hdf)
    daily_sent = aggregate_daily_sentiment(scored)
    print(f"\nSentiment distribution:\n{scored['label'].value_counts()}")
    
    fig4 = px.pie(
        scored,
        names='label',
        title='Headline Sentiment Distribution',
        color_discrete_map={
            'positive': '#00CC96',
            'negative': '#EF553B',
            'neutral': '#636EFA'
        }
    )
    fig4.show()
    
    print(f"\nDaily sentiment stats:\n{daily_sent['sentiment_mean'].describe()}")

print("\nEDA Complete!")
