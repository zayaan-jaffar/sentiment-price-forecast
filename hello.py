# Data Importing Libraries
import yfinance as yf

# Data Modeling Library
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charts
import plotly.graph_objects as go

# Data Manipulation
import pandas as pd
import numpy as np

# Avoid Forecasting on Holidays
import holidays

# Interactive Web App UI
import streamlit as st

import asyncio
from tqdm.asyncio import tqdm_asyncio

import time
from datetime import datetime, timedelta
import feedparser
from urllib.parse import quote

# HuggingFace Sentiment Analysis
from transformers import pipeline

# Load FinBERT sentiment model
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

valid_labels = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def google_news_historical(ticker, company_name, months_back=12):
    """Google News with date-specific searches"""
    all_articles = []
    end_date = datetime.now()

    for i in range(months_back):
        try:
            current_month = end_date - timedelta(days=30 * i)
            prev_month = end_date - timedelta(days=30 * (i + 1))

            after_date = prev_month.strftime('%Y-%m-%d')
            before_date = current_month.strftime('%Y-%m-%d')

            query = f"{company_name} {ticker} after:{after_date} before:{before_date}"
            encoded_query = quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            feed = feedparser.parse(rss_url)

            for entry in feed.entries[:]:
                try:
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6]).date()

                    all_articles.append({
                        'date': pub_date,
                        'title': getattr(entry, 'title', ''),
                        'link': getattr(entry, 'link', ''),
                        'source': 'Google News Historical'
                    })
                except Exception:
                    continue

            time.sleep(0.5)  # Be respectful
        except Exception:
            continue

    return pd.DataFrame(all_articles)


# -------------------------------
# Async Sentiment Classification
# -------------------------------

async def classify_batch_async(titles):
    """Classify a batch of titles with FinBERT asynchronously."""
    def _classify(batch):
        results = classifier(batch, truncation=True)
        labels = []
        for r in results:
            label = r["label"].upper()
            labels.append(label)
        return labels

    return await asyncio.to_thread(_classify, titles)

async def classify_sentiments_async(titles):
    """Classify a batch of titles asynchronously with FinBERT."""
    tasks = [classify_sentiment_async(t) for t in titles]
    sentiments = await asyncio.gather(*tasks)
    return sentiments


async def get_news_data_async(df, batch_size=16):
    """
    Async version of news data processing:
    - Deduplicate
    - Lowercase titles
    - Async batched sentiment classification with progress bar
    - Drop NEUTRAL
    - Parse dates
    """
    news_df = df.drop_duplicates(subset=['title']).copy()
    news_df['title'] = news_df['title'].str.lower()

    # Split into batches
    batches = [
        news_df['title'].iloc[i:i + batch_size].tolist()
        for i in range(0, len(news_df), batch_size)
    ]

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, batch in enumerate(batches):
        sentiments = await classify_batch_async(batch)
        results.extend(sentiments)

        # Update Streamlit progress bar
        progress = int((i + 1) / len(batches) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing batch {i+1} of {len(batches)}...")

    progress_bar.empty()
    status_text.text("Classification complete ✅")

    news_df['Sentiment'] = results
    news_df = news_df[news_df['Sentiment'] != 'NEUTRAL']

    # Parse dates
    news_df['Date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date
    news_df = news_df.dropna(subset=['Date'])
    news_df.set_index('Date', inplace=True)

    return news_df


def process_sentiment_data(news_df):
    grouped = news_df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)

    grouped['7day_avg_positive'] = grouped['POSITIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['NEGATIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_pct_positive'] = grouped['POSITIVE'] / (grouped['POSITIVE'] + grouped['NEGATIVE'])

    grouped.index = pd.to_datetime(grouped.index)
    return grouped


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.columns = stock_data.columns.get_level_values(0)
    stock_data['Pct_change'] = stock_data['Close'].pct_change() * 100
    stock_data = stock_data[['Pct_change']]
    return stock_data


def combine_data(result_df, stock_data):
    combined_df = result_df.join(stock_data, how='inner')
    combined_df['lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)
    return combined_df


def calculate_correlation(combined_df):
    correlation_pct_change = combined_df[['lagged_7day_pct_positive', 'Pct_change']].corr().iloc[0, 1]
    cpc2 = combined_df[['7day_pct_positive', 'Pct_change']].corr().iloc[0, 1]
    return correlation_pct_change, cpc2


def get_future_dates(start_date, num_days):
    us_holidays = holidays.US()
    future_dates = []
    current_date = start_date
    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)
    return future_dates


def fit_and_forecast(combined_df, forecast_steps=3):
    combined_clean = combined_df[['Pct_change', '7day_pct_positive']].dropna()

    endog = combined_clean['Pct_change'].values
    exog = combined_clean[['7day_pct_positive']].values

    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
    fit = model.fit(disp=False)

    future_dates = get_future_dates(combined_df.index[-1], forecast_steps)
    future_exog = combined_clean[['7day_pct_positive']].iloc[-forecast_steps:].values

    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    return forecast_mean, forecast_ci, future_dates, fit


def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):
    sentiment_std = (combined_df['7day_pct_positive'] - combined_df['7day_pct_positive'].mean()) / combined_df[
        '7day_pct_positive'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=sentiment_std,
        name='Standardized Sentiment Proportion',
        line=dict(color='blue'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Pct_change'],
        name='Stock Pct Change',
        line=dict(color='green'),
        yaxis='y2',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted Pct Change',
        line=dict(color='red'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_index, forecast_index[::-1]]),
        y=np.concatenate([forecast_ci[:, 0], forecast_ci[:, 1][::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title='Sentiment Proportion and Stock Percentage Change with Forecast',
        xaxis_title='Date',
        yaxis=dict(
            title=dict(text='Standardized Sentiment Proportion', font=dict(color='blue'))
        ),
        yaxis2=dict(
            title=dict(text='Stock Pct Change', font=dict(color='green')),
            overlaying='y',
            side='right'
        ),
        template='plotly_dark',
        showlegend=False
    )
    st.plotly_chart(fig)


def plot_residuals_time(fit, combined_df):
    residuals = fit.resid
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df.index[-len(residuals):],
        y=residuals,
        mode='lines',
        name='Residuals'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="SARIMAX Residuals Over Time",
        xaxis_title="Date",
        yaxis_title="Residual (Actual - Predicted)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)


# -------------------------------
# Streamlit App
# -------------------------------
st.sidebar.title("Predicting Stock Prices by AI News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker:", value='AAPL')
company_name = st.sidebar.text_input("Enter stock company:", value='Apple')

run_button = st.sidebar.button('Analyze')


async def run_analysis(ticker):
    df = google_news_historical(ticker, company_name, months_back=10)
    news_df = await get_news_data_async(df)
    result_df = process_sentiment_data(news_df)
    start_date = result_df.index.min().strftime('%Y-%m-%d')
    end_date = result_df.index.max().strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)
    combined_df = combine_data(result_df, stock_data)
    correlation_pct_change, cpc2 = calculate_correlation(combined_df)

    forecast_mean, forecast_ci, forecast_index, fit = fit_and_forecast(combined_df)

    return combined_df, forecast_mean, forecast_ci, forecast_index, fit, correlation_pct_change, cpc2


if run_button:
    with st.spinner('Processing sentiment analysis...'):
        try:
            combined_df, forecast_mean, forecast_ci, forecast_index, fit, correlation_pct_change, cpc2 = asyncio.run(
                run_analysis(ticker)
            )

            create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)
            plot_residuals_time(fit, combined_df)
            st.write(f'Pearson correlation between lagged sentiment and stock percentage change: {correlation_pct_change}')
            st.write(f'Pearson correlation between sentiment and stock percentage change: {cpc2}')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

