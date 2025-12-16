# app.py
# AI-Powered Stock Market Simulator with News-Driven Text Generation
# Tech: Python, Streamlit, Pandas, scikit-learn (basic)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Stock Simulator", layout="wide")

st.title("📈 AI-Powered Stock Market Simulator")
st.caption("News-driven simulation with simple ML + text generation (educational)")

# -----------------------------
# Data Source Selection
# -----------------------------
st.sidebar.header("Data Source")
data_source = st.sidebar.selectbox(
    "Choose Stock Dataset Source",
    ["Yahoo Finance", "Alpha Vantage", "Upload CSV"]
)

ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1mo","3mo","6mo","1y"], index=2)

# -----------------------------
# Yahoo Finance Loader
# -----------------------------
def load_yahoo(ticker, period):
    df = yf.download(ticker, period=period)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df[['date','open','high','low','close']]

# -----------------------------
# Alpha Vantage Loader
# -----------------------------
def load_alpha_vantage(symbol):
    key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
    if not key:
        st.warning("Alpha Vantage API key missing")
        return pd.DataFrame()

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY&symbol={symbol}"
        f"&apikey={key}&outputsize=compact"
    )
    r = requests.get(url).json()
    data = r.get("Time Series (Daily)", {})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).T
    df = df.rename(columns={
        "1. open":"open",
        "2. high":"high",
        "3. low":"low",
        "4. close":"close"
    })
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index":"date"})
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df.sort_values("date")

# -----------------------------
# CSV Upload Loader
# -----------------------------
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df[['date','open','high','low','close']]

# -----------------------------
# Load Selected Dataset
# -----------------------------
if data_source == "Yahoo Finance":
    df = load_yahoo(ticker, period)

elif data_source == "Alpha Vantage":
    df = load_alpha_vantage(ticker)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)
    else:
        df = pd.DataFrame()

if df.empty:
    st.error("❌ No data loaded from selected source")
    st.stop()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Simulation Controls")

future_days = st.sidebar.slider("Predict days ahead", 1, 30, 7)
news_sentiment = st.sidebar.selectbox(
    "Market News Sentiment",
    ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
)

sentiment_impact = {
    "Very Negative": -0.05,
    "Negative": -0.02,
    "Neutral": 0.0,
    "Positive": 0.02,
    "Very Positive": 0.05
}[news_sentiment]

# -----------------------------
# Prepare ML Model (Regression)
# -----------------------------

# Sort by date and create a time-based feature
df = df.sort_values("date")
df['day_index'] = np.arange(len(df))

X = df[['day_index']]          # Independent variable (time)
y = df['close']                # Dependent variable (price)

# Ensure sufficient data for regression
if len(df) < 15:
    st.error("❌ Dataset must contain at least 15 rows for regression model")
    st.stop()

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Future Prediction
# -----------------------------
# -----------------------------
last_index = df['day_index'].iloc[-1]
future_index = np.array([
    last_index + i for i in range(1, future_days + 1)
]).reshape(-1, 1)

base_preds = model.predict(future_index)
adjusted_preds = base_preds * (1 + sentiment_impact)

future_dates = [
    df['date'].iloc[-1] + timedelta(days=i)
    for i in range(1, future_days + 1)
]

pred_df = pd.DataFrame({
    "date": future_dates,
    "predicted_price": adjusted_preds
})

# -----------------------------
# Charts (with safety checks)
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Prices (Candlestick / Wave)")
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_width=1,
            decreasing_line_width=1
        )])

        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Candlestick chart unavailable: {e}")

with col2:
    st.subheader("Predicted Prices")
    if not pred_df.empty:
        st.line_chart(pred_df.set_index('date')['predicted_price'])
    else:
        st.warning("Prediction data not available")

# -----------------------------
# News-driven Text Generation (News API)
# -----------------------------
import requests

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

@st.cache_data(ttl=3600)
def fetch_market_news():
    if not NEWS_API_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        "q=stock market OR finance OR economy&"
        "language=en&"
        "sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get("articles", [])[:5]

news_articles = fetch_market_news()

st.subheader("📰 Latest Market News")
if news_articles:
    for article in news_articles:
        st.markdown(f"**{article['title']}**")
        st.caption(article.get('description', ''))
else:
    st.info("No live news available (API key missing or limit reached)")

# AI-style market summary using sentiment + live news
def generate_market_summary(sentiment, days, articles):
    headlines = ", ".join([a['title'] for a in articles[:3]]) if articles else "recent market developments"

    if sentiment == "Very Positive":
        tone = "strong bullish momentum driven by optimistic news"
    elif sentiment == "Positive":
        tone = "steady upward movement supported by positive signals"
    elif sentiment == "Negative":
        tone = "mild downward pressure due to unfavorable headlines"
    elif sentiment == "Very Negative":
        tone = "sharp bearish sentiment caused by negative indicators"
    else:
        tone = "sideways movement as markets digest mixed news"

    return (
        f"📊 Market Outlook:\n"
        f"Over the next {days} days, the stock is expected to show {tone}. "
        f"This outlook considers historical price trends and live news such as: {headlines}. "
        f"This is an educational simulation, not financial advice."
    )

st.subheader("🧠 AI-Generated Market Commentary")
st.write(generate_market_summary(news_sentiment, future_days, news_articles))

# -----------------------------
# Data Preview
# -----------------------------
with st.expander("📂 View Raw Dataset"):
    st.dataframe(df.tail(20))