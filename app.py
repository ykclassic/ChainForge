import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. LIVE DATA ENGINES ---

def get_real_sentiment(pair):
    """Fetches LIVE news headlines from CryptoCompare API."""
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url).json()
        articles = response.get('Data', [])
        coin = pair.split('/')[0]
        relevant_headlines = [a['title'] for a in articles if coin in a['title'] or coin in a['categories']]
        if not relevant_headlines:
            relevant_headlines = [a['title'] for a in articles[:10]]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in relevant_headlines]
        return round(np.mean(scores), 2) if scores else 0.0, relevant_headlines[:5]
    except:
        return 0.0, ["News service temporarily unavailable"]

def get_order_microstructure(pair):
    """Real-time Order Book Imbalance."""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(pair, limit=20)
        spread = ((ob['asks'][0][0] - ob['bids'][0][0]) / ob['asks'][0][0]) * 100
        bid_v, ask_v = sum([x[1] for x in ob['bids']]), sum([x[1] for x in ob['asks']])
        imbalance = (bid_v - ask_v) / (bid_v + ask_v)
        return round(spread, 4), round(imbalance, 2)
    except: return 0.0, 0.0

# --- 2. DEEP LEARNING (LSTM) ---

def train_fusion_lstm(df, sentiment_score):
    """LSTM with Dropout trained on current timeframe sequences."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(10, 1)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    
    last_seq = scaled_data[-10:].reshape(1, 10, 1)
    raw_pred = scaler.inverse_transform(model.predict(last_seq))[0][0]
    return round(raw_pred * (1 + (sentiment_score * 0.02)), 2)

# --- 3. MAIN TERMINAL ---

st.set_page_config(page_title="ChainForge Elite", layout="wide", page_icon="⚡")

@st.cache_data(ttl=120)
def fetch_master_data(pair, timeframe):
    ex = ccxt.bitget({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(pair, timeframe, limit=150)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['vwap'] = (df['close'] * df['vol']).cumsum() / df['vol'].cumsum()
    return df

def main():
    st.title("⚡ ChainForge Elite: Institutional DSS")
    
    with st.sidebar:
        with st.form("config"):
            st.header("Global Controls")
            pair = st.selectbox("Active Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            # RESTORED TIMEFRAME SELECTOR
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "BNB"], default=["BTC", "ETH"])
            st.form_submit_button("Sync Elite Systems")

    df = fetch_master_data(pair, timeframe)
    spread, imbalance = get_order_microstructure(pair)
    sent_score, real_news = get_real_sentiment(pair)

    # Watchlist Tickers
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")
    st.divider()

    tab_mkt, tab_ai, tab_risk, tab_corr = st.tabs(["📊 Market Depth", "🧠 LSTM Fusion", "🧪 Risk Lab", "🌡️ Correlation"])

    with tab_mkt:
        c1, c2, c3 = st.columns(3)
        c1.metric("VWAP Deviation", f"{((df['close'].iloc[-1]/df['vwap'].iloc[-1])-1)*100:.2f}%")
        c2.metric("Order Imbalance", f"{imbalance}", delta="Liquid" if abs(imbalance) < 0.2 else "Pressure")
        c3.metric("Live Sentiment", f"{sent_score}")

        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        if st.button("🚀 Execute Neural Forecast"):
            prediction = train_fusion_lstm(df, sent_score)
            st.metric(f"Fusion {timeframe} Prediction", f"${prediction:,.2f}")
            st.write("**Real-Time Headlines Analyzed:**")
            for h in real_news: st.caption(f"• {h}")

    with tab_risk:
        if st.button("Run Monte Carlo (Real Volatility)"):
            vol = df['close'].pct_change().std() * np.sqrt(365 if timeframe == '1d' else 365*24)
            paths = df['close'].iloc[-1] * (1 + np.random.normal(0, vol/np.sqrt(365), (30, 100))).cumprod(axis=0)
            fig_mc = go.Figure()
            for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.2))
            st.plotly_chart(fig_mc, use_container_width=True)

    with tab_corr:
        prices = pd.DataFrame({c: [x[4] for x in ccxt.bitget().fetch_ohlcv(f"{c}/USDT", '1d', limit=30)] for c in watchlist})
        st.plotly_chart(px.imshow(prices.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

if __name__ == "__main__":
    main()
