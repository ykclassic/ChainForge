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

# --- 1. CORE DATA ENGINES (REAL-TIME ONLY) ---

def get_real_sentiment(pair):
    """Live News Narrative Engine."""
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url).json()
        articles = response.get('Data', [])
        coin = pair.split('/')[0]
        relevant = [a['title'] for a in articles if coin in a['title'] or coin in a['categories']]
        if not relevant: relevant = [a['title'] for a in articles[:8]]
        
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in relevant]
        return round(np.mean(scores), 2) if scores else 0.0, relevant[:5]
    except: return 0.0, ["News feed currently refreshing..."]

def get_microstructure(pair):
    """Real-time Liquidity Depth & Imbalance."""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(pair, limit=20)
        spread = ((ob['asks'][0][0] - ob['bids'][0][0]) / ob['asks'][0][0]) * 100
        bid_v, ask_v = sum([x[1] for x in ob['bids']]), sum([x[1] for x in ob['asks']])
        imbalance = (bid_v - ask_v) / (bid_v + ask_v)
        return round(spread, 4), round(imbalance, 2)
    except: return 0.0, 0.0

# --- 2. THE CONFIDENCE ENGINE ---

def calculate_confidence(lstm_delta, sentiment, imbalance):
    """Institutional Logic: Fuses 3 different market signals."""
    # 1. Neural Signal (Price prediction vs current)
    neural_score = 1 if lstm_delta > 0.5 else (-1 if lstm_delta < -0.5 else 0)
    # 2. Sentiment Score
    sent_score = 1 if sentiment > 0.1 else (-1 if sentiment < -0.1 else 0)
    # 3. Microstructure Score (Bid vs Ask wall)
    ob_score = 1 if imbalance > 0.15 else (-1 if imbalance < -0.15 else 0)
    
    total = neural_score + sent_score + ob_score
    confidence = (abs(total) / 3) * 100
    
    if total >= 2: verdict, color = "STRONG BUY", "#00FF00"
    elif total <= -2: verdict, color = "STRONG SELL", "#FF0000"
    else: verdict, color = "NEUTRAL / HOLD", "#FFA500"
    
    return verdict, confidence, color

# --- 3. DEEP LEARNING (LSTM) PIPELINE ---

def train_fusion_lstm(df, sentiment):
    """LSTM with sequence memory + sentiment bias."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']].values)
    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i, 0]); y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(10, 1)),
        Dropout(0.3), LSTM(32), Dropout(0.3), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    
    last_seq = scaled[-10:].reshape(1, 10, 1)
    raw_pred = scaler.inverse_transform(model.predict(last_seq))[0][0]
    return round(raw_pred * (1 + (sentiment * 0.015)), 2)

# --- 4. MAIN INTERFACE ---

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
    st.title("⚡ ChainForge Elite: Signal-Fusion Terminal")
    
    with st.sidebar:
        with st.form("config"):
            pair = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "BNB"], default=["BTC", "ETH"])
            st.form_submit_button("Update Terminal")

    # Data Sync
    df = fetch_master_data(pair, timeframe)
    spread, imbalance = get_microstructure(pair)
    sent_score, news = get_real_sentiment(pair)
    
    # Header: Live Tickers
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")
    st.divider()

    # Dashboard
    tab_mkt, tab_ai, tab_risk = st.tabs(["📊 Market Intelligence", "🧠 AI Fusion Signal", "🧪 Strategy Lab"])

    with tab_mkt:
        c1, c2, c3 = st.columns(3)
        c1.metric("VWAP Deviation", f"{((df['close'].iloc[-1]/df['vwap'].iloc[-1])-1)*100:.2f}%")
        c2.metric("Order Imbalance", f"{imbalance}")
        c3.metric("Live Sentiment", f"{sent_score}")
        
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        if st.button("🔥 Execute AI Signal Fusion"):
            with st.spinner("Synchronizing Neural Networks..."):
                pred = train_fusion_lstm(df, sent_score)
                current = df['close'].iloc[-1]
                delta = ((pred/current)-1)*100
                
                # Calculate Fusion Verdict
                verdict, conf, color = calculate_confidence(delta, sent_score, imbalance)
                
                # Signal Display
                st.markdown(f"<h2 style='text-align: center; color: {color};'>{verdict}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: {conf:.0f}%</p>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("LSTM Target", f"${pred:,.2f}", f"{delta:+.2f}%")
                with c2:
                    st.write("**Top News Factors:**")
                    for h in news: st.caption(f"• {h}")

    with tab_risk:
        st.subheader("Monte Carlo Risk Simulation")
        if st.button("Run Simulation"):
            vol = df['close'].pct_change().std() * np.sqrt(365 if timeframe == '1d' else 365*24)
            paths = df['close'].iloc[-1] * (1 + np.random.normal(0, vol/np.sqrt(365), (30, 50))).cumprod(axis=0)
            fig_mc = go.Figure()
            for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.2))
            st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
