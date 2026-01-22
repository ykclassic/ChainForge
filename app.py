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
from signal import generate_institutional_signal

# --- 1. CORE DATA ENGINES ---

def send_discord_alert(webhook_url, content, embed=None):
    payload = {"content": content}
    if embed: payload["embeds"] = [embed]
    try: requests.post(webhook_url, json=payload)
    except: pass

def get_real_sentiment(pair):
    """Fetches real-time news and performs VADER sentiment analysis."""
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
    except: return 0.0, ["News feed refreshing..."]

def get_microstructure(pair):
    """Fetches live Order Book to calculate bid/ask imbalance."""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(pair, limit=20)
        spread = ((ob['asks'][0][0] - ob['bids'][0][0]) / ob['asks'][0][0]) * 100
        bid_v, ask_v = sum([x[1] for x in ob['bids']]), sum([x[1] for x in ob['asks']])
        imbalance = (bid_v - ask_v) / (bid_v + ask_v)
        return round(spread, 4), round(imbalance, 2)
    except: return 0.0, 0.0

# --- 2. AI ENGINES ---

def train_fusion_lstm(df, sentiment):
    """Deep Learning with 30% dropout and sentiment bias."""
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
    # Bias prediction slightly based on news sentiment
    return round(raw_pred * (1 + (sentiment * 0.015)), 2)

@st.cache_data(ttl=600)
def get_correlation_data(watchlist):
    """Restored: 30-day rolling correlation heatmap."""
    ex = ccxt.bitget()
    data = {}
    for coin in watchlist:
        try:
            ohlcv = ex.fetch_ohlcv(f"{coin}/USDT", '1d', limit=30)
            data[coin] = [x[4] for x in ohlcv]
        except: continue
    return pd.DataFrame(data).pct_change().corr()

# --- 3. MAIN INTERFACE ---

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
    st.title("⚡ ChainForge Elite: Master Quant Terminal")
    
    # 1. EXPANDED ASSET LIST (Restored)
    full_asset_list = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", 
        "BNB/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", 
        "DOT/USDT", "LTC/USDT"
    ]
    
    with st.sidebar:
        with st.form("config"):
            pair = st.selectbox("Primary Focus Asset", full_asset_list)
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist (Correlation)", ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE"], default=["BTC", "ETH", "SOL"])
            discord_webhook = st.text_input("Discord Webhook", type="password")
            st.form_submit_button("Sync Master Systems")

    # Data Sync
    df = fetch_master_data(pair, timeframe)
    spread, imbalance = get_microstructure(pair)
    sent_score, news = get_real_sentiment(pair)

    # Live Tickers
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")

    tab_mkt, tab_ai, tab_corr, tab_risk = st.tabs(["📊 Market Intelligence", "🧠 Master Signal", "🌡️ Correlation Matrix", "🧪 Risk Lab"])

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
        if st.button("🚀 Generate Institutional Signal"):
            with st.spinner("Calculating Signal Fusion..."):
                pred = train_fusion_lstm(df, sent_score)
                current_price = df['close'].iloc[-1]
                volatility = df['close'].pct_change().std()
                
                # EXECUTE MODULE SIGNAL
                sig = generate_institutional_signal(current_price, pred, sent_score, imbalance, volatility)
                
                if sig['verdict'] == "NEUTRAL":
                    st.warning("SYSTEM NEUTRAL: Market convergence not reached. Stand by.")
                else:
                    st.header(f"TRADE SIGNAL: {sig['verdict']}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ENTRY", f"${sig['entry']:,.2f}")
                    c2.metric("STOP LOSS", f"${sig['stop_loss']:,.2f}", delta_color="inverse")
                    c3.metric("TAKE PROFIT", f"${sig['take_profit']:,.2f}")
                    
                    st.progress(sig['confidence'] / 100)
                    st.caption(f"System Confidence: {sig['confidence']}% | Target R/R: {sig['rr_ratio']}")
                    
                    if discord_webhook:
                        embed = {
                            "title": f"🚨 {sig['verdict']} Signal: {pair}",
                            "description": f"Entry: ${sig['entry']}\nSL: ${sig['stop_loss']}\nTP: ${sig['take_profit']}\nConfidence: {sig['confidence']}%",
                            "color": 65280 if sig['verdict'] == "BUY" else 16711680
                        }
                        send_discord_alert(discord_webhook, f"ChainForge Elite Signal", embed)

    with tab_corr:
        st.subheader("Asset Correlation Heatmap")
        corr_df = get_correlation_data(watchlist)
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_risk:
        st.subheader("Monte Carlo Path Simulation")
        vol_ann = df['close'].pct_change().std() * np.sqrt(365 if timeframe == '1d' else 365*24)
        paths = df['close'].iloc[-1] * (1 + np.random.normal(0, vol_ann/np.sqrt(365), (30, 50))).cumprod(axis=0)
        fig_mc = go.Figure()
        for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.15))
        st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
