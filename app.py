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

# --- 1. DATA ENGINES ---

def send_discord_alert(webhook_url, content, embed=None):
    payload = {"content": content}
    if embed: payload["embeds"] = [embed]
    try: requests.post(webhook_url, json=payload)
    except: pass

def get_real_sentiment(pair):
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
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(pair, limit=20)
        spread = ((ob['asks'][0][0] - ob['bids'][0][0]) / ob['asks'][0][0]) * 100
        bid_v, ask_v = sum([x[1] for x in ob['bids']]), sum([x[1] for x in ob['asks']])
        return round(spread, 4), round((bid_v - ask_v) / (bid_v + ask_v), 2)
    except: return 0.0, 0.0

# --- 2. AI & CORRELATION ENGINES ---

def train_fusion_lstm(df, sentiment):
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

@st.cache_data(ttl=600)
def get_correlation_data(watchlist):
    """Restored: Fetches 30-day history for the entire watchlist to correlate."""
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
    st.title("⚡ ChainForge Elite: Quant Terminal")
    
    with st.sidebar:
        with st.form("config"):
            pair = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "BNB", "XRP"], default=["BTC", "ETH", "SOL"])
            discord_webhook = st.text_input("Discord Webhook", type="password")
            st.form_submit_button("Sync Live Systems")

    df = fetch_master_data(pair, timeframe)
    spread, imbalance = get_microstructure(pair)
    sent_score, news = get_real_sentiment(pair)

    # 1. LIVE TICKERS
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")

    # 2. ANALYSIS TABS
    tab_mkt, tab_ai, tab_corr, tab_risk = st.tabs(["📊 Market", "🧠 AI Fusion", "🌡️ Correlation", "🧪 Risk"])

    with tab_mkt:
        c1, c2, c3 = st.columns(3)
        c1.metric("VWAP Deviation", f"{((df['close'].iloc[-1]/df['vwap'].iloc[-1])-1)*100:.2f}%")
        c2.metric("Order Imbalance", f"{imbalance}")
        c3.metric("Sentiment", f"{sent_score}")
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        if st.button("🚀 Run Neural Signal"):
            pred = train_fusion_lstm(df, sent_score)
            delta = ((pred/df['close'].iloc[-1])-1)*100
            verdict = "NEUTRAL"
            if delta > 0.5 and sent_score > 0 and imbalance > 0.1: verdict = "STRONG BUY"
            elif delta < -0.5 and sent_score < 0 and imbalance < -0.1: verdict = "STRONG SELL"
            
            st.header(f"Verdict: {verdict}")
            st.metric("LSTM Target", f"${pred:,.2f}", f"{delta:+.2f}%")
            for h in news: st.caption(f"• {h}")
            
            if discord_webhook and verdict != "NEUTRAL":
                embed = {"title": f"🚨 {verdict}: {pair}", "description": f"Target: ${pred:,.2f}\nSentiment: {sent_score}", "color": 65280 if "BUY" in verdict else 16711680}
                send_discord_alert(discord_webhook, f"Elite Alert", embed)

    with tab_corr:
        st.subheader("Asset Correlation Matrix (30D Rolling)")
        corr_df = get_correlation_data(watchlist)
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("High correlation (0.8+) suggests assets move together. Diversify if correlation is too high.")

    with tab_risk:
        if st.button("Run Monte Carlo"):
            vol = df['close'].pct_change().std() * np.sqrt(365 if timeframe == '1d' else 365*24)
            paths = df['close'].iloc[-1] * (1 + np.random.normal(0, vol/np.sqrt(365), (30, 50))).cumprod(axis=0)
            fig_mc = go.Figure()
            for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.2))
            st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
