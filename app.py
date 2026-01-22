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

# --- 2. REGIME & AI ENGINES ---

def calculate_market_regime(df, period=14):
    """Detects if market is Trending (<38.2) or Ranging (>61.8)."""
    tr = pd.DataFrame()
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    atr_sum = tr['tr'].rolling(period).sum()
    high_h = df['high'].rolling(period).max()
    low_l = df['low'].rolling(period).min()
    
    # Choppiness Index Formula
    chop = 100 * np.log10(atr_sum / (high_h - low_l)) / np.log10(period)
    val = chop.iloc[-1]
    
    if val < 38.2: return "TRENDING", val, "#00FF00"
    elif val > 61.8: return "CHOPPY / RANGING", val, "#FF4B4B"
    else: return "TRANSITIONAL", val, "#FFA500"

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

@st.cache_data(ttl=600)
def get_correlation_data(watchlist):
    ex = ccxt.bitget()
    data = {}
    for coin in watchlist:
        try:
            ohlcv = ex.fetch_ohlcv(f"{coin}/USDT", '1d', limit=30)
            data[coin] = [x[4] for x in ohlcv]
        except: continue
    return pd.DataFrame(data).pct_change().corr()

def main():
    st.title("⚡ ChainForge Elite: Master Quant Terminal")
    
    # RESTORED ASSET LIST
    full_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT"]
    
    with st.sidebar:
        with st.form("config"):
            pair = st.selectbox("Primary Asset", full_assets)
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "XRP", "BNB"], default=["BTC", "ETH", "SOL"])
            discord_webhook = st.text_input("Discord Webhook", type="password")
            st.form_submit_button("Sync Terminal")

    df = fetch_master_data(pair, timeframe)
    spread, imbalance = get_microstructure(pair)
    sent_score, news = get_real_sentiment(pair)
    regime_name, regime_val, regime_col = calculate_market_regime(df)

    # 1. LIVE TICKER STRIP
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")

    # 2. ANALYSIS TABS
    tab_mkt, tab_ai, tab_corr, tab_risk = st.tabs(["📊 Market", "🧠 Master Signal", "🌡️ Correlation", "🧪 Risk Lab"])

    with tab_mkt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", regime_name)
        c2.metric("VWAP Dev", f"{((df['close'].iloc[-1]/df['vwap'].iloc[-1])-1)*100:.2f}%")
        c3.metric("Imbalance", imbalance)
        c4.metric("Sentiment", sent_score)
        
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        if st.button("🚀 Generate Profitable Signal"):
            pred = train_fusion_lstm(df, sent_score)
            current_price = df['close'].iloc[-1]
            volatility = df['close'].pct_change().std()
            
            sig = generate_institutional_signal(current_price, pred, sent_score, imbalance, volatility)
            
            # Highlight Market Regime status in signal
            st.markdown(f"**Market Condition:** <span style='color:{regime_col};'>{regime_name}</span>", unsafe_allow_html=True)
            
            if sig['verdict'] == "NEUTRAL":
                st.warning("SYSTEM NEUTRAL: Convergence failed.")
            else:
                st.header(f"SIGNAL: {sig['verdict']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("ENTRY", f"${sig['entry']:,.2f}")
                c2.metric("STOP LOSS", f"${sig['stop_loss']:,.2f}")
                c3.metric("TAKE PROFIT", f"${sig['take_profit']:,.2f}")
                
                if discord_webhook:
                    embed = {"title": f"🚨 {sig['verdict']}: {pair}", "description": f"Entry: ${sig['entry']}\nRegime: {regime_name}", "color": 65280 if "BUY" in sig['verdict'] else 16711680}
                    send_discord_alert(discord_webhook, "Elite Alert", embed)

    with tab_corr:
        st.plotly_chart(px.imshow(get_correlation_data(watchlist), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

    with tab_risk:
        vol_ann = df['close'].pct_change().std() * np.sqrt(365 if timeframe == '1d' else 365*24)
        paths = df['close'].iloc[-1] * (1 + np.random.normal(0, vol_ann/np.sqrt(365), (30, 50))).cumprod(axis=0)
        fig_mc = go.Figure()
        for i in range(50): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.15))
        st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
