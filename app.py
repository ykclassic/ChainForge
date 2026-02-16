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

# --- 1. CORE DATA ENGINES ---

# Ensure the modules directory is recognized
try:
    from modules.signal import generate_institutional_signal
except ImportError:
    st.error("🚨 CRITICAL ERROR: Could not find 'modules/signal.py'. Check directory structure.")
    st.stop()

def send_discord_alert(webhook_url, content, embed=None):
    payload = {"content": content}
    if embed: 
        payload["embeds"] = [embed]
    try: 
        requests.post(webhook_url, json=payload, timeout=5)
    except: 
        pass

def get_real_sentiment(pair):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url, timeout=10).json()
        articles = response.get('Data', [])
        coin = pair.split('/')[0]
        relevant = [a['title'] for a in articles if coin in a['title'] or coin in a['categories']]
        if not relevant: 
            relevant = [a['title'] for a in articles[:8]]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in relevant]
        return round(float(np.mean(scores)), 2) if scores else 0.0, relevant[:5]
    except: 
        return 0.0, ["News feed refreshing..."]

def calculate_market_regime(df, period=14):
    """Calculates CHOP Index: <38.2 = Trending, >61.8 = Ranging."""
    tr = pd.DataFrame()
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr_sum = tr['tr'].rolling(period).sum()
    high_h = df['high'].rolling(period).max()
    low_l = df['low'].rolling(period).min()
    
    # Mathematical calculation of CHOP index
    chop = 100 * np.log10(atr_sum / (high_h - low_l)) / np.log10(period)
    val = chop.iloc[-1]
    
    if val < 38.2: 
        return "TRENDING", "#00FF00"
    elif val > 61.8: 
        return "RANGING", "#FF4B4B"
    return "TRANSITIONAL", "#FFA500"

# --- 2. AI FUSION ENGINE ---

def train_fusion_lstm(df, sentiment):
    """Generates a price prediction using LSTM fused with sentiment bias."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']].values)
    
    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i, 0])
        y.append(scaled[i, 0])
    
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
    
    last_seq = scaled[-10:].reshape(1, 10, 1)
    raw_pred = scaler.inverse_transform(model.predict(last_seq))[0][0]
    
    # Apply sentiment bias: Each 1.0 sentiment adds 1.5% to the prediction
    return round(float(raw_pred * (1 + (sentiment * 0.015))), 2)

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
    
    full_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT"]
    
    with st.sidebar:
        with st.form("config"):
            pair = st.selectbox("Primary Asset", full_assets)
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "XRP", "BNB"], default=["BTC", "ETH", "SOL"])
            discord_webhook = st.text_input("Discord Webhook", type="password")
            st.form_submit_button("Sync Terminal")

    df = fetch_master_data(pair, timeframe)
    sent_score, news = get_real_sentiment(pair)
    regime, r_col = calculate_market_regime(df)
    
    # Order Book Imbalance Calculation
    ex = ccxt.bitget()
    ob = ex.fetch_order_book(pair, limit=20)
    bids_sum = sum([x[1] for x in ob['bids']])
    asks_sum = sum([x[1] for x in ob['asks']])
    imbalance = (bids_sum - asks_sum) / (bids_sum + asks_sum)

    # Ticker Header
    tickers = ex.fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")

    # TABS
    tab_mkt, tab_ai, tab_corr, tab_risk = st.tabs(["📊 Market Intelligence", "🧠 Master Signal", "🌡️ Correlation", "🧪 Risk Lab"])

    with tab_mkt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", regime)
        c2.metric("Sentiment", sent_score)
        c3.metric("Imbalance", round(imbalance, 2))
        vwap_val = df['vwap'].iloc[-1]
        close_val = df['close'].iloc[-1]
        c4.metric("VWAP Dev", f"{((close_val/vwap_val)-1)*100:.2f}%")
        
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        if st.button("🚀 Generate Profitable Signal"):
            pred = train_fusion_lstm(df, sent_score)
            vol = df['close'].pct_change().std()
            
            # Logic calling the external module
            sig = generate_institutional_signal(df['close'].iloc[-1], pred, sent_score, imbalance, vol)
            
            st.markdown(f"**Market Regime:** <span style='color:{r_col};'>{regime}</span>", unsafe_allow_html=True)
            if sig['verdict'] == "NEUTRAL":
                st.warning("SYSTEM NEUTRAL: Convergence failed. No trade recommended.")
            else:
                st.header(f"TRADE SIGNAL: {sig['verdict']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("ENTRY", f"${sig['entry']:,.2f}")
                c2.metric("STOP LOSS", f"${sig['stop_loss']:,.2f}")
                c3.metric("TAKE PROFIT", f"${sig['take_profit']:,.2f}")
                st.progress(sig['confidence'] / 100)
                st.caption(f"System Confidence: {sig['confidence']}%")
                
                if discord_webhook:
                    color_code = 65280 if "BUY" in sig['verdict'] else 16711680
                    embed = {
                        "title": f"🚨 {sig['verdict']}: {pair}", 
                        "description": f"Entry: ${sig['entry']}\nTarget: ${sig['take_profit']}\nRegime: {regime}", 
                        "color": color_code
                    }
                    send_discord_alert(discord_webhook, "Elite Trade Alert", embed)

    with tab_corr:
        corr_data = {}
        for c in watchlist:
            try:
                h = ex.fetch_ohlcv(f"{c}/USDT", '1d', limit=30)
                corr_data[c] = [x[4] for x in h]
            except: 
                continue
        if corr_data:
            corr_df = pd.DataFrame(corr_data).pct_change().corr()
            st.plotly_chart(px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

    with tab_risk:
        # Monte Carlo Simulation
        days_in_year = 365 if timeframe == '1d' else 365*24
        vol_ann = df['close'].pct_change().std() * np.sqrt(days_in_year)
        
        # Generating 50 random paths for the next 30 steps
        last_price = df['close'].iloc[-1]
        returns = np.random.normal(0, vol_ann/np.sqrt(365), (30, 50))
        paths = last_price * (1 + returns).cumprod(axis=0)
        
        fig_mc = go.Figure()
        for i in range(50): 
            fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1), opacity=0.15, showlegend=False))
        fig_mc.update_layout(title="Monte Carlo Price Projection (30 Periods)", template="plotly_dark")
        st.plotly_chart(fig_mc, use_container_width=True)

if __name__ == "__main__":
    main()
