import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression

# Import unified backend
from core_engine import (
    get_real_sentiment, get_imbalance, generate_institutional_signal, 
    generate_pro_signal, send_discord_alert, DUMMY_DISCORD_WEBHOOK
)

st.set_page_config(page_title="ChainForge Master Suite", layout="wide", page_icon="⚡")

# --- CORE UTILITIES ---

@st.cache_data(ttl=120)
def fetch_master_data(pair, timeframe, limit=200):
    ex = ccxt.bitget({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(pair, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['vwap'] = (df['close'] * df['vol']).cumsum() / df['vol'].cumsum()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    return df.dropna().reset_index(drop=True)

def calculate_market_regime(df, period=14):
    tr = pd.DataFrame()
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    atr_sum = tr['tr'].rolling(period).sum()
    high_h = df['high'].rolling(period).max()
    low_l = df['low'].rolling(period).min()
    diff = high_h - low_l
    
    chop = 100 * np.log10(atr_sum / diff.replace(0, np.inf)) / np.log10(period)
    val = chop.iloc[-1]
    if val < 38.2: return "TRENDING", "#00FF00"
    elif val > 61.8: return "RANGING", "#FF4B4B"
    return "TRANSITIONAL", "#FFA500"

def train_fusion_lstm(df, sentiment):
    data = df[['close']].values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled = scaler.transform(data)
    
    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i, 0])
        y.append(scaled[i, 0])
    
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

def train_linear_model(df):
    data = df[['close', 'rsi']].copy()
    data['target'] = data['close'].shift(-1)
    train_data = data.dropna()
    
    X = train_data[['close', 'rsi']]
    y = train_data['target']
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_features = df[['close', 'rsi']].iloc[-1:].values
    prediction = model.predict(last_features)[0]
    confidence = model.score(X, y) * 100
    
    return round(prediction, 2), round(confidence, 1)

# --- MAIN APP LOGIC ---

def main():
    st.title("⚡ ChainForge Master Suite")
    full_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    
    with st.sidebar:
        st.header("Global Configuration")
        pair = st.selectbox("Asset", full_assets)
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
        watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "XRP"], default=["BTC", "ETH", "SOL"])
        
    df = fetch_master_data(pair, timeframe)
    
    # State Management for Pro ML Module
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
        st.session_state.confidence = 0

    mode_tab1, mode_tab2 = st.tabs(["🚀 Elite Quant Terminal (Standard)", "🤖 Pro ML Suite"])
    
    # --- STANDARD TERMINAL ---
    with mode_tab1:
        sent_score, news = get_real_sentiment(pair)
        regime, r_col = calculate_market_regime(df)
        imbalance = get_imbalance(pair)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", regime)
        c2.metric("Sentiment", sent_score)
        c3.metric("Imbalance", imbalance)
        c4.metric("VWAP Dev", f"{((df['close'].iloc[-1]/df['vwap'].iloc[-1])-1)*100:.2f}%")
        
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['vwap'], name="VWAP", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Generate Institutional Signal", key="std_sig"):
            with st.spinner("Calculating convergence..."):
                pred = train_fusion_lstm(df, sent_score)
                vol = df['close'].pct_change().std()
                sig = generate_institutional_signal(df['close'].iloc[-1], pred, sent_score, imbalance, vol)
                
                if sig['verdict'] == "NEUTRAL":
                    st.warning("SYSTEM NEUTRAL: Convergence failed. No trade recommended.")
                else:
                    st.success(f"TRADE SIGNAL: {sig['verdict']}")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("ENTRY", f"${sig['entry']:,.2f}")
                    sc2.metric("STOP LOSS", f"${sig['stop_loss']:,.2f}")
                    sc3.metric("TAKE PROFIT", f"${sig['take_profit']:,.2f}")
                    
                    embed = {
                        "title": f"🚨 Standard Alert: {sig['verdict']} on {pair}", 
                        "description": f"Entry: ${sig['entry']}\nRegime: {regime}", 
                        "color": 65280 if "BUY" in sig['verdict'] else 16711680
                    }
                    send_discord_alert(DUMMY_DISCORD_WEBHOOK, "ChainForge Standard", embed)

    # --- PRO ML SUITE ---
    with mode_tab2:
        st.subheader("Machine Learning Price Forecast & Advanced TFT Signals")
        
        if st.button("Run Pro Analysis", key="pro_sig"):
            with st.spinner("Analyzing neural patterns and order flow..."):
                pred, conf = train_linear_model(df)
                st.session_state.prediction = pred
                st.session_state.confidence = conf
                
                pro_sig = generate_pro_signal(df, get_real_sentiment(pair)[0], get_imbalance(pair))
                st.session_state.pro_sig = pro_sig
                
        if st.session_state.prediction:
            col1, col2, col3 = st.columns(3)
            current_price = df['close'].iloc[-1]
            diff = st.session_state.prediction - current_price
            
            col1.metric("Predicted Price (Next Period)", f"${st.session_state.prediction:,}", delta=f"{diff:+.2f}")
            col2.metric("Model Confidence (R²)", f"{st.session_state.confidence}%")
            
            psig = st.session_state.get('pro_sig', {})
            verdict_str = psig.get('verdict', 'NEUTRAL')
            col3.metric("Elite TFT Verdict", verdict_str)
            
            if verdict_str != "NEUTRAL":
                st.info(f"**Pro Risk Management:** Entry: ${psig.get('entry', 0):,.2f} | Stop: ${psig.get('stop_loss', 0):,.2f} | TP: ${psig.get('take_profit', 0):,.2f} | ATR: {psig.get('atr', 0)}")
            
            future_date = df['ts'].iloc[-1] + timedelta(days=1 if timeframe == '1d' else (1/24 if timeframe == '1h' else 4/24))
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df['ts'][-20:], y=df['close'][-20:], name="Actual Price", line=dict(color="cyan")))
            fig_pred.add_trace(go.Scatter(x=[df['ts'].iloc[-1], future_date], y=[df['close'].iloc[-1], st.session_state.prediction], name="ML Forecast", line=dict(color="orange", dash="dash")))
            fig_pred.update_layout(template="plotly_dark", title="Trend Projection", height=400)
            st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()
