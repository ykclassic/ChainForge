import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from modules.ai_query import process_query
from modules.on_chain import get_on_chain_metrics
import ccxt

# Professional Technical Indicators
def add_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['sma'] = df['close'].rolling(window=20).mean()
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma'] + (df['std'] * 2)
    df['lower_band'] = df['sma'] - (df['std'] * 2)
    return df

st.set_page_config("ChainForge Pro", layout="wide", page_icon="⚡")

# Sidebar - Settings
st.sidebar.title("⚡ Pro Settings")
exchange_id = st.sidebar.selectbox("Exchange Source", ["bitget", "binance", "kraken"])
enable_ai = st.sidebar.toggle("Enable AI Confluence", value=True)

# Data Fetching
@st.cache_data(ttl=120)
def get_pro_data(pair, tf='1d'):
    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(pair, tf, limit=100)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return add_indicators(df)

# Main UI
st.title("ChainForge Professional Analytics")

tab_market, tab_quant, tab_whales = st.tabs(["📊 Market Pulse", "📈 Quant Suite", "🐋 Whale Tracker"])

with tab_quant:
    col_sel, col_metrics = st.columns([1, 3])
    with col_sel:
        target = st.selectbox("Select Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT"])
        tf = st.radio("Timeframe", ["1h", "4h", "1d"], index=2)
    
    df = get_pro_data(target, tf)
    last_rsi = df['rsi'].iloc[-1]
    
    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("RSI (14)", f"{last_rsi:.2f}", delta="Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral")
        m2.metric("Volatility (30d)", f"{np.log(df['close']/df['close'].shift(1)).std() * np.sqrt(365) * 100:.1f}%")
        m3.metric("BB Width", f"{((df['upper_band'].iloc[-1] - df['lower_band'].iloc[-1])/df['sma'].iloc[-1])*100:.2f}%")

    # Advanced Charting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_band'], line=dict(color='rgba(173, 216, 230, 0.5)'), name="Upper BB"))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_band'], line=dict(color='rgba(173, 216, 230, 0.5)'), name="Lower BB", fill='tonexty'))
    
    fig.update_layout(title=f"{target} Technical Overlay", height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if enable_ai:
        st.subheader("AI Confluence Report")
        if st.button("Generate Signal"):
            context = {
                "asset": target,
                "rsi": last_rsi,
                "bb_pos": "Upper" if df['close'].iloc[-1] > df['upper_band'].iloc[-1] else "Lower" if df['close'].iloc[-1] < df['lower_band'].iloc[-1] else "Mid",
                "price": df['close'].iloc[-1]
            }
            query = f"Analyze {context['asset']} with RSI at {context['rsi']} and price at {context['bb_pos']} Bollinger Band. Provide a risk-adjusted trade bias."
            response = process_query(query, context)
            st.info(response)

with tab_whales:
    st.subheader("Live Blockchain Large Transfers")
    st.warning("Real-time Whale Alert API Integration Required for Live Data.")
    # Placeholder for Whale Logic
    st.table(pd.DataFrame({
        "Time": ["10:05", "10:12", "10:45"],
        "Amount": ["500 BTC", "12,000 ETH", "1,500,000 SOL"],
        "From": ["Unknown Wallet", "Binance", "Coinbase"],
        "To": ["Kraken", "Whale Wallet", "Unknown Wallet"],
        "Impact": ["High", "Neutral", "Medium"]
    }))
