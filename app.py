import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- MODULE IMPORTS (Assumed internal logic based on previous steps) ---
from modules.ai_query import process_query
from modules.on_chain import get_on_chain_metrics
from modules.whales import WhaleClient
from modules.backtest import run_backtest

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="ChainForge Pro", layout="wide", page_icon="⚡")

def apply_custom_css():
    st.markdown("""
    <style>
        .metric-card { background: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
        .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC FUNCTIONS ---
@st.cache_data(ttl=120)
def fetch_pro_ohlcv(symbol, timeframe='1d', limit=150):
    exchange = ccxt.bitget({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # Technical Indicators (Quant Suite)
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['ma20'] + (df['std20'] * 2)
    df['lower_bb'] = df['ma20'] - (df['std20'] * 2)
    
    return df

# --- MAIN INTERFACE ---
def main():
    apply_custom_css()
    st.title("⚡ ChainForge Pro Intelligence")
    
    # Sidebar: Global Controls
    with st.sidebar:
        st.header("Terminal Settings")
        selected_pair = st.selectbox("Active Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT"], index=0)
        timeframe = st.selectbox("Resolution", ["1h", "4h", "1d"], index=2)
        st.divider()
        whale_min = st.number_input("Whale Alert Threshold ($)", value=500000)
        st.caption("v2.1.0-Pro | 2026 Edition")

    # Data Fetching
    df = fetch_pro_ohlcv(selected_pair, timeframe)
    last_price = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    # Tabs: The Three Pillars of Pro Trading
    tab_quant, tab_whales, tab_backtest, tab_ai = st.tabs([
        "📈 Quant Suite", "🐋 Whale Tracker", "🧪 Strategy Lab", "🧠 AI Confluence"
    ])

    # 1. QUANT SUITE
    with tab_quant:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${last_price:,.2f}")
        col2.metric("RSI (14)", f"{last_rsi:.2f}", delta="Neutral" if 30 < last_rsi < 70 else "Extreme")
        col3.metric("BB Bandwidth", f"{((df['upper_bb'].iloc[-1] - df['lower_bb'].iloc[-1])/df['ma20'].iloc[-1]*100):.2f}%")
        col4.metric("30d Vol (Ann.)", f"{(df['close'].pct_change().std() * np.sqrt(365) * 100):.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_bb'], line=dict(color='rgba(173, 216, 230, 0.4)'), name="Upper BB"))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_bb'], fill='tonexty', line=dict(color='rgba(173, 216, 230, 0.4)'), name="Lower BB"))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig, use_container_width=True)

    # 2. WHALE TRACKER
    with tab_whales:
        st.subheader("Institutional Flow Analysis")
        w_client = WhaleClient() # Uses st.secrets internally
        whale_data = w_client.get_recent_whales(min_value=whale_min)
        if whale_data:
            st.table(pd.DataFrame(whale_data))
        else:
            st.info("No active whale movements above threshold detected.")

    # 3. STRATEGY LAB (Backtesting)
    with tab_backtest:
        st.subheader("Historical Strategy Validation")
        st.write("Current Strategy: **Mean Reversion (RSI + BB Squeeze)**")
        if st.button("Run Simulation on Active Asset"):
            results = run_backtest(df) # Vectorized engine
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Strategy Return", f"{results['total_return']}%")
            res_col2.metric("Win Rate", f"{results['win_rate']}%")
            
            fig_eq = go.Figure(go.Scatter(x=results['df']['ts'], y=results['df']['equity_curve'], line=dict(color="#00ff00")))
            fig_eq.update_layout(title="Equity Curve ($10k Start)", template="plotly_dark")
            st.plotly_chart(fig_eq, use_container_width=True)

    # 4. AI CONFLUENCE
    with tab_ai:
        st.subheader("ChainForge AI Analyst")
        if st.button("Generate Senior Analysis Report"):
            with st.spinner("Synthesizing Quant, On-Chain, and Whale data..."):
                analysis_context = {
                    "pair": selected_pair,
                    "rsi": last_rsi,
                    "price": last_price,
                    "whale_activity": "High" if len(whale_data) > 5 else "Low",
                    "trend": "Bullish" if last_price > df['ma20'].iloc[-1] else "Bearish"
                }
                report = process_query("Provide a professional trade bias based on confluence.", analysis_context)
                st.markdown(f"### Diagnostic Report\n{report}")

if __name__ == "__main__":
    main()
