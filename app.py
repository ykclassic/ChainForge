import sys
from pathlib import Path
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# Import modular components
from modules.on_chain import get_on_chain_metrics
from modules.sentiment import get_sentiment_score
from modules.ai_query import process_query
from modules.whales import WhaleClient
from modules.backtest import run_backtest

# --- Configuration & Constants ---
COIN_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "ADA": "cardano", "LTC": "litecoin", "DOGE": "dogecoin",
    "SHIB": "shiba-inu", "PEPE": "pepe", "TRX": "tron", "LINK": "chainlink",
    "TON": "toncoin", "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "polygon",
    "UNI": "uniswap", "AAVE": "aave", "NEAR": "near-protocol", "SUI": "sui"
}
PAIRS = [f"{k}/USDT" for k in COIN_IDS.keys()]

# --- Cached Data Fetching with Indicator Logic ---
@st.cache_data(ttl=300)
def fetch_pro_data(pair, tf='1d', limit=150):
    try:
        exchange = ccxt.bitget({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(pair, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # --- TECHNICAL INDICATORS (Required for Backtest) ---
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (Calculates lower_band correctly)
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + (df['std20'] * 2)
        df['lower_band'] = df['ma20'] - (df['std20'] * 2)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- UI Setup ---
st.set_page_config("ChainForge Pro", layout="wide", page_icon="⚡")

def main():
    st.markdown('<h1 style="text-align:center; color:#00ff00;">🔗 ChainForge Pro</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Control Panel")
        selected_pair = st.selectbox("Select Asset", PAIRS)
        selected_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
        whale_min = st.number_input("Whale Threshold ($)", value=500000)

    # Fetch Data (Indicators are now included in the DF)
    df = fetch_pro_data(selected_pair, selected_tf)
    
    if df.empty:
        st.warning("No data available for the selected asset.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🐋 Whales", "🧪 Backtest", "🧠 AI Report"])

    # 1. ANALYSIS TAB
    with tab1:
        last_row = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"${last_row['close']:.4f}")
        c2.metric("RSI (14)", f"{last_row['rsi']:.2f}")
        c3.metric("Volatility", f"{(df['close'].pct_change().std() * np.sqrt(365) * 100):.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close']))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_band'], name="Upper BB", line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_band'], name="Lower BB", line=dict(color='gray', dash='dash')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 2. WHALES TAB
    with tab2:
        w_client = WhaleClient()
        whales = w_client.get_recent_whales(min_value=whale_min)
        if whales:
            st.table(whales)
        else:
            st.info("No major whale movements detected.")

    # 3. BACKTEST TAB (Fixes the KeyError)
    with tab3:
        st.subheader("Strategy Simulation")
        if st.button("Run Backtest"):
            # The df now definitively has 'lower_band' and 'rsi' from fetch_pro_data
            results = run_backtest(df) 
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Return", f"{results['total_return']}%")
            res_col2.metric("Win Rate", f"{results['win_rate']}%")
            
            fig_equity = go.Figure(go.Scatter(x=results['df']['ts'], y=results['df']['equity_curve'], line=dict(color="#00ff00")))
            fig_equity.update_layout(title="Equity Growth", template="plotly_dark")
            st.plotly_chart(fig_equity, use_container_width=True)

    # 4. AI REPORT
    with tab4:
        if st.button("Generate AI Confluence"):
            context = {
                "pair": selected_pair,
                "rsi": df['rsi'].iloc[-1],
                "bb_status": "Touching" if df['close'].iloc[-1] <= df['lower_band'].iloc[-1] else "Neutral",
                "sentiment": get_sentiment_score(selected_pair)
            }
            report = process_query("Provide trade bias.", context)
            st.info(report)

if __name__ == "__main__":
    main()
