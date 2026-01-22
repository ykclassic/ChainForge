import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# Import modular components (ensure these files exist in /modules)
from modules.ai_query import process_query
from modules.whales import WhaleClient
from modules.backtest import run_backtest

# --- Configuration & Multi-Pair Logic ---
@st.cache_data(ttl=3600)  # Cache market list for 1 hour
def get_available_pairs():
    try:
        exchange = ccxt.bitget()
        markets = exchange.load_markets()
        # Filter for active USDT spot pairs
        pairs = [symbol for symbol, market in markets.items() 
                 if market['quote'] == 'USDT' and market['active'] and market['type'] == 'spot']
        return sorted(pairs)
    except Exception as e:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

@st.cache_data(ttl=120)
def fetch_enriched_data(pair, tf='1d', limit=150):
    try:
        exchange = ccxt.bitget({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(pair, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # --- TECHNICAL INDICATORS (Mandatory for Backtest) ---
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
        
        # Bollinger Bands
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + (df['std20'] * 2)
        df['lower_band'] = df['ma20'] - (df['std20'] * 2)
        
        return df.dropna() # Remove initial rows with NaN indicators
    except Exception as e:
        st.error(f"Error fetching {pair}: {e}")
        return pd.DataFrame()

# --- UI Application ---
st.set_page_config(page_title="ChainForge Pro", layout="wide")

def main():
    st.title("⚡ ChainForge Pro Intelligence")
    
    # Sidebar - Dynamic Pair Selection
    with st.sidebar:
        st.header("Terminal Settings")
        all_pairs = get_available_pairs()
        selected_pair = st.selectbox("Active Asset", all_pairs, index=all_pairs.index("BTC/USDT") if "BTC/USDT" in all_pairs else 0)
        timeframe = st.selectbox("Resolution", ["1h", "4h", "1d"], index=2)
        whale_min = st.number_input("Whale Threshold ($)", value=500000)

    # Fetch and Enrich Data
    df = fetch_enriched_data(selected_pair, timeframe)
    
    if df.empty:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Quant Suite", "🐋 Whales", "🧪 Backtest", "🧠 AI Report"])

    # 1. QUANT SUITE (Chart + Indicators)
    with tab1:
        last = df.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${last['close']:,.4f}")
        m2.metric("RSI", f"{last['rsi']:.1f}")
        m3.metric("BB Lower", f"${last['lower_band']:,.4f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Market"))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_band'], name="Upper BB", line=dict(color='rgba(255,255,255,0.2)')))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_band'], name="Lower BB", line=dict(color='rgba(255,255,255,0.2)')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 2. WHALE TRACKER
    with tab2:
        w_client = WhaleClient()
        whales = w_client.get_recent_whales(min_value=whale_min)
        if whales: st.dataframe(whales, use_container_width=True)
        else: st.info("No active whale movements detected.")

    # 3. BACKTEST (Strategy Lab)
    with tab3:
        st.subheader(f"Strategy Simulation: {selected_pair}")
        if st.button("Run Volatility Backtest"):
            results = run_backtest(df) # Fixed KeyError issue
            r1, r2 = st.columns(2)
            r1.metric("Total Return", f"{results['total_return']}%")
            r2.metric("Win Rate", f"{results['win_rate']}%")
            
            fig_eq = go.Figure(go.Scatter(x=results['df']['ts'], y=results['df']['equity_curve'], name="Equity"))
            fig_eq.update_layout(title="Growth of $10,000", template="plotly_dark")
            st.plotly_chart(fig_eq, use_container_width=True)

    # 4. AI REPORT
    with tab4:
        if st.button("Generate Analyst Confluence"):
            context = {
                "pair": selected_pair, "rsi": df['rsi'].iloc[-1],
                "price": df['close'].iloc[-1], "bb_lower": df['lower_band'].iloc[-1]
            }
            # AI Logic calls the Fallback automatically if APIs fail
            report = process_query("Give a trade bias based on RSI and Bollinger Bands.", context)
            st.markdown(report)

if __name__ == "__main__":
    main()
