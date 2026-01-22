import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import modular components
from modules.ai_query import process_query
from modules.whales import WhaleClient
from modules.backtest import run_backtest

# --- Configuration ---
st.set_page_config(page_title="ChainForge Pro", layout="wide", page_icon="⚡")

@st.cache_data(ttl=3600)
def get_all_pairs():
    try:
        exchange = ccxt.bitget()
        markets = exchange.load_markets()
        return sorted([s for s, m in markets.items() if m['quote'] == 'USDT' and m['active'] and m['type'] == 'spot'])
    except:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

@st.cache_data(ttl=120)
def fetch_clean_df(pair, tf='1d', limit=100):
    exchange = ccxt.bitget({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(pair, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # Technical Indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    return df.dropna()

def calculate_correlations(pairs):
    """Calculates Pearson correlation matrix using log returns."""
    combined_returns = pd.DataFrame()
    exchange = ccxt.bitget()
    
    for pair in pairs:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=30)
            close_prices = [x[4] for x in ohlcv]
            # Log Returns: ln(Price_t / Price_t-1)
            returns = np.log(pd.Series(close_prices) / pd.Series(close_prices).shift(1))
            combined_returns[pair.split('/')[0]] = returns
        except:
            continue
    return combined_returns.corr()

# --- Main UI ---
def main():
    st.title("⚡ ChainForge Pro: Institutional Terminal")
    
    all_pairs = get_all_pairs()
    
    with st.sidebar:
        st.header("Global Controls")
        watchlist = st.multiselect("Active Watchlist", all_pairs, default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs)
        st.divider()
        selected_pair = st.selectbox("Focus Asset", all_pairs, index=0)
        timeframe = st.selectbox("Interval", ["1h", "4h", "1d"], index=2)
        whale_min = st.number_input("Whale Min ($)", value=500000)

    # 1. Watchlist Ticker Row
    if watchlist:
        st.subheader("📡 Market Pulse")
        tickers = ccxt.bitget().fetch_tickers(watchlist)
        cols = st.columns(len(watchlist))
        for i, pair in enumerate(watchlist):
            t = tickers.get(pair, {})
            cols[i].metric(pair.split('/')[0], f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")
    
    st.divider()

    # 2. Main Analytics Tabs
    tab_charts, tab_matrix, tab_whales, tab_strategy, tab_ai = st.tabs([
        "📈 Charting", "🌡️ Correlation", "🐋 Whales", "🧪 Strategy Lab", "🧠 AI Analyst"
    ])

    df = fetch_clean_df(selected_pair, timeframe)

    with tab_charts:
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price")])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_band'], name="Upper BB", line=dict(color='rgba(255,255,255,0.2)', dash='dot')))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_band'], name="Lower BB", line=dict(color='rgba(255,255,255,0.2)', dash='dot')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_matrix:
        st.subheader("Asset Correlation Matrix (30D)")
        if len(watchlist) > 1:
            with st.spinner("Calculating relationships..."):
                corr_matrix = calculate_correlations(watchlist)
                fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Values close to 1.0 mean assets move together. Negative values mean they move oppositely.")
        else:
            st.info("Add more coins to your watchlist to see correlations.")

    with tab_whales:
        whales = WhaleClient().get_recent_whales(min_value=whale_min)
        st.table(whales) if whales else st.info("No major whale movements.")

    with tab_strategy:
        if st.button("Run Simulation"):
            res = run_backtest(df)
            st.metric("Total Profit", f"{res['total_return']}%", delta=f"Wins: {res['win_rate']}%")
            st.line_chart(res['df'].set_index('ts')['equity_curve'])

    with tab_ai:
        if st.button("Generate Senior Report"):
            ctx = {"pair": selected_pair, "rsi": df['rsi'].iloc[-1], "price": df['close'].iloc[-1]}
            st.markdown(process_query("Perform a technical confluence report.", ctx))

if __name__ == "__main__":
    main()
