import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config("ChainForge Analytics", layout="wide", page_icon="🔗")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .big-font { font-size:50px !important; font-weight:bold; text-align:center; color:#00ff00; }
    .card { padding:20px; border-radius:10px; box-shadow:5px 5px 15px #333; background:#1e1e1e; margin:10px 0; }
    .heatmap { background: linear-gradient(to right, green, yellow, red); -webkit-background-clip: text; color: transparent; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🔗 ChainForge Analytics</p>', unsafe_allow_html=True)
st.caption("Raw Crypto Insights | Volatility • Dominance • Sentiment • Education")

# Your pairs
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "LTC/USDT",
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "TRX/USDT", "LINK/USDT",
    "TON/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "UNI/USDT",
    "AAVE/USDT", "NEAR/USDT", "SUI/USDT"
]

exchange = ccxt.bitget({'enableRateLimit': True})

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "📚 Education"])

with tab1:
    st.header("Live Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Fear & Greed
    with col1:
        fng = requests.get("https://api.alternative.me/fng/?limit=1").json()['data'][0]
        value = int(fng['value'])
        classification = fng['value_classification']
        color = "red" if value < 25 else "orange" if value < 50 else "yellow" if value < 75 else "green"
        st.markdown(f"<div class='card'><h3>Fear & Greed</h3><h1 style='color:{color}'>{value}</h1><p>{classification}</p></div>", unsafe_allow_html=True)

    # BTC Dominance (CoinGecko)
    with col2:
        cg = requests.get("https://api.coingecko.com/api/v3/global").json()['data']
        dominance = round(cg['market_cap_percentage']['btc'], 2)
        st.markdown(f"<div class='card'><h3>BTC Dominance</h3><h1>{dominance}%</h1></div>", unsafe_allow_html=True)

    # Altcoin Index (simple)
    with col3:
        alt_index = round(100 - dominance, 2)
        st.markdown(f"<div class='card'><h3>Altcoin Index</h3><h1>{alt_index}%</h1><p>Higher = Alt Season</p></div>", unsafe_allow_html=True)

    # Placeholder for more
    with col4:
        st.markdown("<div class='card'><h3>Market Sentiment</h3><p>Coming Soon</p></div>", unsafe_allow_html=True)

    st.header("Volatility Heat Map (30d Annualized %)")

    data = []
    for pair in PAIRS:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=30)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ret'] = np.log(df['c'] / df['c'].shift(1))
            vol = df['ret'].std() * np.sqrt(365) * 100
            data.append({"Pair": pair, "Volatility %": round(vol, 2)})
        except:
            data.append({"Pair": pair, "Volatility %": "N/A"})

    df_vol = pd.DataFrame(data).sort_values("Volatility %", ascending=False)

    # Color-coded heatmap style
    def color_vol(val):
        if val == "N/A": return "background: gray"
        color = "red" if val > 100 else "orange" if val > 70 else "yellow" if val > 50 else "green"
        return f"background: {color}; color: white"

    st.dataframe(df_vol.style.applymap(color_vol, subset=["Volatility %"]))

with tab2:
    st.header("Learn Crypto Analysis Basics")

    with st.expander("📈 What is Volatility?"):
        st.write("""
        Volatility measures how much a crypto's price fluctuates. High volatility = big swings (good for short-term trades, high risk).
        - Measured as annualized % from daily returns.
        - Use it to size positions and set stops.
        """)

    with st.expander("🕐 Trading Sessions"):
        st.write("""
        - **Asian (00:00–08:00 UTC)**: Lower volume, consolidation.
        - **London (08:00–16:00 UTC)**: Higher volatility, trend starts.
        - **NY Overlap (12:00–16:00 UTC)**: Peak volume, big moves.
        """)

    with st.expander("📊 Bitcoin Dominance"):
        st.write("""
        % of total crypto market cap held by BTC.
        - High (>60%) = Risk-off, money in BTC.
        - Low (<40%) = Alt season, money flows to alts.
        """)

    with st.expander("😱 Fear & Greed Index"):
        st.write("""
        Sentiment gauge (0-100).
        - Extreme Fear (<25): Potential buying opportunity (contrarian).
        - Extreme Greed (>75): Potential top, caution.
        - Historical bottoms often at low scores.
        """)

st.success("ChainForge Analytics v0.1 Live | Data as of Jan 3, 2026")
