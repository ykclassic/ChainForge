import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import StringIO

from modules.on_chain import get_on_chain_metrics
from modules.sentiment import get_sentiment_score

@st.cache_data(ttl=300)
def fetch_ohlcv_cached(pair: str, tf: str = '1d', limit: int = 30):
    exchange = ccxt.bitget({'enableRateLimit': True})
    return exchange.fetch_ohlcv(pair, tf, limit=limit)

st.set_page_config("ChainForge Analytics", layout="wide", page_icon="🔗")

# Theme Toggle
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
theme_css = """
<style>
    .big-font { font-size:50px !important; font-weight:bold; text-align:center; color:#00ff00; }
    .card { padding:20px; border-radius:10px; box-shadow:5px 5px 15px #333; background:#1e1e1e; margin:10px 0; }
</style>
""" if theme == "Dark" else """
<style>
    .big-font { font-size:50px !important; font-weight:bold; text-align:center; color:#0000ff; }
    .card { padding:20px; border-radius:10px; box-shadow:5px 5px 15px #ccc; background:#f0f0f0; margin:10px 0; }
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

st.markdown('<p class="big-font">🔗 ChainForge Analytics</p>', unsafe_allow_html=True)
st.caption("Raw Crypto Insights | Volatility • Dominance • Sentiment • On-Chain • News • Education")

# Pairs
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "LTC/USDT",
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "TRX/USDT", "LINK/USDT",
    "TON/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "UNI/USDT",
    "AAVE/USDT", "NEAR/USDT", "SUI/USDT"
]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Token Deep Dive", "📰 News Feed", "📚 Education"])

with tab1:
    st.header("Live Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Fear & Greed
    with col1:
        try:
            fng = requests.get("https://api.alternative.me/fng/?limit=1").json()['data'][0]
            value = int(fng['value'])
            classification = fng['value_classification']
            color = "red" if value < 25 else "orange" if value < 50 else "yellow" if value < 75 else "green"
            st.markdown(f"<div class='card'><h3>Fear & Greed</h3><h1 style='color:{color}'>{value}</h1><p>{classification}</p></div>", unsafe_allow_html=True)
        except:
            st.markdown("<div class='card'><h3>Fear & Greed</h3><p>Unavailable</p></div>", unsafe_allow_html=True)

    # BTC Dominance
    with col2:
        try:
            cg = requests.get("https://api.coingecko.com/api/v3/global").json()['data']
            dominance = round(cg['market_cap_percentage']['btc'], 2)
            st.markdown(f"<div class='card'><h3>BTC Dominance</h3><h1>{dominance}%</h1></div>", unsafe_allow_html=True)
        except:
            st.markdown("<div class='card'><h3>BTC Dominance</h3><p>Unavailable</p></div>", unsafe_allow_html=True)

    # Altcoin Index
    with col3:
        try:
            alt_index = round(100 - dominance, 2)
            st.markdown(f"<div class='card'><h3>Altcoin Index</h3><h1>{alt_index}%</h1><p>Higher = Alt Season</p></div>", unsafe_allow_html=True)
        except:
            st.markdown("<div class='card'><h3>Altcoin Index</h3><p>Unavailable</p></div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'><h3>Market Sentiment</h3><p>Coming Soon</p></div>", unsafe_allow_html=True)

    # Volatility Heat Map
    st.header("Volatility Heat Map (30d Annualized %)")

    data = []
    for pair in PAIRS:
        try:
            ohlcv = fetch_ohlcv_cached(pair, '1d', 30)
            if len(ohlcv) < 2:
                data.append({"Pair": pair, "Volatility %": "N/A"})
                continue
                
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ret'] = np.log(df['c'] / df['c'].shift(1))
            vol = df['ret'].std() * np.sqrt(365) * 100
            if np.isnan(vol) or not np.isfinite(vol):
                data.append({"Pair": pair, "Volatility %": "N/A"})
            else:
                data.append({"Pair": pair, "Volatility %": round(vol, 2)})
        except Exception:
            data.append({"Pair": pair, "Volatility %": "N/A"})

    df_vol = pd.DataFrame(data)
    df_vol["Volatility %"] = pd.to_numeric(df_vol["Volatility %"], errors='coerce')
    df_vol = df_vol.sort_values("Volatility %", ascending=False, na_position='last')

    def color_vol(val):
        if pd.isna(val):
            return "background: gray; color: white"
        if val > 120:
            return "background-color: #8B0000; color: white"
        elif val > 90:
            return "background-color: red; color: white"
        elif val > 60:
            return "background-color: orange; color: white"
        elif val > 30:
            return "background-color: yellow; color: black"
        else:
            return "background-color: green; color: white"

    styled_df = df_vol.style.applymap(color_vol, subset=["Volatility %"])
    st.dataframe(styled_df, use_container_width=True)

    csv = df_vol.to_csv(index=False)
    st.download_button("📥 Export Volatility Data (CSV)", csv, "volatility.csv", "text/csv")

    # Correlation Matrix
    st.header("30d Correlation Matrix (Selected Pairs)")
    corr_pairs = st.multiselect("Select Pairs for Correlation", PAIRS, default=PAIRS[:5])

    corr_data = {}
    for pair in corr_pairs:
        ohlcv = fetch_ohlcv_cached(pair, '1d', 30)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        corr_data[pair] = df['c'].pct_change()

    corr_df = pd.DataFrame(corr_data).corr()

    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu', title="Correlation Heat Map")
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    csv_corr = corr_df.to_csv()
    st.download_button("📥 Export Correlation Matrix (CSV)", csv_corr, "correlation.csv", "text/csv")

    # Economic Calendar
    st.header("Upcoming Crypto Events (Next 7 Days)")
    try:
        events = requests.get("https://api.coinmarketcap.com/data-api/v3/calendar/events?limit=10").json()['data']
        for event in events:
            date = event['date'][:10]
            if (datetime.now() - datetime.fromisoformat(date)).days > 7: continue
            with st.expander(f"📅 {event['title']} ({date})"):
                st.write(event['description'])
    except:
        st.info("Economic calendar unavailable — CMC API limit")

# Token Deep Dive, News, Education tabs (unchanged from previous full code)

st.success("ChainForge Analytics v0.7 | Caching Fixed | January 3, 2026")
