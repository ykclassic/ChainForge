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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🔗 ChainForge Analytics</p>', unsafe_allow_html=True)
st.caption("Raw Crypto Insights | Volatility • Dominance • Sentiment • On-Chain • News • Education")

# Your pairs
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "LTC/USDT",
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "TRX/USDT", "LINK/USDT",
    "TON/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "UNI/USDT",
    "AAVE/USDT", "NEAR/USDT", "SUI/USDT"
]

exchange = ccxt.bitget({'enableRateLimit': True})

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

    # Volatility Heat Map (fully fixed)
    st.header("Volatility Heat Map (30d Annualized %)")

    data = []
    for pair in PAIRS:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=30)
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

with tab2:
    st.header("Token Deep Dive")

    selected_pair = st.selectbox("Select Token", PAIRS, key="deep_dive_pair")

    tf_options = ["1h", "4h", "1d", "1w"]
    selected_tf = st.selectbox("Timeframe", tf_options, index=2, key="deep_dive_tf")

    try:
        ohlcv = exchange.fetch_ohlcv(selected_pair, selected_tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        current_price = df['close'].iloc[-1]
        change_24h = ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100) if len(df) > 24 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Current Price", f"${current_price:,.4f}")
        with col2: st.metric("24h Change", f"{change_24h:.2f}%")
        with col3: st.metric("Avg Volume", f"{df['volume'].mean():,.0f}")
        with col4: st.metric("30d Volatility", f"{df['close'].pct_change().std() * np.sqrt(365) * 100:.2f}%")

        # Enhanced On-Chain Metrics (CoinGecko free API)
        base_coin = selected_pair.split('/')[0].lower()
        try:
            cg_data = requests.get(f"https://api.coingecko.com/api/v3/coins/{base_coin}").json()
            market_data = cg_data.get('market_data', {})
            community = cg_data.get('community_data', {})
            developer = cg_data.get('developer_data', {})

            st.subheader("On-Chain & Community Metrics")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a: st.metric("Market Cap Rank", market_data.get('market_cap_rank', 'N/A'))
            with col_b: st.metric("Active Addresses (Est)", "Coming (Glassnode free tier soon)")
            with col_c: st.metric("Twitter Followers", f"{community.get('twitter_followers', 'N/A'):,}")
            with col_d: st.metric("GitHub Stars", developer.get('stars', 'N/A'))

            # Additional fundamentals
            st.write(f"**Circulating Supply**: {market_data.get('circulating_supply', 'N/A')}")
            st.write(f"**Total Supply**: {market_data.get('total_supply', 'N/A')}")
        except:
            st.info("Detailed on-chain data unavailable for this token (CoinGecko limit or token not listed)")

        # Interactive Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{selected_pair} {selected_tf} Chart", height=700, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Data unavailable for {selected_pair}: {str(e)}")

with tab3:
    st.header("Latest Crypto News")

    try:
        # CryptoPanic free public feed (hot news)
        news = requests.get("https://cryptopanic.com/api/v1/posts/?public=true&kind=news&filter=hot").json()['results']
        for article in news[:15]:
            title = article['title']
            published = article['published_at'][:10]
            domain = article['domain']
            url = article['url']
            with st.expander(f"📰 {title} ({published}) • {domain}"):
                st.markdown(f"[Read full article]({url})")
    except Exception as e:
        st.error("News feed unavailable — check connection or API limits")

with tab4:
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

st.success("ChainForge Analytics v0.3 | On-Chain + News Enhanced | January 3, 2026")
