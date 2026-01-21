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
from modules.ai_query import process_query  # AI Query with OpenAI primary + Gemini fallback

@st.cache_data(ttl=300)
def fetch_ohlcv_cached(pair: str, tf: str = '1d', limit: int = 30):
    exchange = ccxt.bitget({'enableRateLimit': True})
    return exchange.fetch_ohlcv(pair, tf, limit=limit)
# Add this near the top of app.py (after imports, before st.set_page_config)
COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "LTC": "litecoin",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "TRX": "tron",
    "LINK": "chainlink",
    "TON": "toncoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "polygon",
    "UNI": "uniswap",
    "AAVE": "aave",
    "NEAR": "near-protocol",
    "SUI": "sui"
}

# New cached ticker function (add with the other cached functions)
@st.cache_data(ttl=60)  # Update every minute for live price/24h
def fetch_ticker_cached(pair: str):
    exchange = ccxt.bitget({'enableRateLimit': True})
    return exchange.fetch_ticker(pair)
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

    # Source Comparison Expander
    with st.expander("📊 Source Comparison & Notes"):
        st.write("""
        **Fear & Greed Index**:
        - ChainForge (alternative.me): Current value shown above (original/official source).
        - CoinMarketCap: Often higher (e.g., 38 vs 29) due to different weighting (more Google Trends, surveys).

        **Altcoin Index**:
        - ChainForge: Simple calculation (100 - BTC dominance) = {alt_index}% (transparent, real-time).
        - CoinMarketCap Alt Season Score: Proprietary score (0-100) based on alt outperformance vs BTC = 26 (more conservative).

        Use both for context — discrepancies are normal across platforms.
        """)

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
    # Economic Calendar (replace the old block)
st.header("Upcoming Crypto Events")
try:
    # CoinGecko has limited event data; use a free alternative or placeholder
    # Alternative: Static recent events or CoinMarketCap RSS (no API key needed)
    events_rss = requests.get("https://www.coingecko.com/en/upcoming_icos").text  # Placeholder page
    st.info("Live event calendar coming soon — check CoinGecko or CoinMarketCap manually for now.")
    st.markdown("[View Upcoming ICOs on CoinGecko](https://www.coingecko.com/en/upcoming_icos)")
    st.markdown("[CoinMarketCap Economic Calendar](https://coinmarketcap.com/events/)")
except:
    st.info("Event data unavailable")
    # === NEW: AI Natural Language Query ===
    st.subheader("Ask AI for Custom Insights")
    query = st.text_input("e.g., 'Compare volatility of PEPE and SHIB' or 'What does high BTC dominance mean?'")
    if query:
        with st.spinner("Thinking (OpenAI primary, Gemini fallback)..."):
            response = process_query(query, {'pairs': PAIRS, 'volatility_data': df_vol.to_dict('records')})
        st.markdown("### AI Response")
        st.write(response)

with tab2:
    st.header("Token Deep Dive")

    selected_pair = st.selectbox("Select Token", PAIRS, index=0)

    tf_options = ["1h", "4h", "1d", "1w"]
    selected_tf = st.selectbox("Timeframe", tf_options, index=2)

    try:
        # Fetch chart data
        ohlcv = fetch_ohlcv_cached(selected_pair, selected_tf, 200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Fetch accurate live ticker data (price, 24h change, 24h volume)
        ticker = fetch_ticker_cached(selected_pair)
        current_price = ticker['last']
        change_24h = ticker.get('percentage', 0) or 0  # Bitget provides 'percentage'
        volume_24h = ticker.get('quoteVolume', df['volume'].sum())  # Fallback to sum if not available

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Current Price", f"${current_price:,.4f}")
        with col2: st.metric("24h Change", f"{change_24h:.2f}%")
        with col3: st.metric("24h Volume", f"${volume_24h:,.0f}")
        with col4: st.metric("Period H/L", f"{df['high'].max():,.4f} / {df['low'].min():,.4f}")

        # app.py - Replace the On-Chain Metrics block in Tab 2 with this safe version
        # On-Chain Metrics
        base_coin = selected_pair.split('/')[0]
        coin_id = COIN_IDS.get(base_coin, base_coin.lower())
        metrics = get_on_chain_metrics(coin_id)

        if 'error' not in metrics:
            st.subheader("On-Chain & Community Metrics")

            # Helper to safely format large numbers
            def fmt(val, decimals=0):
                if val is None:
                    return "N/A"
                try:
                    return f"{val:,.{decimals}f}"
                except:
                    return "N/A"

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                rank = metrics.get('market_cap_rank')
                st.metric("Market Cap Rank", rank if rank is not None else "N/A")
            with col_b:
                st.metric("Twitter Followers", fmt(metrics.get('twitter_followers')))
            with col_c:
                st.metric("GitHub Stars", fmt(metrics.get('github_stars')))

            st.write(f"**Circulating Supply**: {fmt(metrics.get('circulating_supply'))}")
            st.write(f"**Total Supply**: {fmt(metrics.get('total_supply'))}")
        else:
            st.info("On-chain metrics unavailable for this token")

        # Sentiment Scoring (unchanged, but now more reliable with fixed IDs indirectly)
        sentiment_score = get_sentiment_score(selected_pair)
        if sentiment_score != 'N/A':
            sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "yellow"
            st.metric("News Sentiment Score (-100 to 100)", f"{sentiment_score}", delta_color="normal")
            st.caption("Based on recent news titles polarity. Positive = bullish.")
        else:
            st.metric("News Sentiment Score", "Unavailable")

        # Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{selected_pair} {selected_tf} Chart", height=700,
                          template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Data unavailable for {selected_pair}: {str(e)}")

# app.py - Replace the News Feed block in Tab 3 with this improved version
with tab3:
    st.header("Latest Crypto News")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (ChainForge Analytics App)'}
        url = "https://cryptopanic.com/api/v1/posts/?public=true&kind=news&filter=hot&page_size=15"
        news = requests.get(url, headers=headers, timeout=15).json()['results']
        if not news:
            raise Exception("No results")
        for article in news:
            title = article['title']
            published = article['published_at'][:10]
            domain = article['domain']
            url = article['url']
            votes = article.get('votes', {})
            sentiment = article.get('sentiment')
            with st.expander(f"📰 {title} ({published}) • {domain} • 👍{votes.get('positive',0)} 👎{votes.get('negative',0)}"):
                st.markdown(f"[Read full article]({url})")
                if sentiment:
                    st.caption(f"Detected sentiment: {sentiment}")
    except Exception as e:
        st.warning("News feed temporarily unavailable (rate limit, connection issue, or API maintenance). Try again later.")
        st.markdown("""
        Alternative sources:
        - [CryptoPanic](https://cryptopanic.com/)
        - [CoinTelegraph](https://cointelegraph.com/)
        - [CoinDesk](https://www.coindesk.com/)
        """)

with tab4:
    st.header("Learn Crypto Analysis Basics")

    with st.expander("📈 What is Volatility?"):
        st.write("""
        Volatility measures price fluctuations. High = big swings (opportunity + risk).
        - Annualized % from daily returns.
        - Use for position sizing/stops.
        """)

    with st.expander("🕐 Trading Sessions"):
        st.write("""
        - Asian (00:00–08:00 UTC): Low volume.
        - London (08:00–16:00 UTC): Trend starts.
        - NY Overlap (12:00–16:00 UTC): Peak volume.
        """)

    with st.expander("📊 Bitcoin Dominance"):
        st.write("""
        BTC's % of total market cap.
        - High (>60%): Risk-off.
        - Low (<40%): Alt season.
        """)

    with st.expander("😱 Fear & Greed Index"):
        st.write("""
        Sentiment gauge (0-100).
        - Extreme Fear (<25): Buy opportunity.
        - Extreme Greed (>75): Caution.
        """)

st.success("ChainForge Analytics v0.8 | AI Query (OpenAI + Gemini Fallback) | January 3, 2026")
