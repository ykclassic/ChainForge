# === TABS SETUP (Place this after st.title and before any content) ===
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

    # BTC Dominance
    with col2:
        cg = requests.get("https://api.coingecko.com/api/v3/global").json()['data']
        dominance = round(cg['market_cap_percentage']['btc'], 2)
        st.markdown(f"<div class='card'><h3>BTC Dominance</h3><h1>{dominance}%</h1></div>", unsafe_allow_html=True)

    # Altcoin Index
    with col3:
        alt_index = round(100 - dominance, 2)
        st.markdown(f"<div class='card'><h3>Altcoin Index</h3><h1>{alt_index}%</h1><p>Higher = Alt Season</p></div>", unsafe_allow_html=True)

    # Placeholder
    with col4:
        st.markdown("<div class='card'><h3>Market Sentiment</h3><p>Coming Soon</p></div>", unsafe_allow_html=True)

    # Volatility Heat Map
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
    st.header("Learn Crypto Analysis Basics")

    with st.expander("📈 What is Volatility?"):
        st.write("Volatility measures price swings. High volatility = opportunity + risk. Calculated as annualized % from daily returns.")

    with st.expander("🕐 Trading Sessions"):
        st.write("- Asian: Low volume\n- London: Trend starts\n- NY Overlap: Peak action")

    with st.expander("📊 Bitcoin Dominance"):
        st.write("BTC's % of total market cap. High = safe haven; low = alt season.")

    with st.expander("😱 Fear & Greed Index"):
        st.write("Sentiment meter. Extreme Fear = buy signal; Extreme Greed = caution.")
