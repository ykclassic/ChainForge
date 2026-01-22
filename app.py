# app.py (Pro Enhanced Version)
import streamlit as st
import pandas as pd
import numpy as np
from modules.whales import WhaleClient
from modules.ai_query import process_query

# Initialize Pro Clients
whale_client = WhaleClient()

with st.sidebar:
    st.header("⚡ Pro Intelligence")
    whale_threshold = st.slider("Min Whale Value ($)", 100000, 10000000, 500000)

# Inside the "Quant Suite" Tab...
def generate_ai_signal(df, pair, sentiment):
    last_row = df.iloc[-1]
    rsi = last_row['rsi']
    price = last_row['close']
    upper = last_row['upper_band']
    
    # Calculate "Distance to Bands"
    bb_status = "Touching Upper" if price >= upper else "Touching Lower" if price <= last_row['lower_band'] else "Neutral"
    
    context = {
        "pair": pair,
        "rsi": round(rsi, 2),
        "bb_status": bb_status,
        "sentiment": sentiment,
        "volatility": "High" if last_row['std'] > df['std'].mean() else "Stable"
    }
    
    # Professional Confluence Query
    query = f"Provide a confluence analysis for {pair}. RSI is {context['rsi']} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}). Price is {bb_status} Bollinger Band. Recent News Sentiment: {sentiment}. Is there a trade setup?"
    
    return process_query(query, context)

# Whale Tracker Tab
with tab_whales:
    st.subheader(f"🐋 Live Whale Movements (>${whale_threshold:,.0f})")
    whales = whale_client.get_recent_whales(min_value=whale_threshold)
    
    if whales:
        w_df = pd.DataFrame(whales)
        st.dataframe(w_df.style.applymap(
            lambda x: "color: #ff4b4b" if "Bearish" in str(x) else "color: #00ff00" if "Bullish" in str(x) else "",
            subset=['impact']
        ), use_container_width=True)
    else:
        st.info("No massive whale movements detected in the last window or API key missing.")
