import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# --- 1. MODULAR ML ENGINE ---
def run_ml_forecast(df):
    """Trains a Random Forest model on historical OHLCV + RSI data."""
    # Feature Engineering
    data = df[['close', 'rsi', 'vol']].copy()
    data['target'] = data['close'].shift(-1)  # Target: Next day's price
    
    # Use Lags as features (Price memory)
    data['lag_1'] = data['close'].shift(1)
    data['lag_2'] = data['close'].shift(2)
    
    train_df = data.dropna()
    X = train_df[['close', 'rsi', 'vol', 'lag_1', 'lag_2']]
    y = train_df['target']
    
    # Train Random Forest (Institutional Standard for Baseline ML)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict Tomorrow
    last_row = data[['close', 'rsi', 'vol', 'lag_1', 'lag_2']].iloc[-1:].fillna(method='ffill')
    prediction = model.predict(last_row)[0]
    
    # Feature Importance (Why is the model picking this price?)
    importance = dict(zip(X.columns, model.feature_importances_))
    
    return round(prediction, 2), importance

# --- 2. CORE DATA UTILITIES ---
@st.cache_data(ttl=300)
def fetch_pro_data(pair, tf='1d', limit=150):
    exchange = ccxt.bitget({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(pair, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # Indicators (Essential for both Backtest and ML)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    return df.dropna()

# --- 3. MAIN INTERFACE ---
st.set_page_config(page_title="ChainForge Pro", layout="wide")

# Persistent State Management
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

def main():
    st.title("⚡ ChainForge Pro: Intelligence Suite")
    
    # Global Sidebar
    with st.sidebar:
        st.header("Global Config")
        with st.form("settings"):
            selected_pair = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT"])
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "BNB", "XRP"], default=["BTC", "ETH"])
            update = st.form_submit_button("Sync App State")

    # Data Sync
    df = fetch_pro_data(selected_pair, timeframe)

    # 4. TABS: ALL UPDATES MAINTAINED
    tab_pulse, tab_charts, tab_ml, tab_strategy = st.tabs([
        "📡 Market Pulse", "📈 Quant Charts", "🤖 ML Forecast", "🧪 Strategy Lab"
    ])

    with tab_pulse:
        # Watchlist Logic
        st.subheader("Real-time Pulse")
        cols = st.columns(len(watchlist))
        tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
        for i, coin in enumerate(watchlist):
            t = tickers.get(f"{coin}/USDT", {})
            cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")

    with tab_charts:
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ml:
        st.subheader("Random Forest Price Prediction")
        if st.button("🔥 Run ML Training Pipeline"):
            with st.spinner("Training model on historical volatility..."):
                pred, importance = run_ml_forecast(df)
                st.session_state.ml_results = {"pred": pred, "importance": importance}

        if st.session_state.ml_results:
            res = st.session_state.ml_results
            curr_price = df['close'].iloc[-1]
            diff_pct = ((res['pred'] - curr_price) / curr_price) * 100
            
            c1, c2 = st.columns(2)
            c1.metric("Predicted (24h)", f"${res['pred']:,}", f"{diff_pct:+.2f}%")
            
            # Show "Why" the model made this choice
            with c2:
                st.write("**Model Feature Weighting**")
                st.bar_chart(pd.Series(res['importance']))

    with tab_strategy:
        st.info("Strategy Lab: Ready for Backtesting Engine integration.")
        # run_backtest(df) call would go here

if __name__ == "__main__":
    main()
