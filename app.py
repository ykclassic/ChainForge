import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# --- 1. CORE ENGINES (Whales, ML, Backtest) ---

class WhaleClient:
    def get_recent_whales(self, min_value=500000):
        # Simulated logic for the terminal; would connect to Whale Alert API
        return [
            {"time": "12:05", "asset": "BTC", "amount": "$1.2M", "type": "Wallet to Exchange"},
            {"time": "11:42", "asset": "SOL", "amount": "$800k", "type": "Exchange to Wallet"}
        ]

def run_ml_forecast(df):
    """Trains a Random Forest model on technical features."""
    data = df[['close', 'rsi', 'vol']].copy()
    data['target'] = data['close'].shift(-1)
    data['lag_1'] = data['close'].shift(1)
    train_df = data.dropna()
    
    X = train_df[['close', 'rsi', 'vol', 'lag_1']]
    y = train_df['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_row = data[['close', 'rsi', 'vol', 'lag_1']].iloc[-1:]
    prediction = model.predict(last_row)[0]
    importance = dict(zip(X.columns, model.feature_importances_))
    return round(prediction, 2), importance

def run_backtest(data):
    """Simple Mean Reversion Backtest."""
    df = data.copy()
    df['signal'] = 0
    df.loc[(df['close'] <= df['lower_band']) & (df['rsi'] < 30), 'signal'] = 1
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod() * 10000
    
    total_return = round(((df['equity_curve'].iloc[-1] - 10000) / 10000) * 100, 2)
    return {"total_return": total_return, "df": df}

# --- 2. DATA PIPELINE ---

@st.cache_data(ttl=300)
def fetch_complete_data(pair, tf='1d'):
    exchange = ccxt.bitget({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(pair, tf, limit=100)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # Technical Indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    return df.dropna()

# --- 3. MAIN APP INTERFACE ---

st.set_page_config(page_title="ChainForge Pro", layout="wide")

# Persistent State
if 'ml_data' not in st.session_state: st.session_state.ml_data = None

def main():
    st.title("⚡ ChainForge Pro: Full Terminal")

    with st.sidebar:
        with st.form("main_settings"):
            st.header("Terminal Config")
            selected_pair = st.selectbox("Active Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT"])
            watchlist = st.multiselect("Watchlist", ["BTC", "ETH", "SOL", "BNB"], default=["BTC", "ETH"])
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
            st.form_submit_button("Sync Platform")

    # Fetch Data
    df = fetch_complete_data(selected_pair, timeframe)

    # Watchlist Row
    st.subheader("📡 Market Pulse")
    tickers = ccxt.bitget().fetch_tickers([f"{c}/USDT" for c in watchlist])
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        t = tickers.get(f"{coin}/USDT", {})
        cols[i].metric(coin, f"${t.get('last', 0):,.2f}", f"{t.get('percentage', 0):.2f}%")
    st.divider()

    # Tabs (All features restored)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Charts", "🐋 Whales", "🌡️ Correlation", "🧪 Backtest", "🤖 ML Forecast"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['ts'], y=df['upper_band'], name="Upper BB", line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['lower_band'], name="Lower BB", line=dict(color='gray', dash='dash')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.table(WhaleClient().get_recent_whales())

    with tab3:
        st.subheader("Watchlist Correlation")
        prices = pd.DataFrame()
        for c in watchlist:
            prices[c] = [x[4] for x in ccxt.bitget().fetch_ohlcv(f"{c}/USDT", '1d', limit=30)]
        fig_corr = px.imshow(prices.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        if st.button("Run Backtest Simulation"):
            res = run_backtest(df)
            st.metric("Total Profit", f"{res['total_return']}%")
            st.line_chart(res['df'].set_index('ts')['equity_curve'])

    with tab5:
        if st.button("🚀 Train & Predict"):
            pred, imp = run_ml_forecast(df)
            st.session_state.ml_data = {"pred": pred, "imp": imp}
        
        if st.session_state.ml_data:
            st.metric("ML Forecast (24h)", f"${st.session_state.ml_data['pred']:,}")
            st.bar_chart(pd.Series(st.session_state.ml_data['imp']))

if __name__ == "__main__":
    main()
