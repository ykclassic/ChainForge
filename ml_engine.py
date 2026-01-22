import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# --- 1. THE ML ENGINE (Non-rerunning logic) ---
def train_predict_model(df):
    """
    Trains a simple Linear Regression model on price & RSI.
    Returns: Prediction for the next period and the model's confidence.
    """
    # Prepare data for training
    data = df[['close', 'rsi']].copy()
    data['target'] = data['close'].shift(-1) # We want to predict tomorrow's price
    train_data = data.dropna()
    
    X = train_data[['close', 'rsi']]
    y = train_data['target']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the NEXT value using the very last row
    last_features = df[['close', 'rsi']].iloc[-1:].values
    prediction = model.predict(last_features)[0]
    
    # Calculate simple "Confidence" (R^2 Score)
    confidence = model.score(X, y) * 100
    return round(prediction, 2), round(confidence, 1)

# --- 2. MAIN APP START ---
st.set_page_config(page_title="ChainForge Pro", layout="wide")

# Persistent ML State
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = 0

def main():
    st.title("⚡ ChainForge Pro: Machine Learning Suite")

    # Shared Sidebar with st.form (Prevents global reruns)
    with st.sidebar:
        with st.form("ml_config"):
            st.header("Model Configuration")
            pair = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            lookback = st.slider("Training Lookback (Days)", 50, 200, 100)
            submitted = st.form_submit_button("Re-train & Update App")

    # Data Fetching
    exchange = ccxt.bitget()
    ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=lookback)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # Ensure indicators exist for ML features
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df = df.dropna()

    tab_viz, tab_ml_report = st.tabs(["📈 Analysis", "🤖 Prediction Engine"])

    with tab_viz:
        fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 3. ML TAB (Using st.fragment to isolate predictions)
    with tab_ml_report:
        st.subheader("Machine Learning Price Forecast")
        
        if st.button("Generate Tomorrow's Forecast"):
            with st.spinner("Analyzing market patterns..."):
                # Call our ML Engine
                pred, conf = train_predict_model(df)
                st.session_state.prediction = pred
                st.session_state.confidence = conf

        if st.session_state.prediction:
            col1, col2 = st.columns(2)
            current_price = df['close'].iloc[-1]
            diff = st.session_state.prediction - current_price
            
            col1.metric("Predicted Price (24h)", f"${st.session_state.prediction:,}", 
                        delta=f"{diff:+.2f} ({ (diff/current_price)*100 :.2f}%)")
            
            col2.metric("Model Confidence (R²)", f"{st.session_state.confidence}%")
            
            st.success(f"**Analysis:** The model expects {pair} to move toward **${st.session_state.prediction}** based on the correlation between Price and RSI over the last {lookback} days.")
            
            # Prediction Visualization
            future_date = df['ts'].iloc[-1] + timedelta(days=1)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df['ts'][-10:], y=df['close'][-10:], name="Actual Price", line=dict(color="cyan")))
            fig_pred.add_trace(go.Scatter(x=[df['ts'].iloc[-1], future_date], 
                                          y=[df['close'].iloc[-1], st.session_state.prediction], 
                                          name="ML Forecast", line=dict(color="orange", dash="dash")))
            fig_pred.update_layout(template="plotly_dark", title="Trend Projection")
            st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()
