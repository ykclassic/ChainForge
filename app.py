import streamlit as st

from core.exchange import ExchangeClient
from core.data_engine import ohlcv_to_df
from core.indicators import rsi, vwap
from ai.lstm_model import LSTMForecaster


exchange = ExchangeClient()

st.title("⚡ ChainForge Unified Terminal")

pair = st.selectbox(
    "Asset",
    ["BTC/USDT","ETH/USDT","SOL/USDT"]
)

raw = exchange.fetch_ohlcv(pair,"1h",200)

df = ohlcv_to_df(raw)

df["rsi"] = rsi(df["close"])
df["vwap"] = vwap(df)

st.line_chart(df["close"])

if st.button("AI Forecast"):

    model = LSTMForecaster()

    model.train(df["close"])

    pred = model.predict(df["close"])

    st.metric("Next Prediction", round(pred,2))
