import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ... (Keep GatedLinearUnit and GatedResidualNetwork classes) ...

def generate_pro_signal(df, sentiment=0.0, imbalance=0.0):
    try:
        lookback = 50 
        if len(df) < lookback:
            return {"verdict": "NEUTRAL", "trend_raw": 0.0}

        # 1. Use the working iloc logic
        current_price = float(df['close'].iloc[-1]) 
        start_price = float(df['close'].iloc)
        
        # 2. Pattern Recognition (TFT-Lite)
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        raw_data = df[['returns', 'vol']].tail(lookback).fillna(0).values
        input_tensor = tf.expand_dims(tf.convert_to_tensor(raw_data, dtype=tf.float32), 0)

        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        # Extract scalar trend from tensor
        ai_trend = float(tf.reduce_mean(processed[:, -5:, :]).numpy())
        price_velocity = (current_price - start_price) / start_price
        
        # Hybrid Trend (Weighted)
        combined_trend = (ai_trend * 0.2) + (price_velocity * 0.8)

        # 3. Decision Logic (Matching your working Standard logic but for Pro)
        verdict = "NEUTRAL"
        
        # BUY: Trend is up + (OBI is positive OR Sentiment is positive)
        if combined_trend > 0.005: # ~0.5%
            if imbalance > 0.01 or sentiment > 0.1:
                verdict = "BUY"
        # SELL: Trend is down + (OBI is negative OR Sentiment is negative)
        elif combined_trend < -0.005:
            # In pro_signals.py logic:
if imbalance < -0.015 or sentiment < -0.05: # Catching subtler shifts
    verdict = "SELL"

        # 4. Volatility-Based Stops (ATR)
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]

        return {
            "verdict": verdict,
            "entry": current_price,
            "atr": round(float(atr), 2),
            "trend_raw": round(combined_trend, 6),
            "stop_loss": current_price - (atr * 2) if verdict == "BUY" else current_price + (atr * 2),
            "take_profit": current_price + (atr * 4) if verdict == "BUY" else current_price - (atr * 4)
        }

    except Exception as e:
        print(f"⚠️ Pro Logic Error: {e}")
        return {"verdict": "NEUTRAL", "trend_raw": 0.0}
