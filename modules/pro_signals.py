import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ... (Keep GatedLinearUnit and GatedResidualNetwork classes as they were) ...

def generate_pro_signal(df, sentiment=0.0, imbalance=0.0):
    try:
        lookback = 50 
        if len(df) < lookback:
            return {"verdict": "NEUTRAL", "trend_raw": 0.0}

        df = df.copy()
        # Calculate Returns
        df['returns'] = df['close'].pct_change()
        
        # 1. AI Trend Component
        raw_data = df[['returns', 'vol']].tail(lookback).fillna(0).values
        input_tensor = tf.convert_to_tensor(raw_data, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0) 

        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        # 2. Hybrid Momentum (AI Pattern + Raw Price Action)
        # This prevents the "0.0" result by looking at actual price velocity
        ai_strength = tf.reduce_mean(processed[:, -5:, :]).numpy()
        price_velocity = df['returns'].tail(5).mean()
        
        # Combined Trend Strength
        trend_strength = (ai_strength * 0.3) + (price_velocity * 0.7)

        # 3. Dynamic Thresholds
        # If trend_strength is non-zero, this will trigger
        is_strong_up = trend_strength > 0.0005 
        is_strong_down = trend_strength < -0.0005

        # 4. Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        current_price = df['close'].iloc[-1]
        verdict = "NEUTRAL"
        
        # 5. Convergence Logic
        # We only need one confirmation (Sentiment OR OBI) if trend is present
        if is_strong_up and (sentiment > 0.05 or imbalance > 0.01):
            verdict = "BUY"
        elif is_strong_down and (sentiment < -0.05 or imbalance < -0.01):
            verdict = "SELL"

        return {
            "verdict": verdict,
            "entry": current_price,
            "atr": round(float(atr), 2),
            "trend_raw": round(float(trend_strength), 6),
            "stop_loss": current_price - (atr * 2.0) if verdict == "BUY" else current_price + (atr * 2.0),
            "take_profit": current_price + (atr * 4.0) if verdict == "BUY" else current_price - (atr * 4.0)
        }

    except Exception as e:
        print(f"⚠️ Pro Signal Error: {e}")
        return {"verdict": "NEUTRAL", "trend_raw": 0.0}
