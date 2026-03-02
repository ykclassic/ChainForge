import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# --- TFT Architecture Components ---

class GatedLinearUnit(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.elu = layers.Dense(units, activation="elu")
        self.dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.projector = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projector = layers.Dense(self.units)
        super(GatedResidualNetwork, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.elu(inputs)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.glu(x)
        residual = self.projector(inputs) if self.projector else inputs
        return self.layer_norm(x + residual)

# --- Elite Signal Logic ---

def generate_pro_signal(df, sentiment=0.0, imbalance=0.0):
    """
    Elite Engine: Uses TFT-Lite patterns and Institutional Order Flow.
    """
    try:
        lookback = 50 
        if len(df) < lookback:
            return {"verdict": "NEUTRAL", "trend_raw": 0.0}

        # 1. "UNBREAKABLE" SCALAR EXTRACTION
        # Matching the successful bot_worker.py logic
        prices_list = df['close'].tolist()
        current_price = float(prices_list[-1])
        start_price = float(prices_list)
        
        # 2. Pattern Recognition
        df_proc = df.copy()
        df_proc['returns'] = df_proc['close'].pct_change()
        raw_data = df_proc[['returns', 'vol']].tail(lookback).fillna(0).values
        input_tensor = tf.expand_dims(tf.convert_to_tensor(raw_data, dtype=tf.float32), 0)

        # Initialize and run through GRN
        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        # Calculate Hybrid Trend
        ai_trend = float(tf.reduce_mean(processed[:, -5:, :]).numpy())
        price_velocity = (current_price - start_price) / start_price
        combined_trend = (ai_trend * 0.2) + (price_velocity * 0.8)

        # 3. FIXED DECISION LOGIC
        verdict = "NEUTRAL"
        
        if combined_trend > 0.005: # Strong Upward Momentum
            if imbalance > 0.015 or sentiment > 0.05:
                verdict = "BUY"
                
        elif combined_trend < -0.005: # Strong Downward Momentum
            # FIXED: Indented block added here to resolve IndentationError
            if imbalance < -0.015 or sentiment < -0.05:
                verdict = "SELL"

        # 4. VOLATILITY PROTECTION (ATR)
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().tolist()[-1] # List-native for safety

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
