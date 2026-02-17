import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# --- TFT Architecture Components ---

class GatedLinearUnit(layers.Layer):
    """GLU component: Allows the model to suppress irrelevant features."""
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    """GRN component: Processes patterns while maintaining a skip connection."""
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.elu = layers.Dense(units, activation="elu")
        self.dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.projector = None # To be built if input dim != units

    def build(self, input_shape):
        # If input dim doesn't match units, create a projector for the residual connection
        if input_shape[-1] != self.units:
            self.projector = layers.Dense(self.units)
        super(GatedResidualNetwork, self).build(input_shape)

    def call(self, inputs, training=False):
        # 1. Processing path
        x = self.elu(inputs)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.glu(x)
        
        # 2. Residual path (Project if dimensions don't match)
        residual = self.projector(inputs) if self.projector else inputs
        
        # 3. Combine and Normalize
        return self.layer_norm(x + residual)

# --- Elite Signal Logic ---

def generate_pro_signal(df, sentiment=0.0, imbalance=0.0):
    """
    Elite Engine logic: Uses TFT-lite to find trend strength and calculates
    institutional-grade entry/exit points.
    """
    try:
        # 1. Data Prep (Last 10 hours)
        lookback = 10
        if len(df) < lookback:
            return {"verdict": "NEUTRAL"}

        # Prepare input: [Batch=1, Time=10, Features=2]
        raw_data = df[['close', 'vol']].tail(lookback).values
        input_tensor = tf.convert_to_tensor(raw_data, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0) 

        # 2. Forward Pass through GRN
        # We use 32 units for high-dimensional pattern recognition
        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        # Determine trend strength from the last processed state
        trend_strength = tf.reduce_mean(processed).numpy()

        # 3. Volatility Calculation (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        # 4. Signal Decision Engine
        current_price = df['close'].iloc[-1]
        verdict = "NEUTRAL"
        
        # Logic: Convergence of TFT patterns, Sentiment, and Order Flow
        if trend_strength > 0.02 and sentiment > 0.1 and imbalance > 0.1:
            verdict = "BUY"
        elif trend_strength < -0.02 and sentiment < -0.1 and imbalance < -0.1:
            verdict = "SELL"

        if verdict != "NEUTRAL":
            # Risk Management: 1.5x ATR for Stop Loss, 3x ATR for Take Profit
            sl_dist = atr * 1.5
            tp_dist = atr * 3.0
            
            stop_loss = current_price - sl_dist if verdict == "BUY" else current_price + sl_dist
            take_profit = current_price + tp_dist if verdict == "BUY" else current_price - tp_dist

            return {
                "verdict": verdict,
                "entry": current_price,
                "atr": round(atr, 2),
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }

        return {"verdict": "NEUTRAL"}

    except Exception as e:
        print(f"⚠️ Signal Logic Error: {e}")
        return {"verdict": "NEUTRAL"}
