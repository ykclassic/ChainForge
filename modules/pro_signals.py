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
    Elite Engine: Now with 50-hour Institutional Lookback.
    """
    try:
        # 1. Expanded Data Prep
        lookback = 50 
        if len(df) < lookback:
            return {"verdict": "NEUTRAL", "trend_raw": 0.0}

        df = df.copy()
        # Log returns are more stable for long lookbacks
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        # Relative volume vs 50-period average
        df['vol_norm'] = df['vol'] / df['vol'].rolling(window=lookback).mean()
        
        raw_data = df[['returns', 'vol_norm']].tail(lookback).fillna(0).values
        input_tensor = tf.convert_to_tensor(raw_data, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0) 

        # 2. Forward Pass
        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        # Calculate Trend Strength (Mean of recent momentum within the 50h window)
        trend_strength = tf.reduce_mean(processed[:, -5:, :]).numpy()

        # 3. Dynamic Thresholds (Slightly tighter due to 50h smoothing)
        is_strong_up = trend_strength > 0.004
        is_strong_down = trend_strength < -0.004

        # 4. Volatility Calculation (ATR 14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        # 5. Elite Decision Engine
        current_price = df['close'].iloc[-1]
        verdict = "NEUTRAL"
        
        # Convergence Logic (TFT + Sentiment/OBI Confirmation)
        if is_strong_up and (sentiment > 0.1 or imbalance > 0.02):
            verdict = "BUY"
        elif is_strong_down and (sentiment < -0.1 or imbalance < -0.02):
            verdict = "SELL"

        if verdict != "NEUTRAL":
            # Pro Strategy: 2.0x ATR Stop, 4.0x ATR Target (1:2 Risk/Reward)
            return {
                "verdict": verdict,
                "entry": current_price,
                "atr": round(float(atr), 2),
                "trend_raw": round(float(trend_strength), 4),
                "stop_loss": current_price - (atr * 2.0) if verdict == "BUY" else current_price + (atr * 2.0),
                "take_profit": current_price + (atr * 4.0) if verdict == "BUY" else current_price - (atr * 4.0)
            }

        return {"verdict": "NEUTRAL", "trend_raw": round(float(trend_strength), 4)}

    except Exception as e:
        print(f"⚠️ Pro Signal Error: {e}")
        return {"verdict": "NEUTRAL", "trend_raw": 0.0}
