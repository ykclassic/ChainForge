import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

def calculate_atr(df, period=14):
    """Calculates ATR to set volatility-adjusted exits."""
    high_low = df['high'] - df['low']
    high_pc = np.abs(df['high'] - df['close'].shift())
    low_pc = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

class GatedResidualNetwork(layers.Layer):
    """TFT component that automatically skips unnecessary complexity."""
    def __init__(self, units):
        super().__init__()
        self.dense1 = layers.Dense(units, activation='elu')
        self.dense2 = layers.Dense(units)
        self.gate = layers.Dense(units, activation='sigmoid')
        self.norm = layers.LayerNormalization()

    def call(self, x):
        h = self.dense2(self.dense1(x))
        return self.norm(x + (self.gate(x) * h))

def build_tft_lite(window=10, features=2):
    """Lightweight TFT for GitHub Action runners."""
    inputs = layers.Input(shape=(window, features))
    x = GatedResidualNetwork(32)(inputs)
    # Multi-Head Attention allows the bot to 'attend' to specific past spikes
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.Flatten()(layers.LayerNormalization()(x))
    out = layers.Dense(1)(x)
    return Model(inputs, out)

def generate_pro_signal(df, sentiment, imbalance):
    current_price = df['close'].iloc[-1]
    atr = calculate_atr(df)
    
    # Prep data (Price + Volume)
    data = df[['close', 'vol']].tail(10).values
    norm_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
    
    model = build_tft_lite()
    model.compile(optimizer='adam', loss='mse')
    # Inference is fast enough for GitHub Actions
    pred = model.predict(norm_data.reshape(1, 10, 2), verbose=0)[0][0]
    
    # ATR Multipliers: 1.5 for protection, 3.5 for profit capture
    sl_dist = atr * 1.5
    tp_dist = atr * 3.5
    
    verdict = "NEUTRAL"
    if sentiment > 0.15 and imbalance > 0.2: verdict = "BUY"
    elif sentiment < -0.15 and imbalance < -0.2: verdict = "SELL"
    
    return {
        "verdict": verdict,
        "entry": current_price,
        "stop_loss": current_price - sl_dist if verdict == "BUY" else current_price + sl_dist,
        "take_profit": current_price + tp_dist if verdict == "BUY" else current_price - tp_dist,
        "atr": round(atr, 2)
    }
