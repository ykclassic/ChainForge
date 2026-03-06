import numpy as np

def calculate_regime(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = np.maximum.reduce([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ])

    atr = tr.rolling(period).sum()

    diff = high.rolling(period).max() - low.rolling(period).min()

    chop = 100 * np.log10(atr / diff.replace(0,np.nan)) / np.log10(period)

    val = chop.iloc[-1]

    if val < 38:
        return "TRENDING"

    if val > 61:
        return "RANGING"

    return "TRANSITION"
