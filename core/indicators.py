import numpy as np
import pandas as pd

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / (loss + 1e-9)

    return 100 - (100 / (1 + rs))


def vwap(df):

    return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()


def volatility(series):

    return series.pct_change().std()


def market_regime(df, period=14):

    tr = pd.DataFrame()

    tr["h-l"] = df["high"] - df["low"]
    tr["h-pc"] = abs(df["high"] - df["close"].shift())
    tr["l-pc"] = abs(df["low"] - df["close"].shift())

    tr["tr"] = tr.max(axis=1)

    atr = tr["tr"].rolling(period).sum()

    high = df["high"].rolling(period).max()
    low = df["low"].rolling(period).min()

    chop = 100 * np.log10(atr / (high - low)) / np.log10(period)

    val = chop.iloc[-1]

    if val < 38:
        return "TRENDING"

    if val > 61:
        return "RANGING"

    return "TRANSITIONAL"
