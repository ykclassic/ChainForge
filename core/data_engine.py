import pandas as pd

def ohlcv_to_df(data):

    df = pd.DataFrame(
        data,
        columns=["ts","open","high","low","close","volume"]
    )

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    return df
