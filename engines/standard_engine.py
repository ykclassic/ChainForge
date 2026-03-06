from core.exchange import ExchangeClient
from core.data_engine import ohlcv_to_df
from core.indicators import volatility
from modules.obi_engine import get_imbalance


class StandardEngine:

    def __init__(self):

        self.exchange = ExchangeClient()

    def analyze(self, pair):

        raw = self.exchange.fetch_ohlcv(pair,"1h",50)

        df = ohlcv_to_df(raw)

        trend = (
            df["close"].iloc[-1] - df["close"].iloc[0]
        ) / df["close"].iloc[0]

        obi = get_imbalance(pair)

        if trend > 0.004 and obi > 0.01:
            return "BUY"

        if trend < -0.004 and obi < -0.01:
            return "SELL"

        return "NEUTRAL"
