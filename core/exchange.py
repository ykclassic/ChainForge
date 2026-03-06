import ccxt

class ExchangeClient:

    def __init__(self, exchange="bitget"):
        if exchange == "bitget":
            self.exchange = ccxt.bitget({"enableRateLimit": True})
        else:
            raise ValueError("Unsupported exchange")

    def fetch_ohlcv(self, pair, timeframe="1h", limit=200):
        return self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

    def fetch_order_book(self, pair, depth=20):
        return self.exchange.fetch_order_book(pair, limit=depth)

    def fetch_tickers(self, pairs):
        return self.exchange.fetch_tickers(pairs)
