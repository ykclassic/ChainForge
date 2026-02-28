import ccxt
import numpy as np

def get_imbalance(pair):
    try:
        exchange = ccxt.bitget()
        # Fetch 30 levels for a deeper view of institutional walls
        order_book = exchange.fetch_order_book(pair, limit=30)
        
        bids = order_book['bids'] 
        asks = order_book['asks']
        
        if not bids or not asks:
            return 0.0

        # Weighted Volume: Orders at the top of the book count for more
        # Formula: volume * (1 / level_index)
        weighted_bids = sum([bid / (i + 1) for i, bid in enumerate(bids)])
        weighted_asks = sum([ask / (i + 1) for i, ask in enumerate(asks)])
        
        imbalance = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks)
        
        return round(float(imbalance), 4)

    except Exception as e:
        print(f"⚠️ OBI Engine Error for {pair}: {e}")
        return 0.0
