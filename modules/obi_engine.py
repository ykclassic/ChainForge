import ccxt

def get_imbalance(pair):
    try:
        exchange = ccxt.bitget()
        # Fetch 20 levels - enough for a precise institutional view
        order_book = exchange.fetch_order_book(pair, limit=20)
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0

        weighted_bids = 0.0
        weighted_asks = 0.0

        # Corrected Weighting: volume / (index + 1)
        for i in range(len(bids)):
            price, volume = bids[i]
            weighted_bids += volume / (i + 1)

        for i in range(len(asks)):
            price, volume = asks[i]
            weighted_asks += volume / (i + 1)
        
        total_v = weighted_bids + weighted_asks
        if total_v == 0: return 0.0
        
        imbalance = (weighted_bids - weighted_asks) / total_v
        return round(float(imbalance), 4)

    except Exception as e:
        print(f"⚠️ OBI Engine Error for {pair}: {e}")
        return 0.0
