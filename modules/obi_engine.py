import ccxt
import numpy as np

def get_imbalance(pair):
    """
    Calculates the Order Book Imbalance (OBI).
    Formula: (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
    Output: -1.0 (Heavy Sell Pressure) to +1.0 (Heavy Buy Pressure)
    """
    try:
        # Initialize exchange inside the function for fresh connection
        exchange = ccxt.bitget()
        
        # Fetch the order book (Top 20 levels)
        order_book = exchange.fetch_order_book(pair, limit=20)
        
        bids = order_book['bids'] # [[price, size], ...]
        asks = order_book['asks']
        
        if not bids or not asks:
            return 0.0

        # Sum up the volumes (quantity) of the top 20 levels
        total_bid_volume = sum([bid[1] for bid in bids])
        total_ask_volume = sum([ask[1] for ask in asks])
        
        # Calculate Imbalance
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        return round(float(imbalance), 4)

    except Exception as e:
        print(f"⚠️ OBI Engine Error for {pair}: {e}")
        return 0.0

if __name__ == "__main__":
    # Quick local test
    print(f"Test OBI (BTC/USDT): {get_imbalance('BTC/USDT')}")
