import os
import sys

# --- AGGRESSIVE PATH FIX: Forces root and modules into search path ---
# Get the absolute path of the directory containing this script (modules/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
root_dir = os.path.dirname(current_dir)

# Insert at the START of sys.path so these are checked before standard libs
for path in [current_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import ccxt
import pandas as pd
import requests
from datetime import datetime

# --- Module Import Handling ---
try:
    # Standard: Try importing as part of the modules package
    import modules.sentiment as sentiment
    import modules.obi_engine as obi
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback 1: Try importing directly if inside the modules folder
        import sentiment as sentiment
        import obi_engine as obi
    except (ImportError, ModuleNotFoundError):
        # Fallback 2: Check if your file is actually named 'obi.py'
        import sentiment as sentiment
        import obi as obi  # Attempting 'obi' instead of 'obi_engine'

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found in environment.")
        return

    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Launching Standard Analysis...")

    for pair in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Using the functions defined in your engines
            score_sentiment = sentiment.get_sentiment_score(pair)
            
            # Use get_imbalance if it exists, otherwise default to 0.0
            score_obi = getattr(obi, 'get_imbalance', lambda x: 0.0)(pair)
            
            current_price = df['close'].iloc[-1]
            price_change_pct = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            verdict = "NEUTRAL"
            color = 0x95a5a6

            if price_change_pct > 0.5 and score_sentiment > 1.0 and score_obi > 0.1:
                verdict = "BUY"
                color = 0x2ecc71
            elif price_change_pct < -0.5 and score_sentiment < -1.0 and score_obi < -0.1:
                verdict = "SELL"
                color = 0xe74c3c

            if verdict != "NEUTRAL":
                payload = {
                    "username": "ChainForge Standard",
                    "embeds": [{
                        "title": f"📊 {verdict} Signal | {pair}",
                        "color": color,
                        "fields": [
                            {"name": "Current Price", "value": f"${current_price:,.2f}", "inline": True},
                            {"name": "24h Trend", "value": f"{price_change_pct:+.2f}%", "inline": True},
                            {"name": "Sentiment Score", "value": f"{score_sentiment:+.2f}", "inline": True},
                            {"name": "Order Book Imbalance", "value": f"{score_obi:+.2f}", "inline": True}
                        ],
                        "footer": {"text": "ChainForge Standard Engine • 2026 Path-Guard Active"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral")

        except Exception as e:
            print(f"⚠️  Failed to process {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
