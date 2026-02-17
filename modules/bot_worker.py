import os
import sys

# --- BOILERPLATE PATH FIX: Prevents ModuleNotFoundError in GitHub Actions ---
# This looks for files in the current folder (modules/) and the root directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import ccxt
import pandas as pd
import requests
from datetime import datetime

# Import local data processing modules (Maintaining your previous updates)
# We use a try-except block to handle both local and GitHub Runner paths.
try:
    # Use this if running as a package (python -m modules.bot_worker)
    import modules.sentiment_engine as sentiment
    import modules.obi_engine as obi
except ImportError:
    # Use this if running as a script (python modules/bot_worker.py)
    import sentiment_engine as sentiment
    import obi_engine as obi

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found in environment.")
        return

    # Initialize Bitget (Standard for 2026 workflows)
    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Launching Standard Analysis...")

    for pair in assets:
        try:
            # 1. Fetch Market Data (24h Window)
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 2. Get Institutional Data (Sentiment & OBI)
            # These functions should exist in your sentiment_engine and obi_engine files
            score_sentiment = sentiment.get_score(pair)
            score_obi = obi.get_imbalance(pair)
            
            # 3. Decision Logic (Calculated, No Hardcoding)
            current_price = df['close'].iloc[-1]
            price_change_pct = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            verdict = "NEUTRAL"
            color = 0x95a5a6 # Gray
            
            # Signal Criteria: Convergence of Trend + Sentiment + Imbalance
            if price_change_pct > 0.5 and score_sentiment > 0.1 and score_obi > 0.1:
                verdict = "BUY"
                color = 0x2ecc71 # Green
            elif price_change_pct < -0.5 and score_sentiment < -0.1 and score_obi < -0.1:
                verdict = "SELL"
                color = 0xe74c3c # Red

            # 4. Dispatch to Discord
            if verdict != "NEUTRAL":
                print(f"🚀 {pair}: {verdict} Signal Generated!")
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
                        "footer": {"text": "ChainForge Standard Engine • 2026 v2.1"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral (Trend: {price_change_pct:+.2f}%)")

        except Exception as e:
            print(f"⚠️  Failed to process {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
