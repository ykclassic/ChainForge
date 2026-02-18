import os
import sys

# --- BOILERPLATE PATH FIX ---
current_file_path = os.path.abspath(__file__)
modules_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(modules_dir)

# Priority 0: The modules folder itself
sys.path.insert(0, modules_dir)
# Priority 1: The root project folder
sys.path.insert(0, project_root)

import ccxt
import pandas as pd
import requests
from datetime import datetime

# Corrected Imports to match your actual filenames
try:
    # Try importing as a package member
    import modules.sentiment as sentiment
    import modules.obi_engine as obi
except (ImportError, ModuleNotFoundError):
    # Fallback for direct execution
    import sentiment as sentiment
    import obi_engine as obi

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found in environment.")
        return

    # Initialize Bitget
    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Launching Standard Analysis...")

    for pair in assets:
        try:
            # 1. Fetch Market Data (24h Window)
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 2. Get Institutional Data
            # Note: Changed sentiment.get_score to get_sentiment_score to match your file
            score_sentiment = sentiment.get_sentiment_score(pair)
            
            # Ensure obi_engine has a get_imbalance function
            try:
                score_obi = obi.get_imbalance(pair)
            except AttributeError:
                score_obi = 0.0 # Safety fallback
            
            # 3. Decision Logic
            current_price = df['close'].iloc[-1]
            price_change_pct = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            verdict = "NEUTRAL"
            color = 0x95a5a6 # Gray
            
            # Standard thresholds
            if price_change_pct > 0.5 and score_sentiment > 1.0 and score_obi > 0.1:
                verdict = "BUY"
                color = 0x2ecc71 # Green
            elif price_change_pct < -0.5 and score_sentiment < -1.0 and score_obi < -0.1:
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
                        "footer": {"text": "ChainForge Standard Engine • 2026 Path Verified"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral (Trend: {price_change_pct:+.2f}%)")

        except Exception as e:
            print(f"⚠️  Failed to process {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
