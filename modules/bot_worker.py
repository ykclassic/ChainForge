import os
import sys
import ccxt
import pandas as pd
import requests
from datetime import datetime

# --- DIAGNOSTIC & PATH FIXING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Ensure both modules/ and root are in path
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"📂 [DIAGNOSTIC] Modules directory content: {os.listdir(current_dir)}")

# --- DYNAMIC IMPORT LOGIC ---
sentiment = None
obi = None

# 1. Import Sentiment (Confirmed working previously)
try:
    import modules.sentiment as sentiment
except ImportError:
    import sentiment as sentiment

# 2. Import OBI (The failing component)
# This loop tries every possible naming convention to find your OBI file
obi_found = False
for module_name in ['modules.obi_engine', 'obi_engine', 'modules.obi', 'obi']:
    try:
        if 'modules.' in module_name:
            obi = __import__(module_name, fromlist=['*'])
        else:
            obi = __import__(module_name)
        print(f"✅ Successfully loaded OBI via: {module_name}")
        obi_found = True
        break
    except ImportError:
        continue

if not obi_found:
    print("❌ CRITICAL: Could not find any OBI module (tried obi_engine.py, obi.py).")
    # We define a dummy class so the script doesn't crash
    class DummyOBI:
        @staticmethod
        def get_imbalance(pair): return 0.0
    obi = DummyOBI()

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found.")
        return

    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Launching Standard Analysis...")

    for pair in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Use getattr for safety in case the function name is different
            score_sentiment = getattr(sentiment, 'get_sentiment_score', lambda x: 0.0)(pair)
            score_obi = getattr(obi, 'get_imbalance', lambda x: 0.0)(pair)
            
            current_price = df['close'].iloc[-1]
            price_change_pct = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            verdict = "NEUTRAL"
            color = 0x95a5a6

            # Signal Thresholds (Calculated)
            if price_change_pct > 0.5 and score_sentiment > 1.0 and score_obi > 0.1:
                verdict = "BUY"
                color = 0x2ecc71
            elif price_change_pct < -0.5 and score_sentiment < -1.0 and score_obi < -0.1:
                verdict = "SELL"
                color = 0xe74c3c

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
                        "footer": {"text": "ChainForge Standard Engine • 2026 v2.2"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral ({price_change_pct:+.2f}%)")

        except Exception as e:
            print(f"⚠️  Failed to process {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
