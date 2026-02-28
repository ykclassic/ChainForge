import os
import sys
import ccxt
import pandas as pd
import requests
from datetime import datetime
import numpy as np

# --- 1. PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- 2. SECURE IMPORTS ---
try:
    import modules.sentiment as sentiment
    import modules.obi_engine as obi
except ImportError:
    try:
        import sentiment as sentiment
        import obi_engine as obi
    except ImportError:
        sentiment = None
        obi = None

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found.")
        return

    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 Launching Standard Analysis (Sensitivity: ACTIVE)")

    for pair in assets:
        try:
            # 3. Fetch Data (24h lookback for Standard)
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 4. Get Scores
            score_sentiment = sentiment.get_sentiment_score(pair) if sentiment else 0.0
            score_obi = obi.get_imbalance(pair) if obi else 0.0
            
            # 5. FIXED SCALAR EXTRACTION
            # Use .iloc for pandas or proper numpy indexing
            current_price = float(df['close'].iloc[-1])
            start_price = float(df['close'].iloc[0])
            
            trend = ((current_price - start_price) / start_price) * 100
            
            # DIAGNOSTIC LOG
            print(f"📊 {pair} RAW: Trend: {trend:+.2f}% | Sent: {score_sentiment:+.2f} | OBI: {score_obi:+.4f}")

            # 6. Decision Logic (Sensitivity Check)
            verdict = "NEUTRAL"
            color = 0x95a5a6
            
            if trend > 0.35:
                if score_sentiment > 0.5 or score_obi > 0.015:
                    verdict = "BUY"
                    color = 0x2ecc71
            elif trend < -0.35:
                if score_sentiment < -0.5 or score_obi < -0.015:
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
                            {"name": "Price", "value": f"${current_price:,.2f}", "inline": True},
                            {"name": "24h Trend", "value": f"{trend:+.2f}%", "inline": True},
                            {"name": "Sent / OBI", "value": f"{score_sentiment:+.1f} / {score_obi:+.3f}", "inline": True}
                        ],
                        "footer": {"text": "ChainForge Standard Engine • 2026 v2.7 (Fixed)"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral ({trend:+.2f}%)")

        except Exception as e:
            print(f"⚠️ Error processing {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
