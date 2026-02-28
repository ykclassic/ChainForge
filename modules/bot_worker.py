import os
import sys
import ccxt
import pandas as pd
import requests
from datetime import datetime

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import modules.sentiment as sentiment
    import modules.obi_engine as obi
except ImportError:
    import sentiment as sentiment
    import obi_engine as obi

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 Launching Standard Analysis (Sensitivity: ACTIVE)")

    for pair in assets:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Fetch Scores
            score_sentiment = sentiment.get_sentiment_score(pair)
            score_obi = obi.get_imbalance(pair)
            
            # Calculate Trend
            current_price = df['close'].iloc[-1]
            trend = ((current_price - df['close'].iloc) / df['close'].iloc) * 100
            
            # --- DIAGNOSTIC LOG (The most important part for you right now) ---
            print(f"📊 {pair} RAW DATA: Trend: {trend:+.2f}% | Sent: {score_sentiment:+.2f} | OBI: {score_obi:+.4f}")

            # --- UPDATED "ACTIVE" LOGIC ---
            # We lowered the OBI requirement and smoothed sentiment
            verdict = "NEUTRAL"
            color = 0x95a5a6
            
            # BUY Criteria: Price up + (Strong Sentiment OR Strong OBI)
            if trend > 0.35:
                if score_sentiment > 0.5 or score_obi > 0.015:
                    verdict = "BUY"
                    color = 0x2ecc71
            
            # SELL Criteria: Price down + (Weak Sentiment OR Weak OBI)
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
                            {"name": "Sent/OBI", "value": f"{score_sentiment:+.1f} / {score_obi:+.3f}", "inline": True}
                        ],
                        "footer": {"text": "ChainForge Standard • Adjusted Sensitivity"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                # Log why it stayed neutral
                reason = "Wait for OBI/Sentiment convergence"
                print(f"⏸️  {pair}: {reason}")

        except Exception as e:
            print(f"⚠️  Error {pair}: {e}")

if __name__ == "__main__":
    run_standard_engine()
