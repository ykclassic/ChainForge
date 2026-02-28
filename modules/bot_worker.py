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
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK secret not found.")
        return

    exchange = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 Launching Standard Analysis (Sensitivity: ACTIVE)")

    for pair in assets:
        try:
            # 1. Fetch Data
            ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 2. Get Scores
            score_sentiment = sentiment.get_sentiment_score(pair)
            score_obi = obi.get_imbalance(pair)
            
            # 3. FIX: Extract values using .values[-1] to avoid _iLocIndexer issues
            # This extracts the raw NumPy scalar, which is 100% float-compatible
            close_prices = df['close'].values
            current_price = float(close_prices[-1])
            start_price = float(close_prices)
            
            trend = ((current_price - start_price) / start_price) * 100
            
            # DIAGNOSTIC LOG
            print(f"📊 {pair} RAW: Trend: {trend:+.2f}% | Sent: {score_sentiment:+.2f} | OBI: {score_obi:+.4f}")

            # 4. Decision Logic
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
                        "footer": {"text": "ChainForge Standard Engine • 2026 v2.4 (Native Scalars)"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral ({trend:+.2f}%)")

        except Exception as e:
            print(f"⚠️ Error {pair}: {str(e)}")

if __name__ == "__main__":
    run_standard_engine()
