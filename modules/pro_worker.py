import os
import sys
import ccxt
import pandas as pd
import requests
from datetime import datetime

# --- FORCED PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# Import Engines
from pro_signals import generate_pro_signal
try:
    import sentiment as sentiment_engine
    import obi_engine as obi
except ImportError:
    # Fallback if imported via modules.
    import modules.sentiment as sentiment_engine
    import modules.obi_engine as obi

def run_pro_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK not found.")
        return

    ex = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"💎 Elite Engine: Launching Real-Time TFT Analysis...")

    for pair in assets:
        try:
            # Fetch data
            ohlcv = ex.fetch_ohlcv(pair, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Fetch Real Institutional Data (NOT Simulated)
            real_sentiment = sentiment_engine.get_sentiment_score(pair)
            real_imbalance = obi.get_imbalance(pair)
            
            # Generate Signal
            sig = generate_pro_signal(df, sentiment=real_sentiment, imbalance=real_imbalance)

            # Diagnostic
            print(f"📊 {pair} | Trend: {sig.get('trend_raw')} | Sent: {real_sentiment} | OBI: {real_imbalance}")

            if sig['verdict'] != "NEUTRAL":
                payload = {
                    "username": "ChainForge PRO",
                    "embeds": [{
                        "title": f"🚀 ELITE {sig['verdict']} | {pair}",
                        "color": 0x2ecc71 if sig['verdict'] == "BUY" else 0xe74c3c,
                        "fields": [
                            {"name": "Entry Price", "value": f"${sig['entry']:,.2f}", "inline": True},
                            {"name": "Volatility (ATR)", "value": f"${sig['atr']}", "inline": True},
                            {"name": "Confidence", "value": f"{abs(sig['trend_raw'])*100:.2f}%", "inline": True},
                            {"name": "Elite Stop-Loss", "value": f"**${sig['stop_loss']:,.2f}**", "inline": False},
                            {"name": "Elite Take-Profit", "value": f"**${sig['take_profit']:,.2f}**", "inline": False},
                        ],
                        "footer": {"text": f"TFT-Lite • Sentiment: {real_sentiment} • OBI: {real_imbalance}"}
                    }]
                }
                requests.post(webhook_url, json=payload)
                print(f"🔥 {pair}: {sig['verdict']} Signal Dispatched.")
            else:
                print(f"⏸️ {pair}: Neutral")

        except Exception as e:
            print(f"⚠️ Worker Error for {pair}: {e}")

if __name__ == "__main__":
    run_pro_engine()
