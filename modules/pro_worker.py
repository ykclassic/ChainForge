import os
import sys

# --- FORCED PATH INJECTION ---
# We get the absolute path of 'modules' and the project 'root'
current_file_path = os.path.abspath(__file__)
modules_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(modules_dir)

# We insert them at the very beginning of the search list (index 0)
sys.path.insert(0, modules_dir)
sys.path.insert(0, project_root)

import ccxt
import pandas as pd
import requests

# Now we import directly. Since modules_dir is in sys.path[0], 
# Python will see pro_signal.py immediately.
from pro_signal import generate_pro_signal

def run_pro_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK not found.")
        return

    ex = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"💎 Elite Engine: Initializing Path-Safe Analysis...")

    for pair in assets:
        try:
            ohlcv = ex.fetch_ohlcv(pair, '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Simulated institutional inputs for the TFT logic
            sig = generate_pro_signal(df, sentiment=0.15, imbalance=0.20)

            if sig['verdict'] != "NEUTRAL":
                print(f"🔥 {pair}: {sig['verdict']} Signal Found")
                payload = {
                    "username": "ChainForge PRO",
                    "embeds": [{
                        "title": f"🚀 ELITE {sig['verdict']} | {pair}",
                        "color": 0x2ecc71 if sig['verdict'] == "BUY" else 0xe74c3c,
                        "fields": [
                            {"name": "Entry Price", "value": f"${sig['entry']:,.2f}", "inline": True},
                            {"name": "Volatility (ATR)", "value": f"${sig['atr']}", "inline": True},
                            {"name": "Elite Stop-Loss", "value": f"**${sig['stop_loss']:,.2f}**", "inline": False},
                            {"name": "Elite Take-Profit", "value": f"**${sig['take_profit']:,.2f}**", "inline": False},
                        ],
                        "footer": {"text": "TFT-Lite Architecture • Absolute Path Verified"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️ {pair}: Neutral")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_pro_engine()
