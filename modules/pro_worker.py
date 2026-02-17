import os
import sys

# 1. Get the absolute path of the directory containing this script (modules/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Add 'modules/' to the system path so we can import pro_signal directly
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 3. Add the project root to the system path as a fallback
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import ccxt
import pandas as pd
import requests

# 4. CRITICAL: Import WITHOUT the 'modules.' prefix
# Since we added current_dir to sys.path, Python sees pro_signal.py directly.
try:
    from pro_signal import generate_pro_signal
except ImportError:
    # Fallback for different runner environments
    from modules.pro_signal import generate_pro_signal

def run_pro_engine():
    # ... (rest of your existing code)
    webhook = os.getenv("DISCORD_WEBHOOK")
    ex = ccxt.bitget()
    # Expanding to more assets as discussed
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

    print(f"💎 PRO ENGINE START: {len(assets)} assets")

    for pair in assets:
        try:
            ohlcv = ex.fetch_ohlcv(pair, '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Fetch dummy sentiment/imbalance for example (Connect your API here)
            # In your real setup, these will come from whale_alert or news APIs
            sentiment = 0.2 
            imbalance = 0.25 
            
            sig = generate_pro_signal(df, sentiment, imbalance)

            if sig['verdict'] != "NEUTRAL":
                print(f"🔥 PRO SIGNAL: {pair} {sig['verdict']}")
                payload = {
                    "embeds": [{
                        "title": f"💎 PRO {sig['verdict']} ALERT: {pair}",
                        "color": 0x00FF00 if sig['verdict'] == "BUY" else 0xFF0000,
                        "fields": [
                            {"name": "Entry", "value": f"${sig['entry']:,.2f}", "inline": True},
                            {"name": "Volatility (ATR)", "value": f"${sig['atr_volatility']}", "inline": True},
                            {"name": "Dynamic SL", "value": f"${sig['stop_loss']:,.2f}", "inline": False},
                            {"name": "Dynamic TP", "value": f"${sig['take_profit']:,.2f}", "inline": False},
                        ],
                        "footer": {"text": "ChainForge PRO • TFT & ATR Engine v1.0"}
                    }]
                }
                requests.post(webhook, json=payload)
        except Exception as e:
            print(f"Error {pair}: {e}")

if __name__ == "__main__":
    run_pro_engine()
