import os
import sys

# --- PATH FIX: Ensures script can see other files in the modules/ folder ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Also add project root for GitHub Actions environment consistency
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import ccxt
import pandas as pd
import requests
from datetime import datetime

# Import local logic - no "modules." prefix needed due to sys.path append
# If you eventually move pro logic here, you'd use: from pro_signal import generate_pro_signal
# For now, we maintain the standard signal logic.

def run_standard_engine():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK not found.")
        return

    # 1. Initialize Exchange (Bitget)
    ex = ccxt.bitget()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Starting Standard Engine Analysis...")

    for pair in assets:
        try:
            # 2. Fetch Market Data
            ohlcv = ex.fetch_ohlcv(pair, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 3. Core Logic (Calculated, Not Hardcoded)
            # Simple Trend: Current price vs 24h average
            current_price = df['close'].iloc[-1]
            avg_price = df['close'].mean()
            price_change_pct = ((current_price - avg_price) / avg_price) * 100
            
            # 4. Signal Decision
            verdict = "NEUTRAL"
            color = 0x95a5a6 # Gray
            
            if price_change_pct > 0.5:
                verdict = "BUY"
                color = 0x2ecc71 # Green
            elif price_change_pct < -0.5:
                verdict = "SELL"
                color = 0xe74c3c # Red

            # 5. Discord Dispatch
            if verdict != "NEUTRAL":
                print(f"🚀 SIGNAL FOUND for {pair}: {verdict}")
                payload = {
                    "username": "ChainForge Standard",
                    "embeds": [{
                        "title": f"📊 {verdict} Signal | {pair}",
                        "color": color,
                        "fields": [
                            {"name": "Price", "value": f"${current_price:,.2f}", "inline": True},
                            {"name": "24h Avg", "value": f"${avg_price:,.2f}", "inline": True},
                            {"name": "Deviaton", "value": f"{price_change_pct:.2f}%", "inline": True}
                        ],
                        "footer": {"text": "Standard Analysis Engine • 2026 Stable"}
                    }]
                }
                requests.post(webhook_url, json=payload)
            else:
                print(f"⏸️  {pair}: Market Neutral ({price_change_pct:.2f}%)")

        except Exception as e:
            print(f"⚠️  Error processing {pair}: {e}")

if __name__ == "__main__":
    run_standard_engine()
