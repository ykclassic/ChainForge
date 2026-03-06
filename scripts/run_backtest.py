import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from discord_webhook import DiscordWebhook, DiscordEmbed

# --- CONFIGURATION ---
ASSETS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "SOL/USDT", "LINK/USDT", "SUI/USDT"]
TIMEFRAME = '1h'

def fetch_data(symbol):
    exchange = ccxt.bitget()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=150)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None

def generate_signals(df):
    """Fuses RSI, MACD, Aroon, and ATR per user preferences."""
    # 1. Indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.aroon(length=14, append=True)
    df.ta.atr(length=14, append=True)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 2. Extract Values
    close = last['close']
    rsi = last['RSI_14']
    macd_line = last['MACD_12_26_9']
    macd_sig = last['MACDs_12_26_9']
    aroon_osc = last['AROONOSC_14']
    atr = last['ATRr_14']
    
    # 3. Logic: RSI (35/65) + MACD Crossover + Aroon Status
    verdict = "NEUTRAL"
    color = "808080" # Gray
    
    # BULLISH: RSI < 35 AND MACD Cross Up AND Aroon > 0
    if rsi < 35 and (macd_line > macd_sig and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']) and aroon_osc > 0:
        verdict = "BUY / LONG"
        color = "00FF00" # Green
    
    # BEARISH: RSI > 65 AND MACD Cross Down AND Aroon < 0
    elif rsi > 65 and (macd_line < macd_sig and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']) and aroon_osc < 0:
        verdict = "SELL / SHORT"
        color = "FF0000" # Red

    # 4. Risk Management: 2x ATR Stop Loss, 1.5x Risk Take Profit
    sl_dist = atr * 2
    tp_dist = sl_dist * 1.5
    
    if verdict == "BUY / LONG":
        sl, tp = close - sl_dist, close + tp_dist
    elif verdict == "SELL / SHORT":
        sl, tp = close + sl_dist, close - tp_dist
    else:
        sl, tp = 0, 0

    return {
        "verdict": verdict, "price": f"{close:,.4f}", "rsi": round(rsi, 1),
        "sl": f"{sl:,.4f}", "tp": f"{tp:,.4f}", "color": color
    }

def send_to_discord(all_results):
    webhook_url = os.getenv('DISCORD_WEBHOOK')
    if not webhook_url: return
    
    webhook = DiscordWebhook(url=webhook_url)
    embed = DiscordEmbed(title="🚀 Multi-Asset Quant Scan", color="03a9f4")
    
    for asset, res in all_results.items():
        val = f"Price: {res['price']} | RSI: {res['rsi']}\n"
        if res['verdict'] != "NEUTRAL":
            val += f"**SL:** {res['sl']} | **TP:** {res['tp']}"
        
        embed.add_embed_field(name=f"{asset}: {res['verdict']}", value=val, inline=False)
    
    webhook.add_embed(embed)
    webhook.execute()

if __name__ == "__main__":
    report = {}
    for asset in ASSETS:
        df = fetch_data(asset)
        if df is not None:
            report[asset] = generate_signals(df)
    
    send_to_discord(report)
    print("✅ Full market scan sent to Discord.")
