import os
import ccxt
import pandas as pd
import pandas_ta as ta
from discord_webhook import DiscordWebhook, DiscordEmbed

# Assets to scan
ASSETS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "SOL/USDT", "LINK/USDT", "SUI/USDT"]

def fetch_data(symbol):
    exchange = ccxt.bitget()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=150)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def generate_signals(df):
    # Indicators
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.aroon(length=14, append=True)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Values
    close = last['close']
    rsi = last['RSI_14']
    macd_line = last['MACD_12_26_9']
    macd_sig = last['MACDs_12_26_9']
    aroon_osc = last['AROONOSC_14']
    
    # Logic
    verdict = "NEUTRAL"
    color = "808080"
    
    # BUY: RSI < 35 AND MACD Cross Up AND Aroon > 0
    if rsi < 35 and (macd_line > macd_sig and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']) and aroon_osc > 0:
        verdict = "BUY / LONG"
        color = "00FF00"
    # SELL: RSI > 65 AND MACD Cross Down AND Aroon < 0
    elif rsi > 65 and (macd_line < macd_sig and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']) and aroon_osc < 0:
        verdict = "SELL / SHORT"
        color = "FF0000"

    # Build Diagnostic String
    macd_diff = macd_line - macd_sig
    diag = (f"RSI: {rsi:.1f} | "
            f"MACD: {'🟢' if macd_diff > 0 else '🔴'} ({macd_diff:.4f}) | "
            f"Aroon: {aroon_osc:.0f}")

    return {"verdict": verdict, "price": f"{close:,.4f}", "diag": diag, "color": color}

def send_to_discord(all_results):
    webhook_url = os.getenv('DISCORD_WEBHOOK')
    if not webhook_url: return
    
    webhook = DiscordWebhook(url=webhook_url)
    embed = DiscordEmbed(title="🚀 Multi-Asset Quant Scan (with Diagnostics)", color="03a9f4")
    
    for asset, res in all_results.items():
        # Use code blocks for cleaner alignment in Discord
        val = f"Price: **{res['price']}**\n`{res['diag']}`"
        embed.add_embed_field(name=f"{asset}: {res['verdict']}", value=val, inline=False)
    
    webhook.add_embed(embed)
    webhook.execute()

if __name__ == "__main__":
    report = {asset: generate_signals(df) for asset in ASSETS if (df := fetch_data(asset)) is not None}
    send_to_discord(report)
