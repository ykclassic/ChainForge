import os
import ccxt
import pandas as pd
import pandas_ta as ta
from discord_webhook import DiscordWebhook, DiscordEmbed

def fetch_data(symbol, timeframe='1h', limit=100):
    """Fetch historical OHLCV data from Bitget."""
    exchange = ccxt.bitget()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def analyze_signals(df):
    """Calculate RSI and Bollinger Bands to identify market status."""
    # Add Technical Indicators
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    
    last_row = df.iloc[-1]
    close = last_row['close']
    rsi = last_row['RSI_14']
    upper_bb = last_row['BBU_20_2.0']
    lower_bb = last_row['BBL_20_2.0']

    # Simple logic for the report
    status = "Neutral"
    color = "808080"  # Gray

    if rsi > 70 or close > upper_bb:
        status = "Overbought / Potential Short"
        color = "ff0000"  # Red
    elif rsi < 30 or close < lower_bb:
        status = "Oversold / Potential Long"
        color = "00ff00"  # Green

    return {
        "price": close,
        "rsi": round(rsi, 2),
        "status": status,
        "color": color
    }

def send_discord_report(results):
    """Send the backtest summary to Discord via Webhook."""
    webhook_url = os.getenv('DISCORD_WEBHOOK')
    if not webhook_url:
        print("Discord Webhook URL not found in environment.")
        return

    webhook = DiscordWebhook(url=webhook_url)
    embed = DiscordEmbed(
        title="📊 Daily Quant Backtest Report",
        description="Automated market scan results for BTC/USDT",
        color=results['color']
    )
    
    embed.add_embed_field(name="Current Price", value=f"${results['price']}")
    embed.add_embed_field(name="RSI (14)", value=f"{results['rsi']}")
    embed.add_embed_field(name="Market Status", value=results['status'])
    embed.set_timestamp()

    webhook.add_embed(embed)
    webhook.execute()

if __name__ == "__main__":
    print("Starting automated backtest cycle...")
    data = fetch_data('BTC/USDT')
    
    if data is not None:
        analysis = analyze_signals(data)
        send_discord_report(analysis)
        print(f"Cycle complete. Status: {analysis['status']}")
