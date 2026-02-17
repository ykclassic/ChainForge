import pandas as pd
import numpy as np
import ccxt
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from modules.quant_signal import generate_institutional_signal

def get_real_sentiment(pair):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url).json()
        articles = response.get('Data', [])
        coin = pair.split('/')[0]
        relevant = [a['title'] for a in articles if coin in a['title']]
        if not relevant: relevant = [a['title'] for a in articles[:8]]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in relevant]
        return round(np.mean(scores), 2) if scores else 0.0
    except: return 0.0

def train_and_predict(df, sentiment):
    """Headless LSTM prediction logic."""
    data = df[['close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data) # Simplified for hourly runs
    
    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i, 0]); y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([LSTM(32, input_shape=(10, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    
    last_seq = scaled[-10:].reshape(1, 10, 1)
    raw_pred = scaler.inverse_transform(model.predict(last_seq))[0][0]
    return raw_pred * (1 + (sentiment * 0.015))

def run_worker():
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    ex = ccxt.bitget()

    for pair in assets:
        # Fetch Data
        ohlcv = ex.fetch_ohlcv(pair, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        
        # Calculate Signals
        sentiment = get_real_sentiment(pair)
        prediction = train_and_predict(df, sentiment)
        vol = df['close'].pct_change().std()
        
        ob = ex.fetch_order_book(pair, limit=20)
        imbalance = (sum([x[1] for x in ob['bids']]) - sum([x[1] for x in ob['asks']])) / \
                    (sum([x[1] for x in ob['bids']]) + sum([x[1] for x in ob['asks']]))
        
        sig = generate_institutional_signal(df['close'].iloc[-1], prediction, sentiment, imbalance, vol)
        
        if sig['verdict'] != "NEUTRAL":
            payload = {
                "embeds": [{
                    "title": f"🚀 {sig['verdict']} ALERT: {pair}",
                    "color": 65280 if sig['verdict'] == "BUY" else 16711680,
                    "fields": [
                        {"name": "Entry", "value": f"${sig['entry']:,}", "inline": True},
                        {"name": "Target", "value": f"${sig['take_profit']:,}", "inline": True},
                        {"name": "Stop Loss", "value": f"${sig['stop_loss']:,}", "inline": True},
                        {"name": "Confidence", "value": f"{sig['confidence']}%", "inline": False}
                    ]
                }]
            }
            requests.post(webhook_url, json=payload)

if __name__ == "__main__":
    run_worker()
