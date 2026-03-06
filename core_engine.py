import os
import ccxt
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from openai import OpenAI

# Hardcoded dummy keys for development/testing
DUMMY_DISCORD_WEBHOOK = "https://discord.com/api/webhooks/123456789/dummy-token"
DUMMY_OPENAI_KEY = "sk-dummy-openai-key-12345"
DUMMY_GEMINI_KEY = "ai-dummy-gemini-key-67890"
DUMMY_CRYPTOPANIC_KEY = "cp-dummy-key-09876"

# --- 1. DATA & METRICS ENGINES ---

def get_real_sentiment(pair):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url).json()
        articles = response.get('Data', [])
        coin = pair.split('/')[0]
        relevant = [a['title'] for a in articles if coin in a['title'] or coin in a['categories']]
        if not relevant: relevant = [a['title'] for a in articles[:8]]
        
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in relevant]
        return round(np.mean(scores), 2) if scores else 0.0, relevant[:5]
    except Exception: 
        return 0.0, ["News feed refreshing..."]

def get_sentiment_score(ticker):
    ticker_clean = ticker.split('/')[0].upper()
    total_polarity = 0
    article_count = 0
    
    cp_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={DUMMY_CRYPTOPANIC_KEY}&currencies={ticker_clean}"
    try:
        res = requests.get(cp_url, timeout=5).json()
        for post in res.get('results', []):
            blob = TextBlob(post['title'])
            total_polarity += blob.sentiment.polarity
            article_count += 1
    except Exception:
        pass

    if article_count < 3:
        try:
            fs_url = f"https://free-crypto-news.vercel.app/api/search?q={ticker_clean}&limit=10"
            res = requests.get(fs_url, timeout=5).json()
            for article in res.get('articles', []):
                blob = TextBlob(article.get('title', ''))
                total_polarity += blob.sentiment.polarity
                article_count += 1
        except Exception:
            pass
            
    return round(total_polarity / article_count, 2) if article_count > 0 else 0.0

def get_imbalance(pair):
    try:
        exchange = ccxt.bitget()
        order_book = exchange.fetch_order_book(pair, limit=20)
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks: return 0.0
        
        weighted_bids = sum(volume / (i + 1) for i, (price, volume) in enumerate(bids))
        weighted_asks = sum(volume / (i + 1) for i, (price, volume) in enumerate(asks))
        
        total_v = weighted_bids + weighted_asks
        if total_v == 0: return 0.0
        
        return round(float((weighted_bids - weighted_asks) / total_v), 4)
    except Exception as e:
        print(f"OBI Engine Error for {pair}: {e}")
        return 0.0

def get_on_chain_metrics(coin_id: str):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true&sparkline=false"
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return {'error': 'API Limit or Invalid ID'}
            
        data = response.json()
        return {
            'market_cap_rank': data.get('market_data', {}).get('market_cap_rank', 'N/A'),
            'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0),
            'total_supply': data.get('market_data', {}).get('total_supply', 0),
            'twitter_followers': data.get('community_data', {}).get('twitter_followers', 0),
            'github_stars': data.get('developer_data', {}).get('stars', 0),
        }
    except Exception:
        return {'error': 'Service Unavailable'}

# --- 2. AI & ML ENGINES ---

class GatedLinearUnit(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")
    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.elu = layers.Dense(units, activation="elu")
        self.dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.projector = None
    def build(self, input_shape):
        if input_shape[-1] != self.units: self.projector = layers.Dense(self.units)
        super(GatedResidualNetwork, self).build(input_shape)
    def call(self, inputs, training=False):
        x = self.elu(inputs)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.glu(x)
        residual = self.projector(inputs) if self.projector else inputs
        return self.layer_norm(x + residual)

def process_ai_query(query: str, context: dict):
    try:
        client = OpenAI(api_key=DUMMY_OPENAI_KEY)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Context: {context}"}, {"role": "user", "content": query}],
            timeout=10
        )
        return res.choices[0].message.content
    except Exception: pass

    try:
        genai.configure(api_key=DUMMY_GEMINI_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(f"Context: {context}\nQuery: {query}")
        return res.text
    except Exception: pass

    rsi = context.get('rsi', 50)
    bb_status = context.get('bb_status', "Neutral")
    if rsi < 35 and "Lower" in str(bb_status):
        return "### 🛡️ Heuristic Report\n**Bias:** STRONGLY BULLISH\n**Analysis:** Oversold conditions met with volatility band contact."
    elif rsi > 65 and "Upper" in str(bb_status):
        return "### 🛡️ Heuristic Report\n**Bias:** STRONGLY BEARISH\n**Analysis:** Overbought conditions met with upper band contact."
    return f"### 🛡️ Heuristic Report\n**Bias:** NEUTRAL\n**Analysis:** RSI at {rsi:.1f} is within balanced territory."

# --- 3. SIGNAL GENERATION (Strictly Generative, No Execution) ---

def generate_institutional_signal(current_price, predicted_price, sentiment_score, imbalance, volatility):
    expected_change = ((predicted_price / current_price) - 1) * 100
    verdict = "NEUTRAL"
    
    if expected_change > 0.5 and sentiment_score > 0.1 and imbalance > 0.15: verdict = "BUY"
    elif expected_change < -0.5 and sentiment_score < -0.1 and imbalance < -0.15: verdict = "SELL"
    
    if verdict == "NEUTRAL":
        return {"verdict": "NEUTRAL", "entry": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0}
        
    sl_pct = max(volatility * 1.5, 0.01) 
    tp_pct = sl_pct * 2.5 
    
    entry = current_price
    stop_loss = entry * (1 - sl_pct) if verdict == "BUY" else entry * (1 + sl_pct)
    take_profit = entry * (1 + tp_pct) if verdict == "BUY" else entry * (1 - tp_pct)
    
    confidence = min(round((abs(expected_change) * 10 + abs(sentiment_score) * 20 + abs(imbalance) * 20), 1), 100.0)
    
    return {
        "verdict": verdict, "entry": round(entry, 2), "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2), "confidence": confidence, "rr_ratio": "1:2.5"
    }

def generate_pro_signal(df, sentiment=0.0, imbalance=0.0):
    try:
        lookback = 50 
        if len(df) < lookback: return {"verdict": "NEUTRAL", "trend_raw": 0.0}
        
        prices_list = df['close'].tolist()
        current_price = float(prices_list[-1])
        start_price = float(prices_list[0]) 
        
        df_proc = df.copy()
        df_proc['returns'] = df_proc['close'].pct_change()
        raw_data = df_proc[['returns', 'vol']].tail(lookback).fillna(0).values
        input_tensor = tf.expand_dims(tf.convert_to_tensor(raw_data, dtype=tf.float32), 0)
        
        model_layer = GatedResidualNetwork(units=32)
        processed = model_layer(input_tensor)
        
        ai_trend = float(tf.reduce_mean(processed[:, -5:, :]).numpy())
        price_velocity = (current_price - start_price) / start_price if start_price != 0 else 0
        combined_trend = (ai_trend * 0.2) + (price_velocity * 0.8)
        
        verdict = "NEUTRAL"
        if combined_trend > 0.005 and (imbalance > 0.015 or sentiment > 0.05): verdict = "BUY"
        elif combined_trend < -0.005 and (imbalance < -0.015 or sentiment < -0.05): verdict = "SELL"
        
        high_low = df['high'] - df['low']
        atr = float(high_low.rolling(14).mean().tolist()[-1])
        
        return {
            "verdict": verdict, "entry": current_price, "atr": round(atr, 2), "trend_raw": round(combined_trend, 6),
            "stop_loss": current_price - (atr * 2) if verdict == "BUY" else current_price + (atr * 2),
            "take_profit": current_price + (atr * 4) if verdict == "BUY" else current_price - (atr * 4)
        }
    except Exception as e:
        print(f"Pro Logic Error: {e}")
        return {"verdict": "NEUTRAL", "trend_raw": 0.0}

# --- 4. BACKTESTING ENGINE ---

def run_backtest(df, initial_capital=10000, risk_free_rate=0.02):
    data = df.copy()
    data['signal'] = 0
    
    if 'lower_band' in data.columns and 'upper_band' in data.columns and 'rsi' in data.columns:
        data.loc[(data['close'] <= data['lower_band']) & (data['rsi'] < 30), 'signal'] = 1
        data.loc[(data['close'] >= data['upper_band']) | (data['rsi'] > 70), 'signal'] = -1
    
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0) 
    data['pct_change'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'].shift(1) * data['pct_change']
    data['equity_curve'] = initial_capital * (1 + data['strategy_returns']).fillna(0).cumprod()
    
    total_return = (data['equity_curve'].iloc[-1] / initial_capital - 1)
    daily_returns = data['strategy_returns'].dropna()
    
    sharpe_ratio = 0.0
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        excess_returns = daily_returns - (risk_free_rate / 365)
        sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / daily_returns.std())
        
    running_max = data['equity_curve'].cummax()
    drawdown = (data['equity_curve'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    trades = data[data['strategy_returns'] != 0]
    win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0
    
    return {
        "df": data, "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2), "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2), "final_value": round(data['equity_curve'].iloc[-1], 2)
    }

def send_discord_alert(webhook_url, content, embed=None):
    payload = {"content": content}
    if embed: payload["embeds"] = [embed]
    try: requests.post(webhook_url, json=payload)
    except: pass
