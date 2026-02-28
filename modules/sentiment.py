import requests
import os
from textblob import TextBlob

def get_sentiment_score(ticker):
    """
    Combines news from CryptoPanic and a Failsafe RSS Aggregator.
    Returns a score between -1.0 (Bearish) and 1.0 (Bullish).
    """
    ticker_clean = ticker.split('/').upper()
    total_polarity = 0
    article_count = 0
    
    # --- Source 1: CryptoPanic (High Quality) ---
    cp_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={os.getenv('CRYPTOPANIC_KEY')}&currencies={ticker_clean}"
    try:
        res = requests.get(cp_url, timeout=5).json()
        for post in res.get('results', []):
            blob = TextBlob(post['title'])
            total_polarity += blob.sentiment.polarity
            article_count += 1
    except Exception:
        pass

    # --- Source 2: Failsafe (Volume) ---
    # If Source 1 is empty or fails, we use a public aggregator
    if article_count < 3:
        try:
            # Using the 2026 Free-Crypto-News endpoint
            fs_url = f"https://free-crypto-news.vercel.app/api/search?q={ticker_clean}&limit=10"
            res = requests.get(fs_url, timeout=5).json()
            for article in res.get('articles', []):
                blob = TextBlob(article['title'])
                total_polarity += blob.sentiment.polarity
                article_count += 1
        except Exception:
            pass

    if article_count == 0:
        return 0.0
    
    # Return rounded score with higher precision to avoid 0.00
    return round(total_polarity / article_count, 4)
