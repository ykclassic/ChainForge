import requests
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_score(pair):
    """
    Fetches news from CryptoPanic and calculates a compound sentiment score using VADER.
    """
    try:
        # Extract base currency (e.g., BTC from BTC/USDT)
        base = pair.split('/')[0].upper()
        
        # CryptoPanic API call (Using limit=20 for better sample size)
        url = f"https://cryptopanic.com/api/v1/posts/?public=true&currencies={base}&kind=news&limit=20"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return 0.0
            
        data = response.json()
        news = data.get('results', [])
        
        if not news:
            return 0.0

        analyzer = SentimentIntensityAnalyzer()
        titles = [article['title'] for article in news]
        
        # Calculate compound scores for each headline
        scores = [analyzer.polarity_scores(title)['compound'] for title in titles]
        
        # Average and scale from 0 to 100 for easier thresholding
        avg_score = np.mean(scores) * 10
        return round(float(avg_score), 2)
        
    except Exception as e:
        print(f"⚠️ Sentiment Engine Error: {e}")
        return 0.0
