import requests
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_score(pair):
    """
    Fetches news from CryptoPanic and calculates a compound sentiment score using VADER.
    """
    try:
        base = pair.split('/')[0].upper()
        # CryptoPanic API call
        url = f"https://cryptopanic.com/api/v1/posts/?public=true&currencies={base}&kind=news&limit=20"
        response = requests.get(url, timeout=10).json()
        news = response.get('results', [])
        
        if not news:
            return 0.0

        analyzer = SentimentIntensityAnalyzer()
        titles = [article['title'] for article in news]
        scores = [analyzer.polarity_scores(title)['compound'] for title in titles]
        
        avg_score = np.mean(scores) * 100  # Scale to -100 to 100
        return round(avg_score, 2)
    except Exception as e:
        return 0.0
