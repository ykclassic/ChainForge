import requests
from textblob import TextBlob  # Add 'textblob' to requirements.txt

def get_sentiment_score(pair):
    """Simple sentiment from recent news titles (TextBlob polarity)."""
    try:
        base = pair.split('/')[0]
        news = requests.get(f"https://cryptopanic.com/api/v1/posts/?public=true&currencies={base}&kind=news&limit=20").json()['results']
        titles = [article['title'] for article in news]
        scores = [TextBlob(title).sentiment.polarity for title in titles]
        avg_score = np.mean(scores) * 100  # -100 to 100
        return round(avg_score, 2)
    except:
        return 'N/A'
