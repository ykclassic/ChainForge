import requests
from textblob import TextBlob  # Add 'textblob' to requirements.txt
textblob.download_corpora_lite()

def get_sentiment_score(pair):
    try:
        base = pair.split('/')[0].upper()  # Ensure uppercase for CryptoPanic
        news = requests.get(f"https://cryptopanic.com/api/v1/posts/?public=true¤cies={base}&kind=news&limit=20").json()['results']
        # ... rest unchanged        titles = [article['title'] for article in news]
        scores = [TextBlob(title).sentiment.polarity for title in titles]
        avg_score = np.mean(scores) * 100  # -100 to 100
        return round(avg_score, 2)
    except:
        return 'N/A'
