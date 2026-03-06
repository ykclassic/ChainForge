import requests
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def crypto_news_sentiment(pair):

    coin = pair.split("/")[0]

    try:

        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
            timeout=10
        )

        data = r.json()["Data"]

        titles = [x["title"] for x in data if coin in x["title"]]

        if not titles:
            titles = [x["title"] for x in data[:8]]

        scores = [analyzer.polarity_scores(t)["compound"] for t in titles]

        return np.mean(scores)

    except:

        return 0
