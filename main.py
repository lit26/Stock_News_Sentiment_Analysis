import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from finvizfinance.quote import finvizfinance

stock = finvizfinance('tsla')
df = stock.TickerNews()
df = df[['Date','Title']]

senti = SentimentIntensityAnalyzer()
def get_sentiment_score(review):
    try:
        compound_score = senti.polarity_scores(review)['compound']
        return compound_score
    except:
        return "error"
df["Sentiment_score"] = df["Title"].apply(get_sentiment_score)
df2 = df[df['Sentiment_score']!=0]
df2 = df2.reset_index(drop=True)

plt.figure(figsize=(10,8))
plt.plot('Date', 'Sentiment_score', data=df2)
plt.xticks(rotation=50)
plt.show()