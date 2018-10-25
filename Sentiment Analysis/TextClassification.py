import re
from textblob import TextBlob

def tweet_preprocessing(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    tweet = tweet_preprocessing(tweet)
    blob = TextBlob(tweet)
    if blob.sentiment.polarity > 0:
        return 'Positive'
    elif blob.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

tweet="#IBULISL may go up 43% if the bull pattern is confirmed. Chk confirmation [38%] @ http://bit.ly/2uvCUaB . Buy level:428.1000."
classify=get_tweet_sentiment(tweet)

print(classify)
