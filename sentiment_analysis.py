# sentiment_analysis.py
import praw
from textblob import TextBlob
import numpy as np
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

def get_reddit_sentiment(subreddit, query, start_date, end_date):
    sentiments = []
    for submission in reddit.subreddit(subreddit).search(query, time_filter='all', sort='new'):
        sentiment = TextBlob(submission.title).sentiment.polarity
        sentiments.append(sentiment)
    return np.mean(sentiments) if sentiments else 0
