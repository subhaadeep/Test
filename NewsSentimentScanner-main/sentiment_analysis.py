import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import quote

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

labels = ['Positive', 'Negative', 'Neutral']

def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]

    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)
        
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.RequestException:
        return "Content not retrieved."

def analyze_sentiment(text):

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']

    # analysis = TextBlob(text)
    # polarity = analysis.sentiment.polarity

    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment

# def analyze_sentiment(text):
#     if not text.strip():
#         return 0.0, 'Neutral'

#     inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = finbert_model(**inputs)

#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1).numpy()[0]
#     max_index = np.argmax(probabilities)
#     sentiment = labels[max_index]
#     confidence = probabilities[max_index]

#     return confidence, sentiment


def summarize_sentiments(articles):
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for article in articles:
        # print("-"*25)
        # print(f"\n--- Analyzing Article: {article['title']} ---")
        # print(f"Published: {article['published']}")
        # print(article['content'])
        _, sentiment = analyze_sentiment(article['title']) # + " " + article['content'])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")
    for sentiment, count in summary.items():
        percent = (count / total) * 100
        print(f"{sentiment}: {count} ({percent:.2f}%)")

def main():
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]
    num_articles_per_query = 10
    all_articles = []

    for query in queries:
        print(f"Fetching news articles for '{query}'...\n")
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)

    for idx, article in enumerate(all_articles, 1):
        print(f"Article {idx}: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")

        polarity, sentiment = analyze_sentiment(article['title'])  # or article['content']
        print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})\n")

    summarize_sentiments(all_articles)

if __name__ == "__main__":
    main()