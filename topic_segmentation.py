from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

def segment_topics(text):

    sentences = sent_tokenize(text)

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=3)

    labels = kmeans.fit_predict(X)

    segments = []

    for i, s in enumerate(sentences):

        segments.append({
            "sentence": s,
            "topic": int(labels[i])
        })

    return segments