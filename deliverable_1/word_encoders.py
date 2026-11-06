"""Word unigram/bigram encoders compatible with the experiment functions."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from part1.clslib import Encoder


class WordEncoder(Encoder):
    def __init__(self, weighting):
        options = dict(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            stop_words=None,
        )
        if weighting == "tfidf":
            self.vectorizer = TfidfVectorizer(**options)
        else:
            self.vectorizer = CountVectorizer(
                binary=(weighting == "indicator"), **options
            )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def __len__(self):
        return len(self.vectorizer.vocabulary_)

    def __call__(self, sentence):
        return self.vectorizer.transform([sentence]).toarray()[0]


class WordBoW(WordEncoder):
    def __init__(self):
        super().__init__("count")


class WordTokenIndicator(WordEncoder):
    def __init__(self):
        super().__init__("indicator")


class WordTFIDF(WordEncoder):
    def __init__(self):
        super().__init__("tfidf")
