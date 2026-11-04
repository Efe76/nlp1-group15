import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
 
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
 
import part1
from part1.clslib import Encoder

 
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
TOKENIZER_PATH = RESULTS_DIR / "sst2_tokenizer.json"
 
CLASSES = part1.clslib.NamedClasses(['negative', 'positive'])
 
VOCAB_SIZE = 5000
MIN_FREQUENCY = 10
 
NUM_FOLDS = 5
SEEDS = [0]
 
SPLITS = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
BASE = "hf://datasets/stanfordnlp/sst2/"
 
 
def to_pairs(df):
    return list(zip(df["sentence"], df["label"].astype(int)))
 
 
def load_sst2():
    train = to_pairs(pd.read_parquet(BASE + SPLITS["train"]))
    validation = to_pairs(pd.read_parquet(BASE + SPLITS["validation"]))
    test = to_pairs(pd.read_parquet(BASE + SPLITS["test"]))
    return train, validation, test
 
 
def get_tokenizer(train):
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    corpus = [text for text, label in train]
    tokenizer = part1.experiment.train_tokenizer(
        corpus=corpus,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer
 
 
def get_cv_splits(pairs):
    return [part1.experiment.cross_validation_split(pairs, k=NUM_FOLDS, seed=seed) for seed in SEEDS]

# ENCODER 1
class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector

# ENCODER 2    
class BagOfWords(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def __call__(self, x):
        token_ids = self.tokenizer.encode(x).ids
        vector = np.zeros(len(self))
        for token_id in token_ids:
            vector[token_id] += 1
        return vector

# ENCODER 3
# ENCODER 3
class TFIDFbigram(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab = {}
        self.idf = None

    def __len__(self):
        return len(self.vocab)

    def get_bigrams(self, text):
        token_ids = self.tokenizer.encode(text).ids
        return list(zip(token_ids, token_ids[1:]))

    def fit(self, texts):
        document_freq = {}
        n_documents = 0

        for sentences in texts:
            n_documents += 1

            for bigram in set(self.get_bigrams(sentences)):
                document_freq[bigram] = document_freq.get(bigram, 0 ) +1

        self.vocab = {bigram: idx for idx, bigram in enumerate(sorted(document_freq))}
        self.idf = np.zeros(len(self))

        for bigram, index in self.vocab.items():
            self.idf[index] = np.log((n_documents + 1) / (document_freq[bigram] + 1 )) + 1
        
        return self
    
    def __call__(self, sentence):
        tf_idf = np.zeros(len(self))

        for bigram in self.get_bigrams(sentence):
            index = self.vocab.get(bigram)
            if index is not None:
                tf_idf[index] +=1

        tf_idf *= self.idf

        norm = np.linalg.norm(tf_idf)
        if norm > 0:
            tf_idf /= norm

        return tf_idf


class TFIDFunigram(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab = {}
        self.idf = None

    def __len__(self):
        return len(self.vocab)

    def get_unigrams(self, text):
     return self.tokenizer.encode(text).ids

    def fit(self, texts):
        document_freq = {}
        n_documents = 0

        for sentences in texts:
            n_documents += 1

            for token_id in set(self.get_unigrams(sentences)):
                document_freq[token_id] = document_freq.get(token_id, 0) + 1

        self.vocab = {
            token_id: idx
            for idx, token_id in enumerate(sorted(document_freq))
        }
        self.idf = np.zeros(len(self))

        for token_id, index in self.vocab.items():
            self.idf[index] = (
                np.log((n_documents + 1) / (document_freq[token_id] + 1)) + 1
            )
        
        return self
    
    def __call__(self, sentence):
        tf_idf = np.zeros(len(self))

        for token_id in self.get_unigrams(sentence):
            index = self.vocab.get(token_id)
            if index is not None:
                tf_idf[index] += 1

        tf_idf *= self.idf

        norm = np.linalg.norm(tf_idf)
        if norm > 0:
            tf_idf /= norm

        return tf_idf