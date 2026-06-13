"""
preprocessing.py
----------------
Text cleaning and normalization for the Fake News Detection system.

Implements the NLP pipeline described in the thesis (Chapter I & 2.1):
    - lowercasing
    - removal of URLs, HTML, punctuation, numbers
    - tokenization
    - stopword removal
    - lemmatization

The same `clean_text` function is used both at TRAINING time (train.py)
and at PREDICTION time (app.py), so the input the model sees is always
processed in exactly the same way.
"""

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources():
    """Download the NLTK data we need, only if it isn't already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

# Build these once at import time (expensive to recreate per call).
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    """Lemmatize a single token, cached so repeated words are instant."""
    return LEMMATIZER.lemmatize(token)

# Pre-compiled regular expressions for speed.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize a single news article / headline into a clean token string.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Space-separated, lemmatized tokens ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    # Drop punctuation, then anything that isn't a letter or whitespace
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        _lemma(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    # Quick manual sanity check
    sample = "BREAKING!!! Scientists at <b>NASA</b> confirm aliens at http://fake.news in 2024."
    print("RAW :", sample)
    print("CLEAN:", clean_text(sample))
